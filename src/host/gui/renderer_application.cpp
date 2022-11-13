#include <stdexcept>
#include <iostream>
#include <cmath>

#include <GL/glew.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

#include <imgui.h>
#include <imgui_impl_sdl.h>
#include <imgui_impl_opengl2.h>

#include "host/bitmap/bitmap.h"
#include "renderer_application.h"

namespace Xrender
{
    renderer_application::renderer_application(const render_configuration& config)
    {
        static bool sdl_was_init = false;
        if (!sdl_was_init) {
            SDL_SetHint(SDL_HINT_NO_SIGNAL_HANDLERS, "1");
            SDL_Init(SDL_INIT_VIDEO);
            sdl_was_init = true;
        }

        const auto width = config.camera_config.image_width;
        const auto height = config.camera_config.image_height;

        const auto windows_flags = SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI;
        _window = SDL_CreateWindow("Xrender", 0, 0, width, height, windows_flags);

        // Hide the windows until the render is not ready to start
        SDL_HideWindow(_window);

        _gl_context = SDL_GL_CreateContext(_window);
        SDL_GL_MakeCurrent(_window, _gl_context);

        GLenum glew_error = glewInit();
        if (glew_error != GLEW_OK)
        {
            throw std::runtime_error(reinterpret_cast<const char*>(glewGetErrorString(glew_error)));
        }

        // Setup windows
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
        SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);

        // Create openGL texture to develop the rendered image into
        glGenTextures(1, &_texture);
        glBindTexture(GL_TEXTURE_2D, _texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glEnable(GL_TEXTURE_2D);

        // Setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();
        ImGui_ImplSDL2_InitForOpenGL(_window, _gl_context);
        ImGui_ImplOpenGL2_Init();

        // Initialize the renderer and the gui
        _renderer = renderer_frontend::build_renderer_frontend(config, _texture);
        _gui = std::make_unique<renderer_gui>(*_renderer);

        SDL_ShowWindow(_window);
        _update_size();
    }

    renderer_application::~renderer_application() noexcept
    {
        // Renderer must be reset to unmap the texture
        ImGui_ImplOpenGL2_Shutdown();
        ImGui_ImplSDL2_Shutdown();
        ImGui::DestroyContext();
        _renderer.reset();
        glDeleteTextures(1, &_texture);
        SDL_GL_DeleteContext(_gl_context);
        SDL_DestroyWindow(_window);
    }

    void renderer_application::execute()
    {
        std::cout << "Start rendering" << std::endl;
        while (!_handle_events())
        {
            _renderer->update();
            _draw();
        }
    }

    void renderer_application::_next_camera_setting()
    {
        std::cout << "\nSwitch to camera setting ";
        switch (_camera_mouse_wheel_setting)
        {
        case renderer_frontend::lens_setting::SENSOR_LENS_DISTANCE:
            std::cout << "focal length" << std::endl;
            _camera_mouse_wheel_setting = renderer_frontend::lens_setting::FOCAL_LENGTH;
            break;

        case renderer_frontend::lens_setting::FOCAL_LENGTH:
            std::cout << "diaphragm radius" << std::endl;
            _camera_mouse_wheel_setting = renderer_frontend::lens_setting::DIAPHRAGM_RADIUS;
            break;

        case renderer_frontend::lens_setting::DIAPHRAGM_RADIUS:
            std::cout << "sensor-lens distance" << std::endl;
            _camera_mouse_wheel_setting = renderer_frontend::lens_setting::SENSOR_LENS_DISTANCE;
            break;

        default:
            break;
        }
    }

    void renderer_application::_handle_key_down(SDL_Keysym key)
    {
        constexpr auto move_step = 0.1f;

        switch (key.sym)
        {
            case SDLK_TAB:
                _next_camera_setting();
                break;

            case SDLK_ESCAPE:
                _save_current_image();
                break;

            case SDLK_z:
                _renderer->camera_move_forward(move_step);
                break;

            case SDLK_s:
                _renderer->camera_move_forward(-move_step);
                break;

            case SDLK_q:
                _renderer->camera_move_lateral(-move_step);
                break;

            case SDLK_d:
                _renderer->camera_move_lateral(move_step);
                break;

            case SDLK_SPACE:
                _renderer->camera_move(0, 0, move_step);
                break;

            case SDLK_w:
                _renderer->camera_move(0, 0, -move_step);
                break;

            case SDLK_r:
                _switch_mouse_mode();
                break;
        }
    }

    void renderer_application::_handle_camera_mouse_wheel(bool up)
    {
        float factor;
        switch (_camera_mouse_wheel_setting)
        {
            default:
            case renderer_frontend::lens_setting::SENSOR_LENS_DISTANCE:
                factor = 1.0001f;
                break;

            case renderer_frontend::lens_setting::FOCAL_LENGTH:
                factor = 1.01f;
                break;

            case renderer_frontend::lens_setting::DIAPHRAGM_RADIUS:
                factor = 1.1f;
                break;
        }

        const float current_value = _renderer->get_camera_lens_setting(_camera_mouse_wheel_setting);
        const float new_value = current_value * (up ? factor : (1.f/factor));
        _renderer->set_camera_lens_setting(_camera_mouse_wheel_setting, new_value);
    }

    void renderer_application::_handle_camera_mouse_motion(int xrel, int yrel)
    {
        constexpr auto factor = 0.001f;
        _camera_phi -= factor * xrel;
        _camera_theta -= factor * yrel;
        _renderer->camera_rotate(_camera_theta, _camera_phi);
    }

    bool renderer_application::_handle_events()
    {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {

            switch (event.type)
            {
                case SDL_KEYDOWN:
                    if (_mouse_mode == mouse_mode::CAMERA || !ImGui::GetIO().WantCaptureKeyboard)
                        _handle_key_down(event.key.keysym);
                    break;
                case SDL_MOUSEWHEEL:
                    if (_mouse_mode == mouse_mode::CAMERA)
                        _handle_camera_mouse_wheel(event.wheel.y > 0);
                    break;
                case SDL_MOUSEMOTION:
                    if (_mouse_mode == mouse_mode::CAMERA)
                        _handle_camera_mouse_motion(event.motion.xrel, event.motion.yrel);
                    break;
                case SDL_QUIT:
                    return true;
                    break;
                case SDL_WINDOWEVENT:
                    if (event.window.event == SDL_WINDOWEVENT_RESIZED)
                        _update_size();
                    break;
            }

            if (_mouse_mode == mouse_mode::GUI)
            {
                // std::cout << "IMGUI EVENT\n" << std::endl;
                ImGui_ImplSDL2_ProcessEvent(&event);
            }
        }

        return false;
    }

    void renderer_application::_draw()
    {
        // Prepare gui frame
        ImGui_ImplOpenGL2_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();
        _gui->draw();
        ImGui::Render(); // Prepare data for rendering

        glClearColor(1.f, 0.f, 1.f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw the sensor texture
        glBindTexture(GL_TEXTURE_2D, _texture);
        glBegin(GL_QUADS);
        glVertex2i(0, 0); glTexCoord2i(1, 0);
        glVertex2i(1, 0); glTexCoord2i(1, 1);
        glVertex2i(1, 1); glTexCoord2i(0, 1);
        glVertex2i(0, 1); glTexCoord2i(0, 0);
        glEnd();

        // Draw the gui
        ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());

        SDL_GL_SwapWindow(_window);
    }

    void renderer_application::_update_size()
    {
        int width = 0;
        int height = 0;
        SDL_GetWindowSize(_window, &width, &height);
        glViewport(0, 0, width, height);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.f, 1.f, 0.f, 1.f, -1.f, 1.f);
    }

    void renderer_application::_switch_mouse_mode()
    {
        _mouse_mode = (_mouse_mode == mouse_mode::CAMERA) ? mouse_mode::GUI : mouse_mode::CAMERA;
        SDL_SetWindowGrab(_window, static_cast<SDL_bool>(_mouse_mode == mouse_mode::CAMERA));
        SDL_SetRelativeMouseMode(static_cast<SDL_bool>(_mouse_mode == mouse_mode::CAMERA));
    }

    void renderer_application::_save_current_image()
    {
        const auto bitmap = _renderer->get_image();

        std::cout << "\nSave image on disk" << std::endl;
        bitmap_write(
            "out.bmp", bitmap,
            _renderer->get_image_width(),
            _renderer->get_image_height());
    }
}
