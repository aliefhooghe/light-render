
#include <stdexcept>
#include <iostream>
#include <cmath>

#include <GL/glew.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

#include "host/bitmap/bitmap.h"
#include "renderer_application.h"

namespace Xrender
{
    renderer_application::renderer_application(const render_configuration& config)
    {
        static bool sdl_was_init = false;
        if (!sdl_was_init) {
            SDL_Init(SDL_INIT_VIDEO);
            sdl_was_init = true;
        }

        const auto width = config.camera_config.image_width;
        const auto height = config.camera_config.image_height;

        _window = SDL_CreateWindow(
            "Xrender", 0, 0,
            width, height,
            SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);

        // Hide the windows until the render is not ready to start
        SDL_HideWindow(_window);

        _gl_context = SDL_GL_CreateContext(_window);

        GLenum glew_error = glewInit();
        if (glew_error != GLEW_OK)
        {
            throw std::runtime_error(reinterpret_cast<const char*>(glewGetErrorString(glew_error)));
        }

        //
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

        // Create openGL texture to develop into
        glGenTextures(1, &_texture);
        glBindTexture(GL_TEXTURE_2D, _texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glEnable(GL_TEXTURE_2D);

        _renderer = renderer_frontend::build_renderer_frontend(config, _texture);

        SDL_ShowWindow(_window);
        _update_size();
    }

    renderer_application::~renderer_application() noexcept
    {
        // Renderer must be reset to unmap the texture
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
            const auto render_duration =
                std::chrono::milliseconds(_fast_mode ? 10000 : 20);

            _renderer->integrate_for(render_duration);
            _renderer->develop_image();
            _draw();
        }
    }

    void renderer_application::_next_renderer()
    {
        const auto renderer_count = _renderer->get_renderer_count();
        const auto current_renderer = _renderer->get_current_renderer();
        const auto next_renderer = (current_renderer + 1) % renderer_count;
        const auto& next_renderer_desc = _renderer->get_renderer_descriptor(next_renderer);

        _renderer->set_current_renderer(next_renderer);

        std::cout << "\nSwitch to renderer " << next_renderer_desc.name() << std::endl;
    }

    void renderer_application::_next_developer()
    {
        const auto developer_count = _renderer->get_developer_count();
        const auto current_developer = _renderer->get_current_developer();
        const auto next_developer = (current_developer + 1) % developer_count;
        const auto& next_developer_desc = _renderer->get_developer_descriptor(next_developer);

        _renderer->set_current_developer(next_developer);

        std::cout << "\nSwitch to developer " << next_developer_desc.name() << std::endl;
    }

    void renderer_application::_next_control_mode()
    {
        std::cout << "\nSwitch to control mode : ";

        switch (_control_mode)
        {
        case control_mode::CAMERA_SETTINGS:
            std::cout << "Developer" << std::endl;
            _control_mode = control_mode::DEVELOPER_SETTINGS;
            break;
        case control_mode::DEVELOPER_SETTINGS:
            std::cout << "Renderer" << std::endl;
            _control_mode = control_mode::RENDERER_SETTINGS;
            break;
        case control_mode::RENDERER_SETTINGS:
            std::cout << "Camera" << std::endl;
            _control_mode = control_mode::CAMERA_SETTINGS;
            break;
        default:
            break;
        }
    }

    void renderer_application::_next_worker_setting()
    {
        if (_control_mode != control_mode::CAMERA_SETTINGS)
        {
            const auto& worker = _get_current_control_worker();
            const auto& worker_settings = worker.settings();
            const auto setting_count = worker_settings.size();

            if (setting_count > 0) {
                _control_setting_id = (_control_setting_id + 1) % setting_count;
                std::cout << "\nSwitched to control setting : " << worker.name()
                        <<  " - " << worker_settings[_control_setting_id].name() << std::endl;
            }
        }
    }

    const renderer_frontend::worker_descriptor& renderer_application::_get_current_control_worker()
    {
        return
            _control_mode == control_mode::DEVELOPER_SETTINGS ?
                _renderer->get_developer_descriptor(_renderer->get_current_developer()) :
                _renderer->get_renderer_descriptor(_renderer->get_current_renderer());
    }

    void renderer_application::_handle_key_down(SDL_Keysym key)
    {
        switch (key.sym)
        {
            case SDLK_TAB:
                _next_worker_setting();
                break;

            case SDLK_c:
                _next_control_mode();
                break;

            case SDLK_RETURN:
                _next_renderer();
                break;

            case SDLK_RSHIFT:
                _next_developer();
                break;

            case SDLK_SPACE:
                _switch_fast_mode();
                break;

            case SDLK_s:
                _save_current_image();
                break;
        }
    }

    void renderer_application::_handle_mouse_wheel(bool up)
    {
        if (_control_mode == control_mode::CAMERA_SETTINGS)
        {
            _renderer->scale_sensor_lens_distance(up, 1.001f);
        }
        else
        {
            const auto& worker = _get_current_control_worker();
            const auto& worker_settings = worker.settings();
            const auto setting_count = worker_settings.size();

            if (_control_setting_id < setting_count)
            {
                worker_settings[_control_setting_id].scale(up);
            }
        }
    }

    bool renderer_application::_handle_events()
    {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type)
            {
                case SDL_KEYDOWN:    _handle_key_down(event.key.keysym); break;
                case SDL_MOUSEWHEEL: _handle_mouse_wheel(event.wheel.y > 0); break;
                case SDL_QUIT:       return true; break;

                case SDL_WINDOWEVENT:
                    if (event.window.event == SDL_WINDOWEVENT_RESIZED)
                        _update_size();
            }
        }

        return false;
    }

    void renderer_application::_draw()
    {
        glClearColor(1.f, 0.f, 1.f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT);

        glBindTexture(GL_TEXTURE_2D, _texture);

        glBegin(GL_QUADS);

        glVertex2i(0, 0);
        glTexCoord2i(1, 0);

        glVertex2i(1, 0);
        glTexCoord2i(1, 1);

        glVertex2i(1, 1);
        glTexCoord2i(0, 1);

        glVertex2i(0, 1);
        glTexCoord2i(0, 0);

        glEnd();

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

    void renderer_application::_switch_fast_mode()
    {
        _fast_mode = !_fast_mode;
        std::cout << "\n" <<  (_fast_mode ? "enable" : "disable") <<  " fast mode." << std::endl;
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