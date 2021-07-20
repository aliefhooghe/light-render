
#include <stdexcept>
#include <iostream>
#include <cmath>

#include <GL/glew.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

#include "host/camera_handling/camera_configuration.cuh"

#include "renderer_display.cuh"

namespace Xrender
{
    __host__ renderer_display::renderer_display(camera &camera)
    :   _camera{camera}
    {
        static bool sdl_was_init = false;
        if (!sdl_was_init) {
            SDL_Init(SDL_INIT_VIDEO);
            sdl_was_init = true;
        }

        _window = SDL_CreateWindow(
            "Xrender", 0, 0,
            camera.get_image_width(),
            camera.get_image_height(),
            SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);

        _gl_context = SDL_GL_CreateContext(_window);

        GLenum glew_error = glewInit();
        if (glew_error != GLEW_OK)
        {
            throw std::runtime_error(reinterpret_cast<const char*>(glewGetErrorString(glew_error)));
        }

        //
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

        _texture = std::make_unique<gpu_texture>(
            camera.get_image_width(),
            camera.get_image_height());

        _update_size();
    }

    __host__ renderer_display::~renderer_display() noexcept
    {
        _texture.reset(); // before destroying the openGL context
        SDL_GL_DeleteContext(_gl_context);
        SDL_DestroyWindow(_window);
    }

    __host__ void renderer_display::execute()
    {
        while (!_handle_events())
        {
            if (!_renderers.empty())
            {
                auto& renderer = _renderers[_current_renderer];
                renderer.integrate();
                renderer.develop_to_texture(*_texture);
            }

            _draw();
        }
    }

    __host__ void renderer_display::add_view(
        std::unique_ptr<abstract_renderer> &&renderer,
        std::unique_ptr<abstract_image_developer> &&developer)
    {
        _renderers.emplace_back(_camera, std::move(renderer), std::move(developer));
        _renderers.back().set_interval(_interval);
    }

    __host__ void renderer_display::_update_parameter(float& param, bool up, float factor)
    {
        param *= std::pow<float>(factor, up ? 1.f : -1.f);
        _reset_current_renderer();
    }

    __host__ void renderer_display::_next_renderer(bool previous)
    {
        if (_renderers.empty())
            return;

        _current_renderer = (
            _current_renderer +
            _renderers.size() +
            (previous ? -1 : 1)
        ) % _renderers.size();

        _reset_current_renderer();
        std::cout << "\nSwitch to renderer " << _current_renderer << std::endl;
    }

    __host__ void renderer_display::_reset_current_renderer()
    {
        if (_renderers.empty())
            return;
        _renderers[_current_renderer].reset();
    }

    __host__ void renderer_display::_handle_key_down(SDL_Keysym key)
    {
        switch (key.sym)
        {
            case SDLK_PAGEUP:   _next_renderer(); break;
            case SDLK_PAGEDOWN: _next_renderer(true); break;
            case SDLK_SPACE:    _reset_current_renderer(); break;
            case SDLK_UP:       camera_update_focal_length(_camera, _camera._focal_length * 1.1); _reset_current_renderer(); break;
            case SDLK_DOWN:     camera_update_focal_length(_camera, _camera._focal_length / 1.1); _reset_current_renderer(); break;
            case SDLK_ESCAPE:   _switch_fast_mode(); break;
            case SDLK_p:        _update_parameter(_camera._diaphragm_radius, true); break;
            case SDLK_m:        _update_parameter(_camera._diaphragm_radius, false); break;
        }
    }

    __host__ void renderer_display::_handle_mouse_wheel(bool up)
    {
        _update_parameter(_camera._sensor_lens_distance, up, 1.0001f);
    }

    __host__ bool renderer_display::_handle_events()
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

    __host__ void renderer_display::_draw()
    {
        glClearColor(1.f, 0.f, 1.f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT);

        glBindTexture(GL_TEXTURE_2D, _texture->get_gl_texture_id());

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

    __host__ void renderer_display::_update_size()
    {
        int width = 0;
        int height = 0;
        SDL_GetWindowSize(_window, &width, &height);
        glViewport(0, 0, width, height);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.f, 1.f, 0.f, 1.f, -1.f, 1.f);
    }

    __host__ void renderer_display::_switch_fast_mode()
    {
        _fast_mode = !_fast_mode;

        std::cout << "\n" <<  (_fast_mode ? "enable" : "disable") <<  " fast mode." << std::endl;

        if (_fast_mode)
            _interval = std::chrono::milliseconds{fast_interval};
        else
            _interval = std::chrono::milliseconds{interactive_interval};

        for (auto& renderer : _renderers)
            renderer.set_interval(_interval);
    }
}