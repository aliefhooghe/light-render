#ifndef XRENDER_RENDERER_FRONTEND_H_
#define XRENDER_RENDERER_FRONTEND_H_

#include <vector>
#include <functional>
#include <chrono>
#include <GL/gl.h>

#include "host/bitmap/bitmap.h"
#include "host/configuration/configuration.h"
#include "rendering_status.h"

namespace Xrender
{
    /**
     * \brief cuda aware private implementation
     */
    class renderer_frontend_implementation;

    /**
     * \brief Expose renderer feature without having to include cuda stuff
     *
     */
    class renderer_frontend
    {

    public:
        class setting
        {
        public:
            setting(const std::string& n, float& val)
            : _name{n}, _val{val}
            {}

            float& value() const noexcept { return _val; }
            const std::string& name() const { return _name; }
        private:
            const std::string _name;
            float& _val;
        };

        class worker_descriptor
        {
        public:
            worker_descriptor(const std::string& n, std::vector<setting>&& s)
            : _name{n}, _settings{s}
            {}
            const std::string& name() const { return _name; }
            const std::vector<setting>& settings() const { return _settings; }
        private:
            const std::string _name;
            const std::vector<setting> _settings;
        };

        enum class worker_type
        {
            Developer,
            Renderer
        };

        renderer_frontend(renderer_frontend&) = delete;
        renderer_frontend(renderer_frontend&&) noexcept;
        ~renderer_frontend() noexcept;

        /**
         * \brief Build a renderer frontend with the given configuration and
         * texture id. The texture must use the RGBA float format and being sized
         * accordingly to the configuration.
         */
        static std::unique_ptr<renderer_frontend> build_renderer_frontend(
            const render_configuration& configuration,
            GLuint texture_id = GL_INVALID_VALUE);

        /**
         *      Camera api
         */
        void scale_sensor_lens_distance(bool up, float factor);
        void scale_focal_length(bool up, float factor);
        void scale_diaphragm_radius(bool up, float factor);
        void camera_move(float dx, float dy, float dz);
        void camera_move_forward(float distance);
        void camera_move_lateral(float distance);
        void camera_rotate(float theta, float phi);

        /**
         *
         */
        void integrate_for(const std::chrono::milliseconds& max_duration);

        /**
         * \brief Develop the current renderer sensor to the texture.
         */
        void develop_image();

        /**
         * \brief Retrieve the current texture in rgb24 format
         */
        std::vector<rgb24> get_image();

        unsigned int get_image_width() const noexcept;
        unsigned int get_image_height() const noexcept;

        /**
         *
         */

        /**
         *
         */
        std::size_t get_worker_count(worker_type type) const;
        void set_current_worker(worker_type type, std::size_t worker_id);
        std::size_t get_current_worker(worker_type type) const;
        const worker_descriptor& get_current_worker_descriptor(worker_type type) const;
        const worker_descriptor& get_worker_descriptor(worker_type type, std::size_t renderer_id) const;

        const rendering_status& get_rendering_status() const noexcept;

        renderer_frontend(renderer_frontend_implementation* impl);
    private:
        renderer_frontend_implementation *_implementation{nullptr};
    };

}

#endif /* XRENDER_BASTRACT_RENDERER_FRONTEND_H_ */