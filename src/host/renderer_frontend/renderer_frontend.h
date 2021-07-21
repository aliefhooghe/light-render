#ifndef XRENDER_RENDERER_FRONTEND_H_
#define XRENDER_RENDERER_FRONTEND_H_

#include <vector>
#include <functional>
#include <chrono>
#include <GL/gl.h>

#include "host/bitmap/bitmap.h"
#include "host/configuration/configuration.h"

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
            using set_callback = std::function<void(bool)>;
        public:
            setting(const std::string& n, set_callback c)
            : _name{n}, _callback{c}
            {}
            void scale(bool up) const { _callback(up); }
            const std::string& name() const { return _name; }
        private:
            const std::string _name;
            const set_callback _callback;
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
            GLuint texture_id);

        /**
         *      Camera api
         */
        void scale_sensor_lens_distance(bool up, float factor);
        // void scale_focal_length(float factor);

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
        std::size_t get_renderer_count() const;
        void set_current_renderer(std::size_t renderer_id);
        std::size_t get_current_renderer() const;
        const worker_descriptor& get_renderer_descriptor(std::size_t renderer_id) const;

        std::size_t get_developer_count() const;
        void set_current_developer(std::size_t developer_id);
        std::size_t get_current_developer() const;
        const worker_descriptor& get_developer_descriptor(std::size_t developer_id) const;

        renderer_frontend(renderer_frontend_implementation* impl);
    private:
        renderer_frontend_implementation *_implementation{nullptr};
    };

}

#endif /* XRENDER_BASTRACT_RENDERER_FRONTEND_H_ */