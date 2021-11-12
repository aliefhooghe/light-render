
#include <math.h>

#include "gpu/common/abstract_image_developer.cuh"
#include "gpu/common/gpu_texture.cuh"
#include "gpu/common/renderer_manager.cuh"
#include "gpu/image_developers/average_image_developer.cuh"
#include "gpu/image_developers/gamma_image_developer.cuh"
#include "gpu/model/bvh_tree.cuh"
#include "gpu/model/camera.cuh"
#include "gpu/renderers/naive_mc_renderer.cuh"
#include "gpu/renderers/preview_renderer.cuh"
#include "gpu/utils/cuda_exception.cuh"
#include "gpu/utils/device_probing.cuh"
#include "gpu/utils/gpu_vector_copy.cuh"
#include "host/bvh_builder/bvh_builder.cuh"
#include "host/camera_handling/camera_configuration.cuh"
#include "host/camera_handling/camera_configuration.cuh"
#include "host/model_loader/wavefront_obj.cuh"
#include "host/utils/chronometer.h"
#include "renderer_frontend.h"

namespace Xrender
{
    class renderer_frontend_implementation
    {

    public:
        using setting = renderer_frontend::setting;
        using worker_descriptor = renderer_frontend::worker_descriptor;

        renderer_frontend_implementation(
            camera cam,
            const host_bvh_tree::gpu_compatible_bvh &gpu_bvh,
            GLuint texture_id);
        ~renderer_frontend_implementation() noexcept;

        void scale_sensor_lens_distance(bool up, float factor);
        void scale_focal_length(bool up, float factor);
        void scale_diaphragm_radius(bool up, float factor);
        void camera_move(float dx, float dy, float dz);
        void camera_move_forward(float distance);
        void camera_move_lateral(float distance);
        void camera_rotate(float theta, float phi);

        void integrate_for(const std::chrono::milliseconds &max_duration);
        void develop_image();
        std::vector<rgb24> get_image();
        std::size_t get_renderer_count() const;
        void set_current_renderer(std::size_t renderer_id);
        std::size_t get_current_renderer() const;
        const worker_descriptor &get_renderer_descriptor(std::size_t renderer_id) const;

        std::size_t get_developer_count() const;
        void set_current_developer(std::size_t developer_id);
        std::size_t get_current_developer() const;
        const worker_descriptor &get_developer_descriptor(std::size_t developer_id) const;

        unsigned int get_image_width() const noexcept { return _camera.get_image_width(); }
        unsigned int get_image_height() const noexcept { return _camera.get_image_height(); }

    private:
        void _add_renderer(
            worker_descriptor&& descriptor,
            std::unique_ptr<abstract_renderer> &&renderer);
        void _add_image_developer(
            worker_descriptor&& descriptor,
            std::unique_ptr<abstract_image_developer> &&developer);
        void _reset_current_renderer();

        camera _camera;
        bvh_node *_device_tree{nullptr};
        face *_device_model{nullptr};

        std::size_t _current_renderer{0u};
        std::size_t _current_developer{0u};

        std::vector<renderer_manager> _renderers{};
        std::vector<std::unique_ptr<abstract_image_developer>> _developpers{};

        std::vector<worker_descriptor> _renderers_settings{};
        std::vector<worker_descriptor> _developpers_settings{};

        registered_texture _registered_texture;
    };

    /**
     * Private implementation
     */

    renderer_frontend_implementation::renderer_frontend_implementation(
        camera cam,
        const host_bvh_tree::gpu_compatible_bvh &gpu_bvh,
        GLuint texture_id)
    :   _camera{cam},
        _registered_texture{texture_id, cam.get_image_width(), cam.get_image_height()}
    {
        _device_tree = clone_to_device(gpu_bvh.tree);
        _device_model = clone_to_device(gpu_bvh.model);

        // Add average developer
        {
            auto average_developer = std::make_unique<average_image_developer>();
            auto *average_dev = average_developer.get();

            _add_image_developer(
                {
                    "Average Developer",
                    {
                        {
                            "Factor", [average_dev](bool up) { average_dev->scale_factor(up); }
                        }
                    }
                },
                std::move(average_developer));
        }

        // Add gamma developer
        {
            auto gamma_developer = std::make_unique<gamma_image_developer>();
            auto *gamma_dev = gamma_developer.get();

            _add_image_developer(
                {
                    "Gamma developer",
                    {
                        {
                            "Factor", [gamma_dev](bool up) { gamma_dev->scale_factor(up); }
                        },
                        {
                            "Gamma",  [gamma_dev](bool up) { gamma_dev->scale_gamma(up); }
                        }
                    }
                },
                std::move(gamma_developer));
        }

        _add_renderer(
            {"Preview", {}},
            std::make_unique<preview_renderer>(_device_tree, _device_model));

        _add_renderer(
            {"Path Tracer", {}},
            std::make_unique<naive_mc_renderer>(_device_tree, _device_model));
    }

    renderer_frontend_implementation::~renderer_frontend_implementation() noexcept
    {
        CUDA_WARNING(cudaFree(_device_tree));
        CUDA_WARNING(cudaFree(_device_model));
    }

    void renderer_frontend_implementation::_add_renderer(
        worker_descriptor&& descriptor,
        std::unique_ptr<abstract_renderer> &&renderer)
    {
        _renderers_settings.emplace_back(std::move(descriptor));
        _renderers.emplace_back(_camera, std::move(renderer));
    }

    void renderer_frontend_implementation::_add_image_developer(
        worker_descriptor&& descriptor,
        std::unique_ptr<abstract_image_developer> &&developer)
    {
        _developpers_settings.emplace_back(std::move(descriptor));
        _developpers.emplace_back(std::move(developer));
    }

    void renderer_frontend_implementation::_reset_current_renderer()
    {
        if (_current_renderer >= get_renderer_count())
            return;
        _renderers[_current_renderer].reset();
    }
    void renderer_frontend_implementation::scale_sensor_lens_distance(bool up, float factor)
    {
        if (up)
            _camera._sensor_lens_distance *= factor;
        else
            _camera._sensor_lens_distance /= factor;
        _reset_current_renderer();
    }

    void renderer_frontend_implementation::scale_focal_length(bool up, float factor)
    {
        camera_update_focal_length(_camera, up, factor);
        _reset_current_renderer();
    }

    void renderer_frontend_implementation::scale_diaphragm_radius(bool up, float factor)
    {
        if (up)
            _camera._diaphragm_radius *= factor;
        else
            _camera._diaphragm_radius /= factor;
        _reset_current_renderer();
    }

    void renderer_frontend_implementation::camera_move(float dx, float dy, float dz)
    {
        _camera._position += float3{dx, dy, dz};
        _reset_current_renderer();
    }

    void renderer_frontend_implementation::camera_move_forward(float distance)
    {
        camera_update_pos_forward(_camera, distance);
        _reset_current_renderer();
    }

    void renderer_frontend_implementation::camera_move_lateral(float distance)
    {
        camera_update_pos_lateral(_camera, distance);
        _reset_current_renderer();
    }

    void renderer_frontend_implementation::camera_rotate(float theta, float phi)
    {
        camera_update_rotation(_camera, theta, phi);
        _reset_current_renderer();
    }



    void renderer_frontend_implementation::integrate_for(const std::chrono::milliseconds &max_duration)
    {
        if (_current_renderer >= get_renderer_count())
            return;
        _renderers[_current_renderer].integrate_for(max_duration);
    }

    void renderer_frontend_implementation::develop_image()
    {
        if (_current_developer >= get_developer_count() || _current_renderer >= get_renderer_count())
            return;

        auto mapped_surface = _registered_texture.get_mapped_surface();
        auto &renderer = _renderers[_current_renderer];

        _developpers[_current_developer]->call_develop_to_texture_kernel(
            renderer.get_total_sample_count(),
            _camera.get_image_width(),
            _camera.get_image_height(),
            renderer.get_device_sensor(),
            mapped_surface.surface());

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<rgb24> renderer_frontend_implementation::get_image()
    {
        const auto host_texture = _registered_texture.retrieve_texture();
        std::vector<rgb24> image{host_texture.size()};

        // Convert to 24 bit bitmap samples
        std::transform(
            host_texture.begin(), host_texture.end(),
            image.begin(),
            [](const float4& rgba)
            {
                return rgb24::from_float(rgba.x, rgba.y, rgba.z);
            });

        return image;
    }

    std::size_t renderer_frontend_implementation::get_renderer_count() const
    {
        return _renderers.size();
    }

    void renderer_frontend_implementation::set_current_renderer(std::size_t renderer_id)
    {
        if (renderer_id < get_renderer_count()) {
            _current_renderer = renderer_id;
            _reset_current_renderer();
        }
        else {
            throw std::invalid_argument("invalid renderer id");
        }
    }

    std::size_t renderer_frontend_implementation::get_current_renderer() const
    {
        return _current_renderer;
    }

    const renderer_frontend_implementation::worker_descriptor &renderer_frontend_implementation::get_renderer_descriptor(std::size_t renderer_id) const
    {
        if (renderer_id < get_renderer_count())
            return _renderers_settings[renderer_id];
        else
            throw std::invalid_argument("invalid renderer id");
    }

    std::size_t renderer_frontend_implementation::get_developer_count() const
    {
        return _developpers.size();
    }

    void renderer_frontend_implementation::set_current_developer(std::size_t developer_id)
    {
        if (developer_id < get_developer_count()) {
            _current_developer = developer_id;
        }
        else {
            throw std::invalid_argument("invalid renderer id");
        }
    }

    std::size_t renderer_frontend_implementation::get_current_developer() const
    {
        return _current_developer;
    }

    const renderer_frontend_implementation::worker_descriptor &renderer_frontend_implementation::get_developer_descriptor(std::size_t developer_id) const
    {
        if (developer_id < get_renderer_count())
            return _developpers_settings[developer_id];
        else
            throw std::invalid_argument("invalid renderer id");
    }

    /**
     * Private implementation wrapping
     */

    std::unique_ptr<renderer_frontend> renderer_frontend::build_renderer_frontend(const render_configuration &configuration, GLuint texture_id)
    {
        if (!select_openGL_cuda_device())
            throw std::runtime_error("No cuda capable device was found");

        // Load model
        chronometer timewatch{};

        std::cout << "Loading " << configuration.model_path.generic_string() << std::endl;
        timewatch.start();
        const auto model = wavefront_obj_load(configuration.model_path);
        const auto load_duration = timewatch.stop();

        // Create a bvh usable on gpu
        std::cout << "Model loading took " << load_duration.count() << " ms\nBuild bvh tree (" << model.size() << " faces)" << std::endl;
        timewatch.start();
        const auto host_bvh = build_bvh_tree(model);
        const auto gpu_bvh = host_bvh->to_gpu_bvh();
        const auto bvh_build_duration = timewatch.stop();
        std::cout << "Bvh build took " << bvh_build_duration.count() << " ms" << std::endl;
        std::cout << "Bvh tree max depth is " << host_bvh->max_depth() << std::endl;
        std::cout << "GPU bvh tree size is  : " << gpu_bvh.tree.size() << std::endl;
        std::cout << "GPU bvh model size is : " << gpu_bvh.model.size() << std::endl;

        // Configure camera
        camera cam{};
        configure_camera(configuration.camera_config, cam);

        // Initialize the frontend implementation
        std::cout << "Initialize computations" << std::endl;
        auto *implementation =
            new renderer_frontend_implementation{cam, gpu_bvh, texture_id};

        return std::make_unique<renderer_frontend>(implementation);
    }

    renderer_frontend::renderer_frontend(renderer_frontend&& other) noexcept
    :   _implementation{other._implementation}
    {
        other._implementation = nullptr;
    }

    renderer_frontend::renderer_frontend(renderer_frontend_implementation *implementation)
    :   _implementation{implementation}
    {
    }

    renderer_frontend::~renderer_frontend() noexcept
    {
        if (_implementation)
            delete _implementation;
    }

    void renderer_frontend::scale_sensor_lens_distance(bool up, float factor)
    {
        _implementation->scale_sensor_lens_distance(up, factor);
    }

    void renderer_frontend::scale_focal_length(bool up, float factor)
    {
        _implementation->scale_focal_length(up, factor);
    }

    void renderer_frontend::scale_diaphragm_radius(bool up, float factor)
    {
        _implementation->scale_diaphragm_radius(up, factor);
    }

    void renderer_frontend::camera_move(float dx, float dy, float dz)
    {
        _implementation->camera_move(dx, dy, dz);
    }

    void renderer_frontend::camera_move_forward(float distance)
    {
        _implementation->camera_move_forward(distance);
    }

    void renderer_frontend::camera_move_lateral(float distance)
    {
        _implementation->camera_move_lateral(distance);
    }

    void renderer_frontend::camera_rotate(float theta, float phi)
    {
        _implementation->camera_rotate(theta, phi);
    }

    void renderer_frontend::integrate_for(const std::chrono::milliseconds &max_duration)
    {
        _implementation->integrate_for(max_duration);
    }

    void renderer_frontend::develop_image()
    {
        _implementation->develop_image();
    }

    std::vector<rgb24> renderer_frontend::get_image()
    {
        return _implementation->get_image();
    }

    unsigned int renderer_frontend::get_image_width() const noexcept
    {
        return _implementation->get_image_width();
    }

    unsigned int renderer_frontend::get_image_height() const noexcept
    {
        return _implementation->get_image_height();
    }


    std::size_t renderer_frontend::get_renderer_count() const
    {
        return _implementation->get_renderer_count();
    }

    void renderer_frontend::set_current_renderer(std::size_t renderer_id)
    {
        _implementation->set_current_renderer(renderer_id);
    }

    std::size_t renderer_frontend::get_current_renderer() const
    {
        return _implementation->get_current_renderer();
    }

    const renderer_frontend::worker_descriptor &renderer_frontend::get_renderer_descriptor(std::size_t renderer_id) const
    {
        return _implementation->get_renderer_descriptor(renderer_id);
    }

    std::size_t renderer_frontend::get_developer_count() const
    {
        return _implementation->get_developer_count();
    }

    void renderer_frontend::set_current_developer(std::size_t developer_id)
    {
        _implementation->set_current_developer(developer_id);
    }

    std::size_t renderer_frontend::get_current_developer() const
    {
        return _implementation->get_current_developer();
    }

    const renderer_frontend::worker_descriptor &renderer_frontend::get_developer_descriptor(std::size_t developer_id) const
    {
        return _implementation->get_developer_descriptor(developer_id);
    }

}