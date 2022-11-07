#include <imgui.h>
#include <iostream>
#include "renderer_gui.h"

namespace Xrender {

    renderer_gui::renderer_gui(renderer_frontend& frontend)
    : _frontend{frontend}
    {
        std::fill(std::begin(_speed_values), std::end(_speed_values), 1.f);
    }

    void renderer_gui::draw()
    {
        ImGui::Begin("Xrender Control Panel");

        if (ImGui::CollapsingHeader("Developer Settings"))
        {
            _draw_worker_panel(renderer_frontend::worker_type::Developer);
        }

        if (ImGui::CollapsingHeader("Render Settings"))
        {
            _draw_worker_panel(renderer_frontend::worker_type::Renderer);
        }

        if (ImGui::CollapsingHeader("Camera Settings"))
        {
            _draw_camera_panel();
        }

        if (ImGui::CollapsingHeader("Status"))
        {
            _draw_status_panel();
        }

        ImGui::End();
    }

    void renderer_gui::_draw_worker_panel(renderer_frontend::worker_type type)
    {
        _draw_worker_selector(type);

        // Current desc may have been change by the selector
        const auto& current_desc = _frontend.get_current_worker_descriptor(type);
        for (const auto& setting : current_desc.settings())
        {
            ImGui::DragFloat(setting.name().c_str(), &setting.value(),
                /* speed */ 0.01f,
                /* min */ 0.f,
                /* max */ 10.f,
                "%.3f", 2.f);
        }
    }

    void renderer_gui::_draw_worker_selector(renderer_frontend::worker_type type)
    {
        const char* label = (type == renderer_frontend::worker_type::Developer)
            ? "developer algo" : "render algo";
        const auto& current_desc = _frontend.get_current_worker_descriptor(type);
        if (ImGui::BeginCombo(label, current_desc.name().c_str()))
        {
            const auto developer_count = _frontend.get_worker_count(type);
            for (int i = 0; i < developer_count; i++)
            {
                const auto &desc = _frontend.get_worker_descriptor(type, i);
                if (ImGui::Selectable(desc.name().c_str()))
                    _frontend.set_current_worker(type, i);
            }
            ImGui::EndCombo();
        }
    }

    void renderer_gui::_draw_camera_setting(renderer_frontend::lens_setting setting, float speed, float power, float vmin, float vmax, const char* label)
    {
        float value = _frontend.get_camera_lens_setting(setting);
        if (ImGui::DragFloat(label, &value,
            /* speed */ speed,
            /* min */ vmin,
            /* max */ vmax,
            "%.3f", power))
        {
            _frontend.set_camera_lens_setting(setting, value);
        }
    }

    void renderer_gui::_draw_camera_panel()
    {
        _draw_camera_setting(renderer_frontend::lens_setting::FOCAL_LENGTH, 0.001f, 4.f, 0.001f, 0.4f, "focal length");
        _draw_camera_setting(renderer_frontend::lens_setting::SENSOR_LENS_DISTANCE, 0.001f, 10.f, 0.001f, 0.4f, "sensor lens dist");
        _draw_camera_setting(renderer_frontend::lens_setting::DIAPHRAGM_RADIUS, 0.01f, 4.f, 0.f, 2.0f, "diaph radius");
    }

    void renderer_gui::_draw_status_panel()
    {
        const auto& color = ImGui::GetStyle().Colors[ImGuiCol_PlotHistogram];
        const auto& status = _frontend.get_rendering_status();

        if (ImGui::SliderFloat("rendering fps", &_rendering_fps, 0.1, 60, "%.1f", 2.f))
        {
            _frontend.set_integration_duration(
                std::chrono::milliseconds{static_cast<int64_t>(1000.f / _rendering_fps)});
        }

        ImGui::TextColored(color, "frame : %llu samples", status.frame_sample_count);
        ImGui::TextColored(color, "total : %llu samples", status.total_integrated_sample);
        ImGui::TextColored(color, "speed : %.1f spp/sec", status.spp_per_second);
        ImGui::Separator();

        // Print a graph of speeds
        if (_last_speed != status.spp_per_second)
        {
            _last_speed = status.spp_per_second;
            _speed_values[_speed_offset] = status.spp_per_second;
            _speed_offset = (_speed_offset + 1) % speed_buffer_size;
        }

        ImGui::PlotHistogram(
            "rendering speed", _speed_values.data(), speed_buffer_size, _speed_offset,
            "spp/sec", 0, FLT_MAX, ImVec2(0, 96.f));

        // Print a progress bar
        ImGui::Separator();
        ImGui::ProgressBar(static_cast<float>(status.total_integrated_sample) / 1000.0);
    }
}
