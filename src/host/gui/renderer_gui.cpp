#include <imgui.h>
#include <iostream>
#include "renderer_gui.h"

namespace Xrender {

    renderer_gui::renderer_gui(renderer_frontend& frontend)
    : _frontend{frontend}
    {

    }

    void renderer_gui::draw()
    {
        ImGui::Begin("Xrender Control Panel");

        if (ImGui::TreeNode("Developer Settings"))
        {
            _draw_worker_panel(renderer_frontend::worker_type::Developer);
            ImGui::TreePop();
        }
        if (ImGui::TreeNode("Render Settings"))
        {
            _draw_worker_panel(renderer_frontend::worker_type::Renderer);
            ImGui::TreePop();
        }
        if (ImGui::TreeNode("Widget Demo"))
        {
            ImGui::ShowDemoWindow();
            ImGui::TreePop();
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
                "%.3f", 1.f);
        }
    }

    void renderer_gui::_draw_worker_selector(renderer_frontend::worker_type type)
    {
        const auto& current_desc = _frontend.get_current_worker_descriptor(type);
        if (ImGui::BeginCombo("Algo", current_desc.name().c_str()))
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
}
