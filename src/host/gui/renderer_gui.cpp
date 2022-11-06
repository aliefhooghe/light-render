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
            _draw_worker_selector(renderer_frontend::worker_type::Developer);
            ImGui::TreePop();
        }
        if (ImGui::TreeNode("Render Settings"))
        {
            _draw_worker_selector(renderer_frontend::worker_type::Renderer);
            ImGui::TreePop();
        }

        ImGui::End();
    }

    void renderer_gui::_draw_worker_selector(renderer_frontend::worker_type type)
    {
        const auto current_dev = _frontend.get_current_worker(type);
        const auto& current_desc = _frontend.get_worker_descriptor(type, current_dev);
        if (ImGui::BeginCombo("algo", current_desc.name().c_str()))
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
