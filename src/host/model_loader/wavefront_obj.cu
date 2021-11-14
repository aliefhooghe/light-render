
#include <fstream>
#include <stdexcept>
#include <cstdio>
#include <map>
#include <iostream>

#include "wavefront_obj.cuh"
#include "material_builder.cuh"
#include "face_builder.cuh"

namespace Xrender
{
    /**
     * Material library loading
     */
    static void mtl_lib_add(
        const material& mtl, const std::string& name,
        std::vector<material>& mtl_bank,
        std::map<std::string, int> &mtl_name_map)
    {
        const int mtl_index = mtl_bank.size();
        mtl_bank.push_back(mtl);
        mtl_name_map[name] = mtl_index;
    }

    static void load_mtl_lib(
        const std::filesystem::path &path,
        std::vector<material>& mtl_bank,
        std::map<std::string, int> &mtl_name_map)
    {
        std::ifstream stream{path};
        material_builder mtl_builder{};
        std::string current_mtl_name{};

        if (!stream.is_open())
            throw std::invalid_argument("Unable to open material library " + path.generic_string());

        for (std::string line; std::getline(stream, line);)
        {
            switch (line[0])
            {
            case 'n':
            {
                char mtl_name[256];
                if (!current_mtl_name.empty())
                {
                    mtl_lib_add(
                        mtl_builder.make_material(), current_mtl_name, mtl_bank, mtl_name_map);
                }

                if (sscanf(line.c_str(), "newmtl %s\n", mtl_name) == 1)
                {
                    mtl_builder.decl_new_mtl();
                    current_mtl_name = mtl_name;
                }
            }
            break;

            case 'N':
            {
                float x;
                char c;
                if (sscanf(line.c_str(), "N%c %f", &c, &x) == 2)
                {
                    switch (c)
                    {
                        case 's': mtl_builder.decl_ns(x); break;
                        case 'i': mtl_builder.decl_ni(x); break;
                    }
                }
            }
            break;

            case 'K':
            {
                float x, y, z;
                char c;
                if (std::sscanf(line.c_str(), "K%c %f %f %f", &c, &x, &y, &z) == 4)
                {
                    switch (c)
                    {
                    case 'a': mtl_builder.decl_ka({x, y, z}); break;
                    case 'd': mtl_builder.decl_kd({x, y, z}); break;
                    case 's': mtl_builder.decl_ks({x, y, z}); break;
                    }
                }
            }
            break;

            case 'T':
            {
                float x, y, z;
                if (std::sscanf(line.c_str(), "Tf %f %f %f", &x, &y, &z) == 3)
                    mtl_builder.decl_tf({x, y, z});
            }
            break;

            case '#': // comment or annotation
            {
                if (line.rfind("#Source", 0) == 0)
                    mtl_builder.decl_mtl_type(material::SOURCE);
                else if (line.rfind("#Lambertian", 0) == 0)
                    mtl_builder.decl_mtl_type(material::LAMBERTIAN);
                else if (line.rfind("#Phong", 0) == 0)
                    mtl_builder.decl_mtl_type(material::PHONG);
                else if (line.rfind("#Mirror", 0) == 0)
                    mtl_builder.decl_mtl_type(material::MIRROR);
                else if (line.rfind("#Glass", 0) == 0)
                    mtl_builder.decl_mtl_type(material::GLASS);
                else
                {
                    float value;
                    switch (line[1])
                    {
                    case 'T':
                        if (std::sscanf(line.c_str(), "#T %f", &value) == 1)
                            mtl_builder.decl_temperature(value);
                        break;
                    case 'A':
                        if (std::sscanf(line.c_str(), "#A %f", &value) == 1)
                            mtl_builder.decl_cauchy_a(value);
                        break;
                    case 'B':
                        if (std::sscanf(line.c_str(), "#B %f", &value) == 1)
                            mtl_builder.decl_cauchy_b(value);
                        break;
                    case 'R':
                        if (std::sscanf(line.c_str(), "#R %f", &value) == 1)
                            mtl_builder.decl_reflexivity(value);
                        break;
                    }
                }
            }
            break;
            }
        }

        //  push last mtl
        if (!current_mtl_name.empty())
        {
            mtl_lib_add(
                mtl_builder.make_material(), current_mtl_name, mtl_bank, mtl_name_map);
        }
    }

    /**
     * Geometry loading
     */

    static void parse_vector(const std::string& line, std::vector<float3>& vertex, std::vector<float3>& normals)
    {
        float x, y, z;
        if (std::sscanf(line.c_str(), "v %f %f %f", &x, &y, &z) == 3)
            vertex.emplace_back(make_float3(x, y, z));
        else if (std::sscanf(line.c_str(), "vn %f %f %f", &x, &y, &z) == 3)
            normals.emplace_back(make_float3(x, y, z));
    }

    static void parse_face(
        const std::string& line, const std::vector<float3>& vertex, const std::vector<float3>& normals,
        const int current_mtl, std::vector<face>& geometry)
    {
        unsigned int v1, v2, v3;
        unsigned int vt1, vt2, vt3;
        unsigned int vn1, vn2, vn3;
        if (std::sscanf(line.c_str(), "f %u//%u %u//%u %u//%u\n",
                        &v1, &vn1, &v2, &vn2, &v3, &vn3) == 6 ||
            std::sscanf(line.c_str(), "f %u/%u/%u  %u/%u/%u  %u/%u/%u",
                        &v1, &vt1, &vn1, &v2, &vt2, &vn2, &v3, &vt3, &vn3) == 9)
        {
            if (v1 <= vertex.size() && v2 <= vertex.size() && v3 <= vertex.size() &&
                vn1 <= normals.size() && vn2 <= normals.size() && vn3 <= normals.size())
            {
                geometry.push_back(
                    make_face(current_mtl,
                              vertex[v1 - 1u], vertex[v2 - 1u], vertex[v3 - 1u],
                              normals[vn1 - 1u], normals[vn2 - 1u], normals[vn3 - 1u]));
            }
            else
            {
                std::cerr << "OBJ load : Warning : invalid vertex/normal id" << std::endl;
            }
        }
    }

    static void parse_mtl_lib_declaration(
        const std::string& line, const std::filesystem::path& entry_path,
        std::vector<material>& mtl_bank, std::map<std::string, int> &mtl_name_map)
    {
        char mtl_lib_filename[256];
        if (std::sscanf(line.c_str(), "mtllib %s\n", mtl_lib_filename) == 1)
        {
            load_mtl_lib(
                std::filesystem::path{entry_path}.replace_filename(mtl_lib_filename),
                mtl_bank, mtl_name_map);
        }
    }

    static void parse_mtl_use(const std::string& line, const std::map<std::string, int> &mtl_name_map, int& current_mtl)
    {
        char mtl_name[256];
        if (std::sscanf(line.c_str(), "usemtl %s\n", mtl_name) == 1)
        {
            const auto it = mtl_name_map.find(mtl_name);
            if (it != mtl_name_map.end())
            {
                current_mtl = it->second;
            }
            else
            {
                current_mtl = -1;
                std::cerr << "OBJ load : Warning mtl " << mtl_name << " not found" << std::endl;
            }
        }
    }

    model wavefront_obj_load(const std::filesystem::path& path)
    {
        std::ifstream stream{path};

        std::vector<face> geometry{};
        std::vector<float3> vertex{};
        std::vector<float3> normals{};

        std::vector<material> mtl_bank{};
        std::map<std::string, int> mtl_name_map{};

        int default_mtl = -1; // may not be needed
        int current_mtl = -1;

        if (!stream.is_open())
            throw std::invalid_argument("Unable to open file " + path.generic_string());

        //  Read line by line
        for (std::string line; std::getline(stream, line);)
        {
            switch (line[0])
            {
                case 'v':
                    parse_vector(line, vertex, normals);
                break;

                case 'f':
                {
                    if (current_mtl == -1)
                    {
                        if (default_mtl == -1)
                        {
                            // Create a default material if none was provided
                            default_mtl = mtl_bank.size();
                            mtl_bank.emplace_back(
                                make_lambertian_materal(float3{0.7, 0.7, 0.7}));
                        }
                        current_mtl = default_mtl;
                    }

                    parse_face(line, vertex, normals, current_mtl, geometry);
                }
                break;

                case 'm':
                    parse_mtl_lib_declaration(line, path, mtl_bank, mtl_name_map);
                break;

                case 'u':
                    parse_mtl_use(line, mtl_name_map, current_mtl);
                break;
            }
        }

        return model{
            std::move(geometry),
            std::move(mtl_bank)
        };
    }
}