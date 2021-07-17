
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

    static void load_mtl_lib(const std::filesystem::path &path, std::map<std::string, material> &material_map)
    {
        std::ifstream stream{path};
        material_builder mtl_builder{};
        std::string current_mtl_name{};

        if (!stream.is_open())
            throw std::invalid_argument("mtllib path");

        for (std::string line; std::getline(stream, line);)
        {
            switch (line[0])
            {
            case 'n':
            {
                char mtl_name[256];
                if (!current_mtl_name.empty())
                    material_map[current_mtl_name] = mtl_builder.make_material();
                if (sscanf(line.c_str(), "newmtl %s\n", mtl_name) == 1)
                {
                    mtl_builder.decl_new_mtl();
                    current_mtl_name = mtl_name;
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
            material_map[current_mtl_name] = mtl_builder.make_material();
    }


    std::vector<face> wavefront_obj_load(const std::filesystem::path& path)
    {
        std::vector<face> model{};
        std::vector<float3> vertex{};
        std::vector<float3> normals{};
        std::ifstream stream{path};

        std::map<std::string, material> material_map{};
        material current_mtl;

        if (!stream.is_open())
            throw std::invalid_argument("obj path");

        //  Read line by line
        for (std::string line; std::getline(stream, line);)
        {
            switch (line[0])
            {
                case 'v':
                {
                    float x, y, z;
                    if (std::sscanf(line.c_str(), "v %f %f %f", &x, &y, &z) == 3)
                        vertex.emplace_back(float3{x, y, z});
                    else if (std::sscanf(line.c_str(), "vn %f %f %f", &x, &y, &z) == 3)
                        normals.emplace_back(float3{x, y, z});
                }
                break;

                case 'f':
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
                            model.push_back(
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
                break;

                case 'm':
                {
                    char mtl_lib_filename[256];
                    if (std::sscanf(line.c_str(), "mtllib %s\n", mtl_lib_filename) == 1)
                    {
                        load_mtl_lib(
                            std::filesystem::path{path}.replace_filename(mtl_lib_filename),
                            material_map);
                    }
                }
                break;

                case 'u':
                {
                    char mtl_name[256];
                    if (std::sscanf(line.c_str(), "usemtl %s\n", mtl_name) == 1)
                    {
                        const auto it = material_map.find(mtl_name);
                        if (it != material_map.end())
                            current_mtl = it->second;
                        else
                            std::cerr << "OBJ load : Warning mtl " << mtl_name << " not found" << std::endl;
                    }
                }
                break;
            }
        }

        return model;
    }
}