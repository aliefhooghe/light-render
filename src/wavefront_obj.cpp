
#include <fstream>
#include <stdexcept>
#include <cstdio>
#include <map>
#include <iostream>

#include "wavefront_obj.h"

namespace Xrender
{

    class material_builder
    {

    public:
        enum mtl_type
        {
            SOURCE,
            LAMBERTIAN,
            MIRROR,
            GLASS
        };

        void decl_new_mtl();
        void decl_mtl_type(mtl_type);
        void decl_ka(const vecf& kd);
        void decl_kd(const vecf& kd);
        void decl_ks(const vecf& kd);
        void decl_tf(const vecf& tf);
        void decl_ns(float ni);
        void decl_ni(float ns);
        void decl_temperature(float t);
        void decl_cauchy_a(float a);
        void decl_cauchy_b(float b);
        void decl_reflexivity(float r);
        material make_material();

    private:
        mtl_type _current_mtl_type{LAMBERTIAN};

        vecf _ka;
        vecf _kd;
        vecf _ks;
        vecf _tf;
        float _ni;
        float _ns;
        float _temperature;
        float _cauchy_a;
        float _cauchy_b;
        float _reflexivity;
    };

    void material_builder::decl_new_mtl()
    {
        decl_mtl_type(LAMBERTIAN);
    }

    void material_builder::decl_mtl_type(mtl_type type)
    {
        // Set default value for this material model
        _current_mtl_type = type;
        switch (type)
        {
        case LAMBERTIAN:
            _kd = {0.8, 0.8, 0.8};
            break;
        case SOURCE:
            _temperature = 4000.0f;
            break;
        case MIRROR:
            _ks = {0.99f, 0.99f, 0.99f};
            break;
        case GLASS:
            _reflexivity = 0.1f;
            _tf = {0.8, 0.8, 0.8};
            _cauchy_a = 1.2f;
            _cauchy_b = 1.3f;
            break;
        }
    }

    void material_builder::decl_ka(const vecf &ka)
    {
        _ka = ka;
    }

    void material_builder::decl_kd(const vecf &kd)
    {
        _kd = kd;
    }

    void material_builder::decl_ks(const vecf &ks)
    {
        _ks = ks;
    }

    void material_builder::decl_tf(const vecf& tf)
    {
        _tf = tf;
    }

    void material_builder::decl_ns(float ni)
    {
        _ni = ni;
    }

    void material_builder::decl_ni(float ns)
    {
        _ns = ns;
    }

    void material_builder::decl_temperature(float t)
    {
        _temperature = t;
    }

    void material_builder::decl_cauchy_a(float a)
    {
        _cauchy_a = a;
    }

    void material_builder::decl_cauchy_b(float b)
    {
        _cauchy_b = b;
    }

    void material_builder::decl_reflexivity(float r)
    {
        _reflexivity = r;
    }

    material material_builder::make_material()
    {
        switch (_current_mtl_type)
        {
        case LAMBERTIAN:
            return make_lambertian_material(_kd);
            break;
        case SOURCE:
            return make_source_material(_temperature);
            break;
        case MIRROR:
            return make_mirror_material(_ks);
            break;
        case GLASS:
            return make_glass_material(_reflexivity, _tf, _ks, _cauchy_a, _cauchy_b);
            break;
        default:
            throw std::runtime_error("FATAL ERROR");
        }
    }

    static void load_mtl_lib(const std::filesystem::path& path, std::map<std::string, material> &material_map)
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

                case 'N':
                {
                    // TODO
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
                    if      (line.rfind("#Source", 0) == 0) mtl_builder.decl_mtl_type(material_builder::SOURCE);
                    else if (line.rfind("#Lambertian", 0) == 0) mtl_builder.decl_mtl_type(material_builder::LAMBERTIAN);
                    else if (line.rfind("#Mirror", 0) == 0) mtl_builder.decl_mtl_type(material_builder::MIRROR);
                    else if (line.rfind("#Glass", 0) == 0) mtl_builder.decl_mtl_type(material_builder::GLASS);
                    else
                    {
                        float value;
                        switch (line[1])
                        {
                            case 'T': if (std::sscanf(line.c_str(), "#T %f", &value) == 1) mtl_builder.decl_temperature(value);
                            case 'A': if (std::sscanf(line.c_str(), "#A %f", &value) == 1) mtl_builder.decl_cauchy_a(value);
                            case 'B': if (std::sscanf(line.c_str(), "#B %f", &value) == 1) mtl_builder.decl_cauchy_b(value);
                            case 'R': if (std::sscanf(line.c_str(), "#R %f", &value) == 1) mtl_builder.decl_reflexivity(value);
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
        std::vector<vecf> vertex{};
        std::vector<vecf> normals{};
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
                        vertex.emplace_back(vecf{x, y, z});
                    else if (std::sscanf(line.c_str(), "vn %f %f %f", &x, &y, &z) == 3)
                        normals.emplace_back(vecf{x, y, z});
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

} /* namespace Xrender */