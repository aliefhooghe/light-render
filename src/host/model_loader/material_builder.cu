
#include <stdexcept>

#include "material_builder.cuh"

namespace Xrender
{
    void material_builder::decl_new_mtl()
    {
        decl_mtl_type(material::LAMBERTIAN);
    }

    void material_builder::decl_mtl_type(mtl_type type)
    {
        // Set default value for this material model
        _current_mtl_type = type;
        switch (type)
        {
        case material::LAMBERTIAN:
            _kd = {0.8, 0.8, 0.8};
            break;
        case material::PHONG:
            _ks = {0.8, 0.8, 0.8};
            _ns = 50.f;
            break;
        case material::SOURCE:
            _temperature = 4500.0f;
            break;
        case material::MIRROR:
            _ks = {0.99f, 0.99f, 0.99f};
            break;
        case material::GLASS:
            _reflexivity = 0.1f;
            _tf = {0.8, 0.8, 0.8};
            _ni = 1.2f;
            break;
        case material::DISPERSIVE_GLASS:
            _reflexivity = 0.1f;
            _tf = {0.8, 0.8, 0.8};
            _cauchy_a = 1.2f;
            _cauchy_b = 1.3f;
            break;
        }
    }

    void material_builder::decl_ka(const float3 &ka)
    {
        _ka = ka;
    }

    void material_builder::decl_kd(const float3 &kd)
    {
        _kd = kd;
    }

    void material_builder::decl_ks(const float3 &ks)
    {
        _ks = ks;
    }

    void material_builder::decl_tf(const float3 &tf)
    {
        _tf = tf;
    }

    void material_builder::decl_ns(float ns)
    {
        _ns = ns;
    }

    void material_builder::decl_ni(float ni)
    {
        _ni = ni;
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
        case material::LAMBERTIAN:
            return make_lambertian_materal(_kd);
        case material::PHONG:
            return make_phong_material(_ks, _ns);
        case material::SOURCE:
            return make_source_material(_temperature);
        case material::MIRROR:
            return make_mirror_material(_ks);
        case material::GLASS:
            return make_glass_material(_reflexivity, _tf, _ks, _ni);
        case material::DISPERSIVE_GLASS:
            return make_dispersive_glass_material(_reflexivity, _tf, _ks, _cauchy_a, _cauchy_b);
        default:
            throw std::runtime_error("materail_builder : Invalid internal mtl type");
        }
    }

}