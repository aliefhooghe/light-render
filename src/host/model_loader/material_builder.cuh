#ifndef XRENDER_MATERIAL_BUILDER_H_
#define XRENDER_MATERIAL_BUILDER_H_

#include "gpu/model/material.cuh"

namespace Xrender
{
    class material_builder
    {

    public:
        using mtl_type = material::mtl_type;

        void decl_new_mtl();
        void decl_mtl_type(mtl_type);
        void decl_ka(const float3 &kd);
        void decl_kd(const float3 &kd);
        void decl_ks(const float3 &kd);
        void decl_tf(const float3 &tf);
        void decl_ns(float ni);
        void decl_ni(float ns);
        void decl_temperature(float t);
        void decl_cauchy_a(float a);
        void decl_cauchy_b(float b);
        void decl_reflexivity(float r);
        material make_material();

    private:
        mtl_type _current_mtl_type{material::LAMBERTIAN};

        float3 _ka;
        float3 _kd;
        float3 _ks;
        float3 _tf;
        float _ni;
        float _ns;
        float _temperature;
        float _cauchy_a;
        float _cauchy_b;
        float _reflexivity;
    };
}

#endif