#ifndef CUDA_EXCEPTION_CUH
#define CUDA_EXCEPTION_CUH

#include <iostream>
#include <stdexcept>
#include <cuda.h>

#define CUDA_EXCEPTION_STR2(x) #x
#define CUDA_EXCEPTION_STR(x) CUDA_EXCEPTION_STR2(x)

#define CUDA_CHECK(call)               \
    {                                  \
        const auto err = (call);       \
        if (cudaSuccess != err)        \
            throw cuda_exception("==> " #call "\n at " __FILE__ ":" CUDA_EXCEPTION_STR(__LINE__), err); \
    }

#define CUDA_WARNING(call)             \
    {                                  \
        const auto err = (call);       \
        if (cudaSuccess != err)        \
            std::cerr << "Warning '" << #call \
            "' returned an error \nin " \
            __FILE__ ":" CUDA_EXCEPTION_STR(__LINE__) "\n" \
            << cudaGetErrorString(err) << std::endl; \
    }


namespace Xrender {

    class cuda_exception : public std::exception {

    public:
        cuda_exception(const char *call, cudaError_t error) noexcept
        : _msg{std::string{call} + "\nerror : " +cudaGetErrorString(error)}
        {}

        const char *what() const noexcept override
        {
            return _msg.c_str();
        }

        private:
            const std::string _msg;
    };

}

#endif