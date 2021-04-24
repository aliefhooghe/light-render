#ifndef CUDA_EXCEPTION_CUH
#define CUDA_EXCEPTION_CUH

#include <cuda.h>

#define CUDA_CHECK(call)               \
    {                                  \
        const auto err = (call);       \
        if (cudaSuccess != err)        \
            throw cuda_exception(err); \
    }

namespace Xrender {

    class cuda_exception : public std::exception {

    public:
        cuda_exception(cudaError_t error) noexcept
        : _error{error}
        {}

        const char *what() const noexcept override
        {
            return cudaGetErrorString(_error);
        }

        private:
            const cudaError_t _error;
    };

}

#endif