#ifndef __COMMON_MUSA_H__
#define __COMMON_MUSA_H__

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_WARP_PER_BLOCK 32
#define WARP_SIZE 32

#include <iostream>
#include "data_type.h"
#include <musa.h>
#include <musa_runtime_api.h>
#include <mudnn.h>

enum class Type {
    QINT4,
    QINT8,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    HALF,
    BFLOAT16,
    FLOAT,
    DOUBLE,
    BOOL,
};

enum class Format {
    UNKNOWN,
    NCW,
    NWC,
    NCHW,
    NHWC,
    HWCN,
    NCDHW,
    NDHWC,
    DHWCN,
};

#define checkMusaErrorWithCode(call, errorCode)                       \
    do {                                                              \
        if (auto status = call; status != musaSuccess) {              \
            std::cerr << "MUSA error: " << musaGetErrorString(status) \
                      << " in file " << __FILE__                      \
                      << ", function " << __func__                    \
                      << ", line " << __LINE__ << std::endl;          \
            return errorCode;                                         \
        }                                                             \
    } while (0)

#define checkMusaError(call) checkMusaErrorWithCode(call, STATUS_BAD_DEVICE)

// get the corresponding offset in the destination given the flat index of the source (for element mapping in shape broadcast)
inline __device__ uint64_t getDstOffset(uint64_t flat_index, uint64_t ndim, int64_t const *src_strides, int64_t const *dst_strides) {
    uint64_t res = 0;
    for (uint64_t i = 0; i < ndim; ++i) {
        res += flat_index / src_strides[i] * dst_strides[i];
        flat_index %= src_strides[i];
    }
    return res;
}

// get the memory offset of the given element in a tensor given its flat index
inline __device__ uint64_t getOffset(uint64_t flat_index, uint64_t ndim, uint64_t const *shape, int64_t const *strides) {
    uint64_t res = 0;
    for (long i = ndim - 1; i >= 0; --i) {
        res += (flat_index % shape[i]) * strides[i];
        flat_index /= shape[i];
    }
    return res;
}

#endif // __COMMON_MUSA_H__