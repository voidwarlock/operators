#ifndef __COMMON_MACA_H__
#define __COMMON_MACA_H__

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_WARP_PER_BLOCK 32
#define WARP_SIZE 32

#include <iostream>

#define checkMacaErrorWithCode(call, errorCode)                       \
    do {                                                              \
        if (auto status = call; status != cudaSuccess) {              \
            std::cerr << "MACA error: " << hcGetErrorString(status) \
                      << " in file " << __FILE__                      \
                      << ", function " << __func__                    \
                      << ", line " << __LINE__ << std::endl;          \
            return errorCode;                                         \
        }                                                             \
    } while (0)

#define checkMacaError(call) checkMacaErrorWithCode(call, STATUS_BAD_DEVICE)

#define checkMcdnnError(call)                                           \
    do {                                                                \
        if (auto status = call; status != HCDNN_STATUS_SUCCESS) {       \
            std::cerr << "MCDNN error: " << hcdnnGetErrorString(status) \
                      << " in file " << __FILE__                        \
                      << ", function " << __func__                      \
                      << ", line " << __LINE__ << std::endl;            \
            return STATUS_EXECUTION_FAILED;                             \
        }                                                               \
    } while (0)

#include "data_type.h"
#include <hcdnn/hcdnn.h>

typedef struct DTMcdnnMapping {
    DT layout;
    hcdnnDataType_t hcdnn_type;
} DTMcdnnMapping;

// DT cudnnDataType_t mapping table
const DTMcdnnMapping dtMappings[] = {
    {F16, HCDNN_DATA_HALF},
    {F32, HCDNN_DATA_FLOAT},
    {F64, HCDNN_DATA_DOUBLE},
    {BF16, HCDNN_DATA_BFLOAT16},
    {I8, HCDNN_DATA_INT8},
    {I32, HCDNN_DATA_INT32},
    {I64, HCDNN_DATA_INT64},
    {U8, HCDNN_DATA_UINT8},
};

typedef struct DataLayoutMap {
    int operator[](const DataLayout &layout) const {
        for (const auto &mapping : dtMappings) {
            if (mapping.layout == layout) {
                return mapping.hcdnn_type;
            }
        }
        return -1;
    }
} DTMap;

constexpr DTMap dataTypeMap;

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

#endif// __COMMON_MACA_H__
