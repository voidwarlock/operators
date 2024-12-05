#ifndef __MUSA_RELU_H__
#define __MUSA_RELU_H__

#include "../../../devices/musa/common_musa.h"
#include "../../../devices/musa/musa_handle.h"
#include "operators.h"
#include <musa_fp16.h>
#include <numeric>

struct ReluMusaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    uint64_t ndim;
    uint64_t data_size;
    uint64_t max_grid_size;
};

typedef struct ReluMusaDescriptor *ReluMusaDescriptor_t;

infiniopStatus_t musaCreateReluDescriptor(MusaHandle_t,
                                          ReluMusaDescriptor_t *,
                                          infiniopTensorDescriptor_t y,
                                          infiniopTensorDescriptor_t x);

infiniopStatus_t musaRelu(ReluMusaDescriptor_t desc,
                          void *y, void const *x,
                          void *stream);

infiniopStatus_t musaDestroyReluDescriptor(ReluMusaDescriptor_t desc);

#endif
