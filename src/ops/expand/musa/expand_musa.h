#ifndef __MUSA_EXPAND_H__
#define __MUSA_EXPAND_H__

#include "../../../devices/musa/common_musa.h"
#include "../../../devices/musa/musa_handle.h"
#include "operators.h"
#include <musa_fp16.h>
#include <numeric>

struct ExpandMusaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    uint64_t ndim;
    uint64_t y_data_size;
    uint64_t max_grid_size;
    char const *strides_and_shape_d;
};

typedef struct ExpandMusaDescriptor *ExpandMusaDescriptor_t;

infiniopStatus_t musaCreateExpandDescriptor(MusaHandle_t,
                                            ExpandMusaDescriptor_t *,
                                            infiniopTensorDescriptor_t y,
                                            infiniopTensorDescriptor_t x);

infiniopStatus_t musaExpand(ExpandMusaDescriptor_t desc,
                            void *y, void const *x,
                            void *stream);

infiniopStatus_t musaDestroyExpandDescriptor(ExpandMusaDescriptor_t desc);

#endif
