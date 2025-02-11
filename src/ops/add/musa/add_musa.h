#ifndef __MUSA_ADD_H__
#define __MUSA_ADD_H__

#include "../../../devices/musa/common_musa.h"
#include "../../../devices/musa/musa_handle.h"
#include "operators.h"
#include <musa_fp16.h>
#include <numeric>

struct AddMusaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    uint64_t ndim;
    uint64_t c_data_size;
    uint64_t max_grid_size;
    int64_t const *a_strides;
    int64_t const *b_strides;
    int64_t const *c_strides;
    bool broadcasted;
};

typedef struct AddMusaDescriptor *AddMusaDescriptor_t;

infiniopStatus_t musaCreateAddDescriptor(MusaHandle_t,
                                         AddMusaDescriptor_t *,
                                         infiniopTensorDescriptor_t c,
                                         infiniopTensorDescriptor_t a,
                                         infiniopTensorDescriptor_t b);

infiniopStatus_t musaAdd(AddMusaDescriptor_t desc,
                         void *c, void const *a, void const *b,
                         void *stream);

infiniopStatus_t musaDestroyAddDescriptor(AddMusaDescriptor_t desc);

#endif
