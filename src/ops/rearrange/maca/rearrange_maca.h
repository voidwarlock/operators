#ifndef __MACA_REARRANGE_H__
#define __MACA_REARRANGE_H__

#include "../../../devices/maca/maca_handle.h"
#include "operators.h"

struct RearrangeMacaDescriptor {
    Device device;
    int device_id;
    uint64_t unit, r, c;
    int64_t dst_rs, dst_cs, src_rs, src_cs;
};

typedef struct RearrangeMacaDescriptor *RearrangeMacaDescriptor_t;

infiniopStatus_t macaCreateRearrangeDescriptor(MacaHandle_t handle,
                                               RearrangeMacaDescriptor_t *desc_ptr,
                                               infiniopTensorDescriptor_t dst,
                                               infiniopTensorDescriptor_t src);

infiniopStatus_t macaRearrange(RearrangeMacaDescriptor_t desc,
                               void *dst,
                               void const *src,
                               void *stream);

infiniopStatus_t macaDestroyRearrangeDescriptor(RearrangeMacaDescriptor_t desc);

void rearrange_mc_gpu(RearrangeMacaDescriptor_t, void *y, void const *x, void *stream);
#endif// __MACA_REARRANGE_H__
