#ifndef __MACA_RMS_NORM_H__
#define __MACA_RMS_NORM_H__

#include "../../../devices/maca/maca_handle.h"
#include "operators.h"

struct RMSNormMacaDescriptor {
    Device device;
    int device_id;
    DT dtype;
    uint64_t n;
    uint64_t d;
    int64_t stride_y;
    int64_t stride_x;
    DT w_datatype;
    float epsilon;
};

typedef struct RMSNormMacaDescriptor *RMSNormMacaDescriptor_t;

infiniopStatus_t macaCreateRMSNormDescriptor(MacaHandle_t handle,
                                             RMSNormMacaDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t y_desc,
                                             infiniopTensorDescriptor_t x_desc,
                                             infiniopTensorDescriptor_t w_desc,
                                             float epsilon);

infiniopStatus_t macaGetRMSNormWorkspaceSize(RMSNormMacaDescriptor_t desc, uint64_t *size);

infiniopStatus_t macaRMSNorm(RMSNormMacaDescriptor_t desc,
                             void *workspace,
                             uint64_t workspace_size,
                             void *y, void const *x, void const *w,
                             void *stream);

infiniopStatus_t macaDestroyRMSNormDescriptor(RMSNormMacaDescriptor_t desc);

void rms_norm_mc_gpu_f16(RMSNormMacaDescriptor_t desc, void *y, void const *x, void const *w, float epsilon, void *stream);

#endif// __MACA_RMS_NORM_H__
