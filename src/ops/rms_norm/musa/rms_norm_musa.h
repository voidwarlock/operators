#ifndef __MUSA_RMS_NORM_H__
#define __MUSA_RMS_NORM_H__

#include "operators.h"
#include "../../../devices/musa/musa_handle.h"

struct RMSNormMusaDescriptor {
    Device device;
    int device_id;
    DT dtype;
    uint64_t n;
    uint64_t d;
    uint64_t stride_y;
    uint64_t stride_x;
    DT w_datatype;
    float epsilon;
};

typedef struct RMSNormMusaDescriptor *RMSNormMusaDescriptor_t;

infiniopStatus_t musaCreateRMSNormDescriptor(MusaHandle_t handle,
                                             RMSNormMusaDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t y_desc,
                                             infiniopTensorDescriptor_t x_desc,
                                             infiniopTensorDescriptor_t w_desc,
                                             float epsilon);

infiniopStatus_t musaGetRMSNormWorkspaceSize(RMSNormMusaDescriptor_t desc, uint64_t *size);

infiniopStatus_t musaRMSNorm(RMSNormMusaDescriptor_t desc,
                                   void *workspace,
                                   uint64_t workspace_size,
                                   void *y, void const *x, void const *w,
                                   void *stream);

infiniopStatus_t musaDestroyRMSNormDescriptor(RMSNormMusaDescriptor_t desc);

void rms_norm_mt_gpu_f16(RMSNormMusaDescriptor_t desc, void *y, void const *x, void const *w, float epsilon, void *stream);

#endif// __MT_GPU_RMS_NORM_H__
