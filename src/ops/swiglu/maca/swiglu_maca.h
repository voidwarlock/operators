#ifndef __MACA_SWIGLU_H__
#define __MACA_SWIGLU_H__
#include "../../../devices/maca/maca_handle.h"
#include "../../utils.h"
#include "operators.h"

struct SwiGLUMacaDescriptor {
    Device device;
    DT dtype;
    uint64_t seq_len;
    uint64_t di;
    uint64_t stride_a;
    uint64_t stride_b;
    uint64_t stride_c;
};

typedef struct SwiGLUMacaDescriptor *SwiGLUMacaDescriptor_t;

infiniopStatus_t macaCreateSwiGLUDescriptor(MacaHandle_t handle,
                                            SwiGLUMacaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t c_dec,
                                            infiniopTensorDescriptor_t a_desc,
                                            infiniopTensorDescriptor_t b_desc);

infiniopStatus_t macaSwiGLU(SwiGLUMacaDescriptor_t desc,
                            void *c,
                            void const *a,
                            void const *b,
                            void *stream);

infiniopStatus_t macaDestroySwiGLUDescriptor(SwiGLUMacaDescriptor_t desc);

void swiglu_mc_gpu_f16(SwiGLUMacaDescriptor_t desc, void *c, void const *a, void const *b, void *stream);

#endif// __MC_GPU_SWIGLU_H__
