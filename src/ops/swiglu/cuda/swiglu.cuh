#ifndef __NV_GPU_SWIGLU_H__
#define __NV_GPU_SWIGLU_H__

#include "../../../operators.h"

void swiglu_nv_gpu_f16(MutTensor gate, ConstTensor up, void *stream);

#endif// __NV_GPU_SWIGLU_H__
