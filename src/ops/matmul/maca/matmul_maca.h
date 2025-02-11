#ifndef __MACA_MATMUL_H__
#define __MACA_MATMUL_H__

#include "../../../devices/maca/maca_handle.h"
#include "../blas.h"
#include "operators.h"
#include <memory>

typedef struct MatmulMacaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    MatmulInfo info;
    float alpha;
    float beta;
    std::shared_ptr<Pool<hcblasHandle_t>> mcblas_handles_t;
} MatmulMacaDescriptor;

typedef struct MatmulMacaDescriptor *MatmulMacaDescriptor_t;

infiniopStatus_t macaCreateMatmulDescriptor(MacaHandle_t handle,
                                            MatmulMacaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t c_desc,
                                            float alpha,
                                            infiniopTensorDescriptor_t a_desc,
                                            infiniopTensorDescriptor_t b_desc,
                                            float beta);

infiniopStatus_t macaGetMatmulWorkspaceSize(MatmulMacaDescriptor_t desc, uint64_t *size);

infiniopStatus_t macaMatmul(MatmulMacaDescriptor_t desc,
                            void *workspace,
                            uint64_t workspace_size,
                            void *c,
                            void const *a,
                            void const *b,
                            void *stream);

infiniopStatus_t macaDestroyMatmulDescriptor(MatmulMacaDescriptor_t desc);

#endif// __MACA_MATMUL_H__
