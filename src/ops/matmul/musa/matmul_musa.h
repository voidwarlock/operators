#ifndef __MUSA_MATMUL_H__
#define __MUSA_MATMUL_H__

#include <memory>
#include <musa.h>
#include <musa_runtime.h>
#include <mudnn.h>
#include <mudnn_base.h>
#include "../blas.h"
#include "operators.h"
#include "../../../devices/musa/musa_handle.h"

typedef struct MatmulMusaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    MatmulInfo info;
    float alpha;
    float beta;
    std::shared_ptr<Pool<mublasHandle_t>> mublas_handles_t;
} MatmulMusaDescriptor;

typedef struct MatmulMusaDescriptor *MatmulMusaDescriptor_t;

infiniopStatus_t musaCreateMatmulDescriptor(MusaHandle_t handle,
                                            MatmulMusaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t c_desc,
                                            float alpha,
                                            infiniopTensorDescriptor_t a_desc,
                                            infiniopTensorDescriptor_t b_desc,
                                            float beta);

infiniopStatus_t musaGetMatmulWorkspaceSize(MatmulMusaDescriptor_t desc, uint64_t *size);

infiniopStatus_t musaMatmul(MatmulMusaDescriptor_t desc,
                            void *workspace,
                            uint64_t workspace_size,
                            void *c,
                            void const *a,
                            void const *b,
                            void *stream);

infiniopStatus_t musaDestroyMatmulDescriptor(MatmulMusaDescriptor_t desc);

#endif // __MUSA_MATMUL_H__