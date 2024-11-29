#include "../../../devices/musa/musa_handle.h"
#include "../../utils.h"
#include "../blas.h"
#include "matmul_musa.h"
#include <mublas.h>
#include <musa_fp16.h>

template<typename Tdata>
infiniopStatus_t matmul_musa(MatmulMusaDescriptor_t desc, void *c, float beta, void const *a, void const *b, float alpha, void *stream) {
    auto info = desc->info;

    if (info.is_transed) {
        std::swap(a, b);
    }

    Tdata alpha_, beta_;
    musaDataType_t a_type, b_type, c_type;
    mublasComputeType_t compute_type;

    if constexpr (std::is_same<Tdata, half>::value) {
        alpha_ = __float2half(alpha);
        beta_ = __float2half(beta);
        a_type = b_type = c_type = MUSA_R_16F;
        compute_type = MUBLAS_COMPUTE_16F;
    } else {
        alpha_ = alpha;
        beta_ = beta;
        a_type = b_type = c_type = MUSA_R_32F;
        compute_type = MUBLAS_COMPUTE_32F_FAST_TF32;
    }

    auto op_a = info.a_matrix.row_stride == 1 ? MUBLAS_OP_N : MUBLAS_OP_T;
    auto op_b = info.b_matrix.row_stride == 1 ? MUBLAS_OP_N : MUBLAS_OP_T;

    use_mublas(desc->mublas_handles_t, desc->device_id, (MUstream) stream,
               [&](mublasHandle_t handle) { mublasGemmStridedBatchedEx(
                                                handle,
                                                op_a,
                                                op_b,
                                                info.m,
                                                info.n,
                                                info.k,
                                                &alpha_,
                                                a,
                                                a_type,
                                                info.a_matrix.ld(),
                                                info.a_matrix.stride,
                                                b,
                                                b_type,
                                                info.b_matrix.ld(),
                                                info.b_matrix.stride,
                                                &beta_,
                                                c,
                                                c_type,
                                                info.c_matrix.ld(),
                                                info.c_matrix.stride,
                                                info.batch,
                                                compute_type,
                                                MUBLAS_GEMM_DEFAULT);});
    return STATUS_SUCCESS;
}

infiniopStatus_t musaMatmul(MatmulMusaDescriptor_t desc,
                            void *workspace,
                            uint64_t workspace_size,
                            void *c,
                            void const *a,
                            void const *b,
                            void *stream) {
    if (desc->dtype == F16) {
        return matmul_musa<half>(desc, c, desc->beta, a, b, desc->alpha, stream);
    }
    if (desc->dtype == F32) {
        return matmul_musa<float>(desc, c, desc->beta, a, b, desc->alpha, stream);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}