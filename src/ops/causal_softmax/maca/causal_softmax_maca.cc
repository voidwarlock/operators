#include "causal_softmax_maca.h"
#include "../../../devices/maca/common_maca.h"
#include "../../utils.h"

infiniopStatus_t macaCreateCausalSoftmaxDescriptor(MacaHandle_t handle,
                                                   CausalSoftmaxMacaDescriptor_t *desc_ptr,
                                                   infiniopTensorDescriptor_t y) {
    uint64_t ndim = y->ndim;
    // TODO: only support 2d or 3d tensor
    if (ndim != 2 && ndim != 3) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (!dtype_eq(y->dt, F16)) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    uint64_t total_seq_len = y->shape[ndim - 1];
    uint64_t seq_len = y->shape[ndim - 2];
    uint64_t batch_size = 1;
    uint64_t stride_b = 0;
    uint64_t stride_i = y->strides[ndim - 2];
    uint64_t stride_j = y->strides[ndim - 1];
    if (stride_j != 1) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    for (int i = 0; i < ndim - 2; i++) {
        batch_size *= y->shape[i];
    }
    if (ndim == 3)
        stride_b = y->strides[ndim - 3];
    unsigned int max_items_per_thread = ROUND_UP_DIV(total_seq_len, MAX_THREADS_PER_BLOCK);

    *desc_ptr = new CausalSoftmaxMacaDescriptor{
        handle->device,
        handle->device_id,
        y->dt,
        batch_size,
        stride_b,
        seq_len,
        stride_i,
        total_seq_len,
        stride_j,
        max_items_per_thread};

    return STATUS_SUCCESS;
}

infiniopStatus_t macaGetCausalSoftmaxWorkspaceSize(CausalSoftmaxMacaDescriptor_t desc, uint64_t *size) {
    *size = 0;
    return STATUS_SUCCESS;
}

infiniopStatus_t macaDestroyCausalSoftmaxDescriptor(CausalSoftmaxMacaDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}
