#include "add_musa.h"
#include "../../../devices/musa/common_musa.h"
#include "../../utils.h"

infiniopStatus_t musaCreateAddDescriptor(MusaHandle_t handle,
                                         AddMusaDescriptor_t *desc_ptr,
                                         infiniopTensorDescriptor_t c,
                                         infiniopTensorDescriptor_t a,
                                         infiniopTensorDescriptor_t b) {
    uint64_t ndim = c->ndim;
    if (!isValidBroadcastShape(a, b, c)) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (!is_contiguous(a) || !is_contiguous(b) || !is_contiguous(c)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (c->dt != F16 && c->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (c->dt != a->dt || c->dt != b->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    bool broadcasted = false;
    if (ndim != a->ndim || ndim != b->ndim) {
        broadcasted = true;
    } else {
        for (uint64_t i = 0; i < ndim; ++i) {
            if (c->shape[i] != a->shape[i] || c->shape[i] != b->shape[i]) {
                broadcasted = true;
                break;
            }
        }
    }

    uint64_t c_data_size = std::accumulate(c->shape, c->shape + c->ndim, 1ULL, std::multiplies<uint64_t>());

    // get the adjusted strides for a and b
    int64_t *a_strides = new int64_t[ndim];
    int64_t *b_strides = new int64_t[ndim];
    for (size_t i = 0; i < ndim; ++i) {
        a_strides[i] = (i < ndim - a->ndim || c->shape[i] != a->shape[i + a->ndim - ndim]) ? 0 : a->strides[i + a->ndim - ndim];
        b_strides[i] = (i < ndim - b->ndim || c->shape[i] != b->shape[i + b->ndim - ndim]) ? 0 : b->strides[i + b->ndim - ndim];
    }

    musaDeviceProp prop;
    musaGetDeviceProperties(&prop, handle->device_id);

    int64_t *a_strides_d, *b_strides_d, *c_strides_d;
    checkMusaErrorWithCode(musaMalloc(&a_strides_d, ndim * sizeof(int64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkMusaErrorWithCode(musaMalloc(&b_strides_d, ndim * sizeof(int64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkMusaErrorWithCode(musaMalloc(&c_strides_d, ndim * sizeof(int64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkMusaErrorWithCode(musaMemcpy(a_strides_d, a_strides, ndim * sizeof(int64_t), musaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
    checkMusaErrorWithCode(musaMemcpy(b_strides_d, b_strides, ndim * sizeof(int64_t), musaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
    checkMusaErrorWithCode(musaMemcpy(c_strides_d, c->strides, ndim * sizeof(int64_t), musaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);

    *desc_ptr = new AddMusaDescriptor{
        DevMtGpu,
        c->dt,
        handle->device_id,
        ndim,
        c_data_size,
        static_cast<uint64_t>(prop.maxGridSize[0]),
        a_strides_d,
        b_strides_d,
        c_strides_d,
        broadcasted,
    };

    delete[] a_strides;
    delete[] b_strides;

    return STATUS_SUCCESS;
}

infiniopStatus_t musaDestroyAddDescriptor(AddMusaDescriptor_t desc) {
    checkMusaErrorWithCode(musaFree((void *) desc->a_strides), STATUS_EXECUTION_FAILED);
    checkMusaErrorWithCode(musaFree((void *) desc->b_strides), STATUS_EXECUTION_FAILED);
    checkMusaErrorWithCode(musaFree((void *) desc->c_strides), STATUS_EXECUTION_FAILED);
    delete desc;
    return STATUS_SUCCESS;
}
