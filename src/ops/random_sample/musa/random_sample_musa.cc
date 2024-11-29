#include "../../../devices/musa/common_musa.h"
#include "../../utils.h"
#include "random_sample_musa.h"

infiniopStatus_t musaCreateRandomSampleDescriptor(MusaHandle_t handle,
                                                  RandomSampleMusaDescriptor_t *desc_ptr, infiniopTensorDescriptor_t result,
                                                  infiniopTensorDescriptor_t probs) {
    if (probs->ndim != 1) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (!dtype_eq(result->dt, U64))
        return STATUS_BAD_TENSOR_DTYPE;
    int voc = probs->shape[0];
    int rLength = result->shape[0];
    if (result->ndim != 1 && rLength != 1) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    *desc_ptr = new RandomSampleMusaDescriptor{
        handle->device,
        handle->device_id,
        probs->dt,
        voc,
        result->dt,
        rLength};

    return STATUS_SUCCESS;
}

infiniopStatus_t musaGetRandomSampleWorkspaceSize(RandomSampleMusaDescriptor_t desc, unsigned long int *size) {
    *size = desc->voc * (2 * sizeof(uint64_t) + sizeof(desc->dtype));
    return STATUS_SUCCESS;
}

infiniopStatus_t musaDestroyRandomSampleDescriptor(RandomSampleMusaDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}
