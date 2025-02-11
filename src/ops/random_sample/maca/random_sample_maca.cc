#include "../../../devices/maca/common_maca.h"
#include "../../utils.h"
#include "random_sample_maca.h"

infiniopStatus_t macaCreateRandomSampleDescriptor(MacaHandle_t handle,
                                                  RandomSampleMacaDescriptor_t *desc_ptr, infiniopTensorDescriptor_t result,
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
    *desc_ptr = new RandomSampleMacaDescriptor{
        handle->device,
        handle->device_id,
        probs->dt,
        voc,
        result->dt,
        rLength};

    return STATUS_SUCCESS;
}

infiniopStatus_t macaGetRandomSampleWorkspaceSize(RandomSampleMacaDescriptor_t desc, uint64_t *size) {
    *size = desc->voc * (2 * sizeof(uint64_t) + sizeof(desc->dtype));
    return STATUS_SUCCESS;
}

infiniopStatus_t macaDestroyRandomSampleDescriptor(RandomSampleMacaDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}
