#ifndef __MUSA_CAUSAL_SOFTMAX_H__
#define __MUSA_CAUSAL_SOFTMAX_H__

#include "operators.h"
#include "../../../devices/musa/musa_handle.h"

struct CausalSoftmaxMusaDescriptor {
    Device device;
    int device_id;
    DT dtype;
    uint64_t batch_size;
    uint64_t stride_b;
    uint64_t seq_len;
    uint64_t stride_i;
    uint64_t total_seq_len;
    uint64_t stride_j;
    uint64_t max_items_per_thread;
};

typedef struct CausalSoftmaxMusaDescriptor *CausalSoftmaxMusaDescriptor_t;

infiniopStatus_t musaCreateCausalSoftmaxDescriptor(MusaHandle_t handle,
                                                   CausalSoftmaxMusaDescriptor_t *desc_ptr,
                                                   infiniopTensorDescriptor_t y_desc);

infiniopStatus_t musaGetCausalSoftmaxWorkspaceSize(CausalSoftmaxMusaDescriptor_t desc, uint64_t *size);

infiniopStatus_t musaCausalSoftmax(CausalSoftmaxMusaDescriptor_t desc,
                                   void *workspace,
                                   uint64_t workspace_size,
                                   void *data,
                                   void *stream);

infiniopStatus_t musaDestroyCausalSoftmaxDescriptor(CausalSoftmaxMusaDescriptor_t desc);
#endif
