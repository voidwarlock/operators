#ifndef __MACA_CAUSAL_SOFTMAX_H__
#define __MACA_CAUSAL_SOFTMAX_H__

#include "../../../devices/maca/maca_handle.h"
#include "operators.h"

struct CausalSoftmaxMacaDescriptor {
    Device device;
    int device_id;
    DT dtype;
    uint64_t batch_size;
    uint64_t stride_b;
    uint64_t seq_len;
    uint64_t stride_i;
    uint64_t total_seq_len;
    uint64_t stride_j;
    unsigned int max_items_per_thread;
};

typedef struct CausalSoftmaxMacaDescriptor *CausalSoftmaxMacaDescriptor_t;

infiniopStatus_t macaCreateCausalSoftmaxDescriptor(MacaHandle_t handle,
                                                   CausalSoftmaxMacaDescriptor_t *desc_ptr,
                                                   infiniopTensorDescriptor_t y_desc);

infiniopStatus_t macaGetCausalSoftmaxWorkspaceSize(CausalSoftmaxMacaDescriptor_t desc, uint64_t *size);

infiniopStatus_t macaCausalSoftmax(CausalSoftmaxMacaDescriptor_t desc,
                                   void *workspace,
                                   uint64_t workspace_size,
                                   void *data,
                                   void *stream);

infiniopStatus_t macaDestroyCausalSoftmaxDescriptor(CausalSoftmaxMacaDescriptor_t desc);

#endif
