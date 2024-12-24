#ifndef __METAX_GPU_ROTARY_EMBEDDING_H__
#define __METAX_GPU_ROTARY_EMBEDDING_H__

#include "../../../devices/maca/maca_handle.h"
#include "operators.h"

struct RoPEMacaDescriptor {
    Device device;
    int device_id;
    DT dtype;
    uint64_t seq_len;
    uint64_t nhead;
    uint64_t dim;
    uint64_t total_seq_len;
    int64_t strides[2];
};

typedef struct RoPEMacaDescriptor *RoPEMacaDescriptor_t;

infiniopStatus_t macaCreateRoPEDescriptor(MacaHandle_t handle,
                                          RoPEMacaDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t t,
                                          infiniopTensorDescriptor_t pos_ids,
                                          infiniopTensorDescriptor_t sin_table,
                                          infiniopTensorDescriptor_t cos_table);

infiniopStatus_t macaGetRoPEWorkspaceSize(RoPEMacaDescriptor_t desc, uint64_t *size);

infiniopStatus_t macaRoPE(RoPEMacaDescriptor_t desc,
                          void *workspace,
                          uint64_t workspace_size,
                          void *t,
                          void const *pos_ids,
                          void const *sin_table,
                          void const *cos_table,
                          void *stream);

infiniopStatus_t macaDestroyRoPEDescriptor(RoPEMacaDescriptor_t desc);

#endif// __METAX_GPU_ROTARY_EMBEDDING_H__
