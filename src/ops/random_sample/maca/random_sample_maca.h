#ifndef __MACA_RANDOM_SAMPLE_H__
#define __MACA_RANDOM_SAMPLE_H__

#include "../../../devices/maca/maca_handle.h"
#include "operators.h"

struct RandomSampleMacaDescriptor {
    Device device;
    int device_id;
    DT dtype;
    int voc;
    DT rDtype;
    int rLength;
};

typedef struct RandomSampleMacaDescriptor *RandomSampleMacaDescriptor_t;

infiniopStatus_t macaCreateRandomSampleDescriptor(MacaHandle_t handle,
                                                  RandomSampleMacaDescriptor_t *desc_ptr, infiniopTensorDescriptor_t result,
                                                  infiniopTensorDescriptor_t probs);

infiniopStatus_t macaGetRandomSampleWorkspaceSize(RandomSampleMacaDescriptor_t desc, uint64_t *size);

infiniopStatus_t macaRandomSample(RandomSampleMacaDescriptor_t desc,
                                  void *workspace,
                                  uint64_t workspace_size,
                                  void *result,
                                  void const *probs,
                                  float random_val,
                                  float topp,
                                  int topk,
                                  float temperature,
                                  void *stream);

infiniopStatus_t macaDestroyRandomSampleDescriptor(RandomSampleMacaDescriptor_t desc);


#endif
