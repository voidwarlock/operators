#ifndef __MUSA_RANDOM_SAMPLE_H__
#define __MUSA_RANDOM_SAMPLE_H__

#include "../../../devices/musa/musa_handle.h"
#include "operators.h"

struct RandomSampleMusaDescriptor {
    Device device;
    int device_id;
    DT dtype;
    int voc;
    DT rDtype;
    int rLength;
};

typedef struct RandomSampleMusaDescriptor *RandomSampleMusaDescriptor_t;

infiniopStatus_t musaCreateRandomSampleDescriptor(MusaHandle_t handle,
                                                  RandomSampleMusaDescriptor_t *desc_ptr, infiniopTensorDescriptor_t result,
                                                  infiniopTensorDescriptor_t probs);

infiniopStatus_t musaGetRandomSampleWorkspaceSize(RandomSampleMusaDescriptor_t desc, unsigned long int *size);

infiniopStatus_t musaRandomSample(RandomSampleMusaDescriptor_t desc,
                                  void *workspace,
                                  uint64_t workspace_size,
                                  void *result,
                                  void const *probs,
                                  float random_val,
                                  float topp,
                                  int topk,
                                  float temperature,
                                  void *stream);

infiniopStatus_t musaDestroyRandomSampleDescriptor(RandomSampleMusaDescriptor_t desc);


#endif
