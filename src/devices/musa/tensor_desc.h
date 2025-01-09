#ifndef __TENSOR_DESC_H__
#define __TENSOR_DESC_H__

#include "tensor.h"
#include "common_musa.h"
#include <musa.h>
#include <musa_runtime.h>
#include <mudnn.h>
#include <mudnn_base.h>

// using namespace musa::dnn;

// struct mudnnTensorDesc {
//     Type type;
//     Format format;
//     int64_t ndims;
//     int64_t *dim;
//     int64_t *stride;
//     int64_t *scales;
//     int64_t *addr;
// };

// typedef mudnnTensorDesc *mudnnTensorDesc_t;

// void mudnnCreateTensorDescriptor(mudnnTensorDesc_t *desc);

// void mudnnSetTensorDescriptor(mudnnTensorDesc_t &desc, int64_t *shape,
//                               int64_t *stride, int64_t ndim, int64_t offset,
//                               Type type, Format format);

// void mudnnSetTensorDescriptorFromTensorLayout(mudnnTensorDesc_t &desc, const TensorLayout *layout);

// void mudnnDestroyTensorDescriptor(mudnnTensorDesc_t &desc);

int mudnnCreateTensor(TensorDescriptor desc, void *data, musa::dnn::Tensor **tensor);

// void mudnnSetTensorDescriptorFromTensorLayout(mudnnTensorDesc_t &desc, const TensorLayout *layout);

// void mudnnSqueezeTensorDim(mudnnTensorDesc_t &ldesc, mudnnTensorDesc_t &rdesc, mudnnTensorDesc_t &outdesc);


#endif // __TENSOR_DESC_H__ 