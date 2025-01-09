
#include "tensor_desc.h"
#include <iostream>
#include <vector>

// void mudnnSqueezeTensorDim(mudnnTensorDesc_t &ldesc, mudnnTensorDesc_t &rdesc, mudnnTensorDesc_t &outdesc) {
//     if (outdesc->ndims > 2) {
//         if (ldesc->ndims > 2 && *ldesc->dim == 1) {
//             ldesc->ndims -= 1;
//             ldesc->dim = ldesc->dim+1;
//         }
//         if (rdesc->ndims > 2 && *rdesc->dim == 1) {
//             rdesc->ndims -= 1;
//             rdesc->dim = rdesc->dim+1;
//         }
//     }
// }

// void mudnnCreateTensorDescriptor(mudnnTensorDesc_t *desc) {
//     *desc = new mudnnTensorDesc;
//     (*desc)->type = Type::FLOAT;
//     (*desc)->format = Format::UNKNOWN;
//     (*desc)->ndims = 0;
//     (*desc)->dim = nullptr;
//     (*desc)->stride = nullptr;
//     (*desc)->scales = nullptr;
//     (*desc)->addr = nullptr;
// }


// void mudnnSetTensorDescriptor(mudnnTensorDesc_t &desc, int64_t *shape, int64_t *stride, int64_t ndim,
//                               int64_t offset, Type type, Format format) {
//     desc->type = type;
//     desc->format = format;
//     desc->ndims = ndim;
//     desc->dim = shape;
//     if (stride) {
//         desc->stride = stride;
//     } else {
//         std::vector<int64_t> stride_v(ndim, 1);
//         for (int64_t i = ndim - 2; i >= 0; i--) {
//             stride_v[i] = shape[i + 1] * stride_v[i + 1];
//         }
//         desc->stride = stride_v.data();
//     }
// }

// void mudnnSetTensorDescriptorFromTensorLayout(mudnnTensorDesc_t &desc, const TensorLayout *layout) {
//     auto dims = new int64_t(layout->ndim);
//     for (uint64_t i = 0; i < layout->ndim; i++) {
//         dims[i] = static_cast<int64_t>(layout->shape[i]);
//     }
//     // Cast bytes stride to element stride
//     auto strides = new int64_t(layout->ndim);
//     for (uint64_t i = 0; i < layout->ndim; i++) {
//         strides[i] = layout->strides[i] / (layout->dt).size;
//     }

//     Type type = Type::HALF;
//     Format format = Format::NCHW;

//     mudnnSetTensorDescriptor(desc, dims, strides, layout->ndim, 0, type, format);
// }

// void mudnnDestroyTensorDescriptor(mudnnTensorDesc_t &desc) {
//     if (desc) {
//         delete desc;
//         desc = nullptr;
//     }
// }

// int mudnnCreateTensor(TensorDescriptor desc, void *data, musa::dnn::Tensor **tensor) {
//     *tensor = new musa::dnn::Tensor();
    
//     (*tensor)->SetAddr(data);
//     // (*tensor)->SetType(musa::dnn::Tensor::Type(desc->type));
//     (*tensor)->SetFormat(musa::dnn::Tensor::Format(desc->format));
//     // (*tensor)->SetNdInfo(desc->ndims, desc->dim, desc->stride);
//     (*tensor)->SetNdInfo(desc->ndims, desc->dim);
//     return 0;
// }