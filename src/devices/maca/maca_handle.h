#ifndef MACA_HANDLE_H
#define MACA_HANDLE_H

#include "../pool.h"
#include "common_maca.h"
#include "device.h"
#include "status.h"
#include <hcblas/hcblas.h>
#include <hcdnn/hcdnn.h>
#include <memory>

struct MacaContext {
    Device device;
    int device_id;
    std::shared_ptr<Pool<hcblasHandle_t>> mcblas_handles_t;
    std::shared_ptr<Pool<hcdnnHandle_t>> mcdnn_handles_t;
    hcDeviceProp_t prop;
    int compute_capability_major;
    int compute_capability_minor;
};
typedef struct MacaContext *MacaHandle_t;

infiniopStatus_t createMacaHandle(MacaHandle_t *handle_ptr, int device_id);

infiniopStatus_t deleteMacaHandle(MacaHandle_t handle_ptr);

template<typename T>
void use_mcblas(std::shared_ptr<Pool<hcblasHandle_t>> mcblas_handles_t, int device_id, hcStream_t stream, T const &f) {
    auto handle = mcblas_handles_t->pop();
    if (!handle) {
        hcSetDevice(device_id);
        hcblasCreate(&(*handle));
    }
    hcblasSetStream(*handle, (hcStream_t) stream);
    f(*handle);
    mcblas_handles_t->push(std::move(*handle));
}

template<typename T>
hcdnnStatus_t use_mcdnn(std::shared_ptr<Pool<hcdnnHandle_t>> mcdnn_handles_t, int device_id, hcStream_t stream, T const &f) {
    auto handle = mcdnn_handles_t->pop();
    if (!handle) {
        hcSetDevice(device_id);
        hcdnnCreate(&(*handle));
    }
    hcdnnSetStream(*handle, stream);
    hcdnnStatus_t status = f(*handle);
    mcdnn_handles_t->push(std::move(*handle));
    return status;
}

#endif
