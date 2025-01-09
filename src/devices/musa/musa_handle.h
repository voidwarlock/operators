#ifndef __MUSA_HANDLE_H__
#define __MUSA_HANDLE_H__

#include "pool.h"
#include "device.h"
#include "status.h"
#include "ops/matmul/matmul.h"
#include <memory>
#include <musa.h>
#include <musa_runtime_api.h>
#include <mudnn.h>
#include <mublas.h>

struct MusaContext {
    Device device;
    int device_id;
    std::shared_ptr<Pool<mublasHandle_t>> mublas_handles_t;
    std::shared_ptr<Pool<musa::dnn::Handle>> mudnn_handles_t;
    musaDeviceProp prop;
    int compute_capability_major;
    int compute_capability_minor;
};
typedef struct MusaContext *MusaHandle_t;

infiniopStatus_t createMusaHandle(MusaHandle_t *handle_ptr, int device_id);

infiniopStatus_t deleteMusaHandle(MusaHandle_t handle_ptr);

template<typename T>
void use_mublas(std::shared_ptr<Pool<mublasHandle_t>> mublas_handles_t, int device_id, MUstream stream, T const &f) {
    mublasHandle_t *handle = mublas_handles_t->pop();
    if (!handle) {
        int current_device;
        musaGetDevice(&current_device);
        if (current_device != device_id) {
            musaSetDevice(device_id);
        }
        mublasHandle_t *handle = new mublasHandle_t;
        mublasCreate(handle);
    }
    mublasSetStream(*handle, (MUstream) stream);
    f(*handle);
    mublas_handles_t->push(handle);
}

template<typename T>
void use_mudnn(std::shared_ptr<Pool<musa::dnn::Handle>> mudnn_handles_t, int device_id, musaStream_t stream, T const &f) {
    musa::dnn::Handle* handle = mudnn_handles_t->pop();
    if (!handle) {
        int current_device;
        musaGetDevice(&current_device);
        if (current_device != device_id) {
            musaSetDevice(device_id);
        }
        handle = new musa::dnn::Handle(device_id);
        // mudnnCreate(handle);
    }
    // mudnnSetStream(*handle, (MUstream) stream);
    handle->SetStream(stream);
    f(handle);
    mudnn_handles_t->push(handle);
}

#endif // __MUSA_HANDLE_H__