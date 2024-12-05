#include "musa_handle.h"
#include <iostream>

infiniopStatus_t createMusaHandle(MusaHandle_t* handle_ptr, int device_id) {
    int device_count;
    musaGetDeviceCount(&device_count);
    if (device_id >= device_count) {
        return STATUS_BAD_DEVICE;
    }

    int current_device;
    if (musaGetDevice(&current_device) != musaSuccess) {
        return STATUS_BAD_DEVICE; 
    }
    if (current_device != device_id && musaSetDevice(device_id) != musaSuccess) {
        return STATUS_BAD_DEVICE;
    }

    // set CUDA device property
    musaDeviceProp prop;
    musaGetDeviceProperties(&prop, device_id);


    auto mublas_pool = std::make_shared<Pool<mublasHandle_t>>();
    mublasHandle_t *mublas_handle = new mublasHandle_t;
    mublasCreate(mublas_handle);
    mublas_pool->push(mublas_handle);

    *handle_ptr = new MusaContext{DevMtGpu, device_id, std::move(mublas_pool), std::move(prop)};

    return STATUS_SUCCESS;
}

infiniopStatus_t deleteMusaHandle(MusaHandle_t handle_ptr) {
    handle_ptr->mublas_handles_t = nullptr;
    delete handle_ptr;

    return STATUS_SUCCESS;
}