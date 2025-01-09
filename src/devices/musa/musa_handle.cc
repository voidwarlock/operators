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

    // create a mublas handle pool
    auto mublas_pool = std::make_shared<Pool<mublasHandle_t>>();
    mublasHandle_t *mublas_handle = new mublasHandle_t;
    mublasCreate(mublas_handle);
    mublas_pool->push(mublas_handle);

    // create a mudnn handle pool
    auto mudnn_pool = std::make_shared<Pool<musa::dnn::Handle>>();
    musa::dnn::Handle *mudnn_handle = new musa::dnn::Handle;
    mudnn_pool->push(mudnn_handle);

    int capability_major;
    int capability_minor;
    musaDeviceGetAttribute(&capability_major, musaDevAttrComputeCapabilityMajor, device_id);
    musaDeviceGetAttribute(&capability_minor, musaDevAttrComputeCapabilityMinor, device_id);

    *handle_ptr = new MusaContext{
        DevMtGpu,
        device_id,
        std::move(mublas_pool),
        std::move(mudnn_pool),
        std::move(prop),
        capability_major,
        capability_minor,};

    return STATUS_SUCCESS;
}

infiniopStatus_t deleteMusaHandle(MusaHandle_t handle_ptr) {
    handle_ptr->mublas_handles_t = nullptr;
    handle_ptr->mudnn_handles_t = nullptr;
    delete handle_ptr;

    return STATUS_SUCCESS;
}