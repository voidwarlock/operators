#include "maca_handle.h"

infiniopStatus_t createMacaHandle(MacaHandle_t *handle_ptr, int device_id) {
    // Check if device_id is valid
    int device_count;
    hcGetDeviceCount(&device_count);
    if (device_id >= device_count) {
        return STATUS_BAD_DEVICE;
    }

    // Create a new mcblas handle pool
    auto pool = std::make_shared<Pool<hcblasHandle_t>>();
    if (hcSetDevice(device_id) != hcSuccess) {
        return STATUS_BAD_DEVICE;
    }
    hcblasHandle_t handle;
    hcblasCreate(&handle);
    pool->push(std::move(handle));

    // create a mcdnn handle pool
    auto mcdnn_pool = std::make_shared<Pool<hcdnnHandle_t>>();
    hcdnnHandle_t mcdnn_handle;
    checkMcdnnError(hcdnnCreate(&mcdnn_handle));
    mcdnn_pool->push(std::move(mcdnn_handle));

    // set MACA device property
    hcDeviceProp_t prop;
    hcGetDeviceProperties(&prop, device_id);

    // set device compute capability numbers
    int capability_major;
    int capability_minor;
    hcDeviceGetAttribute(&capability_major, hcDeviceAttributeComputeCapabilityMajor, device_id);
    hcDeviceGetAttribute(&capability_minor, hcDeviceAttributeComputeCapabilityMinor, device_id);

    *handle_ptr = new MacaContext{
        DevMetaxGpu,
        device_id,
        std::move(pool),
        std::move(mcdnn_pool),
        std::move(prop),
        capability_major,
        capability_minor,
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t deleteMacaHandle(MacaHandle_t handle_ptr) {
    handle_ptr->mcblas_handles_t = nullptr;
    handle_ptr->mcdnn_handles_t = nullptr;
    delete handle_ptr;

    return STATUS_SUCCESS;
}
