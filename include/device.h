#ifndef __DEVICE_H__
#define __DEVICE_H__

enum DeviceEnum {
    DevCpu = 0,
    DevNvGpu = 1,
    DevCambriconMlu = 2,
    DevAscendNpu = 3,
    DevMetaxGpu = 4,
    DevMthreadsGpu = 5,
};

typedef enum DeviceEnum Device;

#endif// __DEVICE_H__
