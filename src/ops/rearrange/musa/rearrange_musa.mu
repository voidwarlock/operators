#include "../../../devices/musa/common_musa.h"
#include "rearrange_musa.h"

template<class Tmem>
static __global__ void rearrange(
    void *__restrict__ dst,
    int const rsa,
    int const csa,
    void const *__restrict__ src,
    int const rsb,
    int const csb,
    unsigned int const ncols) {

    auto row = blockIdx.y,
         col = blockIdx.x * blockDim.y + threadIdx.y;
    if (col >= ncols) return;

    auto thread = threadIdx.x,
         warp_size = blockDim.x;
    auto i = (row * rsa + col * csa) * warp_size + thread;
    auto j = (row * rsb + col * csb) * warp_size + thread;

    reinterpret_cast<Tmem *>(dst)[i] = reinterpret_cast<Tmem const *>(src)[j];
}


void rearrange_mt_gpu(RearrangeMusaDescriptor_t desc, void *y, void const *x, void *stream) {
    auto musa_stream = reinterpret_cast<musaStream_t>(stream);
    auto unit = desc->unit,
         r = desc->r, c = desc->c;
    auto dst_rs = desc->dst_rs, dst_cs = desc->dst_cs,
         src_rs = desc->src_rs, src_cs = desc->src_cs;

    if (r == 1 && c == 1) {
        musaMemcpyAsync(y, x, unit, musaMemcpyDeviceToDevice, musa_stream);
        return;
    }

    auto warps = 1024 / WARP_SIZE;
    auto grid = dim3((c + warps - 1) / warps, r);
    auto block = dim3(WARP_SIZE, (c + grid.x - 1) / grid.x);
    dst_rs /= unit;
    dst_cs /= unit;
    src_rs /= unit;
    src_cs /= unit;

    switch (unit / WARP_SIZE) {
        case 1:
            rearrange<uchar1><<<grid, block, 0, musa_stream>>>(y, dst_rs, dst_cs, x, src_rs, src_cs, c);
            break;
        case 2:
            rearrange<uchar2><<<grid, block, 0, musa_stream>>>(y, dst_rs, dst_cs, x, src_rs, src_cs, c);
            break;
        case 4:
            rearrange<float1><<<grid, block, 0, musa_stream>>>(y, dst_rs, dst_cs, x, src_rs, src_cs, c);
            break;
        case 8:
            rearrange<float2><<<grid, block, 0, musa_stream>>>(y, dst_rs, dst_cs, x, src_rs, src_cs, c);
            break;
        case 16:
            rearrange<float4><<<grid, block, 0, musa_stream>>>(y, dst_rs, dst_cs, x, src_rs, src_cs, c);
            break;
        case 32:
            rearrange<double4><<<grid, block, 0, musa_stream>>>(y, dst_rs, dst_cs, x, src_rs, src_cs, c);
            break;
        default:
            break;
    }
}
infiniopStatus_t musaRearrange(RearrangeMusaDescriptor_t desc,
                               void *dst, void const *src, void *stream) {
    int current_device;
    if (musaGetDevice(&current_device) != musaSuccess) {
        return STATUS_BAD_DEVICE; 
    }
    if (current_device != desc->device_id && musaSetDevice(desc->device_id) != musaSuccess) {
        return STATUS_BAD_DEVICE;
    }   
    rearrange_mt_gpu(desc, dst, src, stream);
    return STATUS_SUCCESS;
}
