#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceId = 0; // 设备ID，如果有多个GPU可以更改为其他ID
    cudaSetDevice(deviceId);

    int sharedMemPerBlock;
    cudaDeviceGetAttribute(&sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, deviceId);

    printf("Device %d:\n", deviceId);
    printf("Max shared memory per block: %d bytes\n", sharedMemPerBlock);

    // 计算可以存储的float数量
    size_t floatSize = sizeof(float);
    size_t numFloats = sharedMemPerBlock / floatSize;

    printf("Number of floats that can be stored in shared memory: %zu\n", numFloats);

    return 0;
}
