#pragma once
#include <cuda.h>
#include <cutil_inline.h>
#include <cutil_inline_runtime.h>

inline cudaTextureObject_t createTexureObjectFromArray(cudaArray* arr, cudaTextureReadMode readMode) {
    // Approach implemented below is based off of
    // https://developer.nvidia.com/blog/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.normalizedCoords = true;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.readMode = readMode;

    cudaTextureObject_t tex = 0;
    cutilSafeCall(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));
    return tex;
}
