// ----------------------------------
// Copyright (c) 2011, Brown University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// (1) Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// (2) Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// (3) Neither the name of Brown University nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY BROWN UNIVERSITY “AS IS” WITH NO
// WARRANTIES OR REPRESENTATIONS OF ANY KIND WHATSOEVER EITHER EXPRESS OR
// IMPLIED, INCLUDING WITHOUT LIMITATION ANY WARRANTY OF DESIGN OR
// MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, EACH OF WHICH ARE
// SPECIFICALLY DISCLAIMED, NOR ANY WARRANTY OR REPRESENTATIONS THAT THE
// SOFTWARE IS ERROR FREE OR THAT THE SOFTWARE WILL NOT INFRINGE ANY
// PATENT, COPYRIGHT, TRADEMARK, OR OTHER THIRD PARTY PROPRIETARY RIGHTS.
// IN NO EVENT SHALL BROWN UNIVERSITY BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY OR CAUSE OF ACTION, WHETHER IN CONTRACT,
// STRICT LIABILITY, TORT, NEGLIGENCE OR OTHERWISE, ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE. ANY RECIPIENT OR USER OF THIS SOFTWARE ACKNOWLEDGES THE
// FOREGOING, AND ACCEPTS ALL RISKS AND LIABILITIES THAT MAY ARISE FROM
// THEIR USE OF THE SOFTWARE.
// ---------------------------------

/// \file Ncc_kernels.cu
/// \author Andy Loomis

#include "Ncc_kernels.h"

#include <iostream>

using namespace std;

#include <cutil_inline.h>

//////// Global variables ////////

static unsigned int g_max_n = 0;

static unsigned int g_maxNumThreads = 0;

static float* d_sums = NULL;
static float* d_nums = NULL;
static float* d_den1s = NULL;
static float* d_den2s = NULL;

//////// Helper functions ////////

static void get_device_params(unsigned int n, unsigned int maxNumThreads,
                              unsigned int& numThreads,
                              unsigned int& numBlocks,
                              unsigned int& sizeMem);

static float sum(float* f, unsigned int n);

//////// Cuda kernels ////////

__global__
void sum_kernel(float* f, float* sums, unsigned int n);

__global__
void cuda_ncc_kernel(float* f, float meanF, float* g, float meanG, float* mask,
                     float* nums, float* den1s, float* den2s,
                     unsigned int n);


//////// Interface Definitions ////////

namespace xromm
{

namespace gpu
{

void ncc_init(unsigned int max_n, unsigned int maxNumThreads)
{
    if (g_max_n != max_n || g_maxNumThreads != maxNumThreads) {
        ncc_deinit();

        unsigned int numThreads, numBlocks, sizeMem;
        get_device_params(max_n, maxNumThreads, numThreads, numBlocks, sizeMem);

        cutilSafeCall(cudaMalloc(&d_sums, numBlocks*sizeof(float)));
        cutilSafeCall(cudaMalloc(&d_nums, max_n*sizeof(float)));
        cutilSafeCall(cudaMalloc(&d_den1s, max_n*sizeof(float)));
        cutilSafeCall(cudaMalloc(&d_den2s, max_n*sizeof(float)));

        g_max_n = max_n;
        g_maxNumThreads = maxNumThreads;
    }
}

void ncc_deinit()
{
    cutilSafeCall(cudaFree(d_sums));
    cutilSafeCall(cudaFree(d_nums));
    cutilSafeCall(cudaFree(d_den1s));
    cutilSafeCall(cudaFree(d_den2s));

    g_max_n = 0;
    g_maxNumThreads = 0;
}

float ncc(float* f, float* g, float* mask, unsigned int n)
{
	float nbPixel = sum(mask, n);
	float meanF = sum(f, n) / nbPixel;
	float meanG = sum(g, n) / nbPixel;

    unsigned int numThreads, numBlocks, sizeMem;
    get_device_params(n, g_maxNumThreads, numThreads, numBlocks, sizeMem);

	cuda_ncc_kernel << <numBlocks, numThreads, sizeMem >> >(f, meanF, g, meanG, mask,
                                                        d_nums, d_den1s,
                                                        d_den2s, n);

    float den = sqrt(sum(d_den1s,n)*sum(d_den2s,n));

    if (den < 1e-5) {
        return 1e5;
    }

    return sum(d_nums,n)/den;
}

} // namespace gpu

} // namespace xromm

//////// Helper Function Definitions ////////

void get_device_params(unsigned int n,
                       unsigned int maxNumThreads,
                       unsigned int& numThreads,
                       unsigned int& numBlocks,
                       unsigned int& sizeMem)
{
    numThreads = n < maxNumThreads? n: maxNumThreads;
    numBlocks = (n+numThreads-1)/numThreads;
    sizeMem = numThreads*sizeof(float);
}

float sum(float* f, unsigned int n)
{
    unsigned int numThreads, numBlocks, sizeMem;
    get_device_params(n, g_maxNumThreads, numThreads, numBlocks, sizeMem);

    while (n > 1) {
        sum_kernel<<<numBlocks, numThreads, sizeMem>>>(f, d_sums, n);
        n = numBlocks;
        get_device_params(n, g_maxNumThreads, numThreads, numBlocks, sizeMem);
        f = d_sums;
    }

    float h_sum;
    cutilSafeCall(cudaMemcpy(&h_sum,
                             d_sums,
                             sizeof(float),
                             cudaMemcpyDeviceToHost));
    return h_sum;
}

__global__
void sum_kernel(float* f, float* sums, unsigned int n)
{
    extern __shared__ float sdata[];

    unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;

    sdata[threadIdx.x] = (i < n) ? f[i] : 0.0f;

    __syncthreads();
    for(unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        sums[blockIdx.x] = sdata[0];
    }
}

__global__
void cuda_ncc_kernel(float* f, float meanF, float* g, float meanG, float* mask,
                     float* nums, float* den1s, float* den2s,
                     unsigned int n)
{
    unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;

	if (i < n && mask[i] > 0.5f) {
        float fMinusMean = f[i]-meanF;
        float gMinusMean = g[i]-meanG;

        nums[i] = fMinusMean*gMinusMean;
        den1s[i] = fMinusMean*fMinusMean;
        den2s[i] = gMinusMean*gMinusMean;
    }
    else {
        nums[i] = 0.0f;
        den1s[i] = 0.0f;
        den2s[i] = 0.0f;
    }
}

