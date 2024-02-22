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

/// \file  RayCaster_kernels.cu
/// \author Andy Loomis, Benjamin Knorlein

#include "RayCaster_kernels.h"

#include <cutil_inline.h>
#include <cutil_math.h>

struct Ray
{
  float3 origin;
  float3 direction;
};

struct float3x4
{
  float4 m[3];
};

// Forward declarations

__global__ void cuda_volume_render_kernel(cudaTextureObject_t tex,
                                          float* output,
                                          size_t width,
                                          size_t height,
                                          float step,
                                          float intensity,
                                          float cutoff);

// Global variables

static __constant__ float4 d_viewport;
static __constant__ float3x4 d_invModelView;

namespace xromm {
namespace gpu {
void volume_viewport(float x, float y, float width, float height)
{
  float4 viewport = make_float4(x, y, width, height);
  cutilSafeCall(cudaMemcpyToSymbol(d_viewport, &viewport, sizeof(float4)));
}

void volume_render(cudaTextureObject_t tex,
                   float* buffer,
                   size_t width,
                   size_t height,
                   const float* invModelView,
                   float step,
                   float intensity,
                   float cutoff)
{
  // Copy the matrix to the device.
  cutilSafeCall(cudaMemcpyToSymbol(d_invModelView, invModelView, sizeof(float3x4)));

  // Calculate the block and grid sizes.
  dim3 blockDim(32, 32);
  dim3 gridDim(((unsigned int)width + blockDim.x - 1) / blockDim.x,
               ((unsigned int)height + blockDim.y - 1) / blockDim.y);

  // Call the kernel
  cuda_volume_render_kernel<<<gridDim, blockDim>>>(tex, buffer, width, height, step, intensity, cutoff);

  // This crashes it under windows
  // cutilSafeCall(cudaThreadSynchronize());
  // cutilSafeCall(cudaGetLastError());
}
} // namespace gpu
} // namespace xromm

// Device and Kernel functions

// Intersect a ray with an axis aligned box.
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
// https://education.siggraph.org/static/HyperGraph/raytrace/rtrace1.htm
__device__ int box_intersect(Ray ray, float3 boxMin, float3 boxMax, float* _near, float* _far)
{
  // Compute intersection of ray with all six planes.
  float3 invDirection = make_float3(1.0f) / ray.direction;
  float3 tBot = invDirection * (boxMin - ray.origin);
  float3 tTop = invDirection * (boxMax - ray.origin);

  // Re-order intersections to find smallest and largest on each axis.
  float3 tMin = fminf(tTop, tBot);
  float3 tMax = fmaxf(tTop, tBot);

  // Find the largest tMin and the smallest tMax.
  *_near = fmaxf(fmaxf(tMin.x, tMin.y), tMin.z);
  *_far = fminf(fminf(tMax.x, tMax.y), tMax.z);

  return *_far > *_near;
}

// Transform vector by matrix (no translation).
__device__ float3 mul(const float3x4& M, const float3& v)
{
  return make_float3(dot(v, make_float3(M.m[0])), dot(v, make_float3(M.m[1])), dot(v, make_float3(M.m[2])));
}

// Transform vector by matrix with translation.
__device__ float4 mul(const float3x4& M, const float4& v)
{
  return make_float4(dot(v, M.m[0]), dot(v, M.m[1]), dot(v, M.m[2]), 1.0f);
}

// Render the volume using ray marching.
__global__ void cuda_volume_render_kernel(cudaTextureObject_t tex,
                                          float* buffer,
                                          size_t width,
                                          size_t height,
                                          float step,
                                          float intensity,
                                          float cutoff)
{
  uint x = blockIdx.x * blockDim.x + threadIdx.x;
  uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x > width - 1 || y > height - 1) {
    return;
  }

  // Calculate the normalized device coordinates using the viewport
  float u = d_viewport.x + d_viewport.z * (x / (float)width);
  float v = d_viewport.y + d_viewport.w * (y / (float)height);

  // Determine the look ray in camera space.
  float4 eye = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
  float3 look = step * normalize(make_float3(u, v, -2.0f));

  // Calculate the ray in world space.
  Ray ray = { make_float3(mul(d_invModelView, eye)), mul(d_invModelView, look) };

  // Find intersection with box.
  float3 boxMin = make_float3(0.0f, 0.0f, -1.0f);
  float3 boxMax = make_float3(1.0f, 1.0f, 0.0f);
  float _near;
  float _far;
  if (!box_intersect(ray, boxMin, boxMax, &_near, &_far)) {
    buffer[y * width + x] = 0.0f;
    return;
  }

  // Clamp to near plane.
  if (_near < 0.0f)
    _near = 0.0f;

  // Perform the ray marching from back to front.
  float t = _far;
  float density = 0.0f;
  while (t > _near) {
    float3 point = ray.origin + t * ray.direction;
    float sample = tex3D<float>(tex, point.x, 1.0f - point.y, -point.z);
    density += sample > cutoff ? step * sample : 0.0f;
    t -= 1.0f;
  }

  buffer[y * width + x] = clamp(density / intensity, 0.0f, 1.0f);
}
