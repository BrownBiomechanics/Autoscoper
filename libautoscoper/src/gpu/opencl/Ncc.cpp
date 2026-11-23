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

/// \file Ncc.cpp
/// \author Andy Loomis, Mark Howison

#include "Ncc.hpp"

#include <iostream>
#include <cmath>
#include <algorithm>

namespace xromm {
namespace gpu {
//////// Global variables ////////

static unsigned g_max_n = 0;
static size_t g_maxNumThreads = 0;

// Partial sums alternate between these two buffers from one reduction pass
// to the next.
static Buffer* d_sums_a = NULL;
static Buffer* d_sums_b = NULL;
static Buffer* d_nums = NULL;
static Buffer* d_den1s = NULL;
static Buffer* d_den2s = NULL;

//////// Cuda kernels ////////
#include "gpu/opencl/kernel/Ncc.cl.h"
#include "gpu/opencl/kernel/NccSum.cl.h"

static Program ncc_kernel_;
static Program ncc_sum_kernel_;

//////// Helper functions ////////

// Largest power of two that is less than or equal to x (x >= 1).
static size_t floor_pow2(size_t x)
{
  size_t p = 1;
  while (p * 2 <= x) {
    p *= 2;
  }
  return p;
}

// Smallest power of two that is greater than or equal to x.
static size_t ceil_pow2(size_t x)
{
  size_t p = 1;
  while (p < x) {
    p *= 2;
  }
  return p;
}

static void get_max_threads()
{
  if (!g_maxNumThreads) {
    g_maxNumThreads = Kernel::getMaxGroup();
    size_t* max_items = Kernel::getMaxItems();
    if (max_items[0] < g_maxNumThreads)
      g_maxNumThreads = max_items[0];
    delete max_items;

    // HACK: automatic detection above is not working on
    // Granoff iMac 10.7 (reports 1024, but throws
    // CL_INVALID_WORK_GROUP_SIZE). Hard set to 128 for now.
    g_maxNumThreads = 128;

    /* reduce threads to fit in local mem */
    size_t maxLocalMem = Kernel::getLocalMemSize();
    if (g_maxNumThreads * sizeof(float) > maxLocalMem) {
      g_maxNumThreads = maxLocalMem / sizeof(float);
    }

    // The reduction kernel halves the work-group size at each step, so the
    // size must be a power of two.
    g_maxNumThreads = floor_pow2(g_maxNumThreads);

#if DEBUG
    std::cerr << "ncc: maxLocalMem = " << maxLocalMem << std::endl;
    std::cerr << "ncc: maxNumThreads = " << g_maxNumThreads << std::endl;
#endif
  }
}

static void get_device_params(unsigned n, size_t& numThreads, size_t& numBlocks, size_t& sizeMem)
{
  numThreads = std::min(g_maxNumThreads, ceil_pow2(n));
  numBlocks = (n + numThreads - 1) / numThreads;
  sizeMem = numThreads * sizeof(float);
}

// Returns the sum of the first n values of f. Each pass adds up blocks of
// values into a partial-sum buffer; the next pass reads that buffer and writes
// the other one, so no pass reads a buffer that it is also writing.
static float ncc_sum(Buffer* f, unsigned n)
{
  if (n == 0) {
    return 0.0f;
  }

  Kernel* kernel = ncc_sum_kernel_.compile(NccSum_cl, "ncc_sum_kernel");

  Buffer* src = f;
  Buffer* dst = d_sums_a;
  while (n > 1) {
    size_t numThreads, numBlocks, sizeMem;
    get_device_params(n, numThreads, numBlocks, sizeMem);
#if DEBUG
    std::cerr << "ncc_sum[" << n << "] numThreads = " << numThreads << " numBlocks = " << numBlocks << std::endl;
#endif

    kernel->block2d(numThreads, 1);
    kernel->grid2d(1, numBlocks);

    kernel->addBufferArg(src);
    kernel->addBufferArg(dst);
    kernel->addLocalMem(sizeMem);
    kernel->addArg(n);

    kernel->launch();
    kernel->reset();

    n = static_cast<unsigned>(numBlocks);
    src = dst;
    dst = (src == d_sums_a) ? d_sums_b : d_sums_a;
  }

  delete kernel;

  float h_sum;
  src->write(&h_sum, sizeof(float));
  return h_sum;
}

//////// Interface Definitions ////////

void ncc_init(unsigned max_n)
{
  if (g_max_n != max_n) {
    ncc_deinit();
    get_max_threads();

    size_t numThreads, numBlocks, sizeMem;
    get_device_params(max_n, numThreads, numBlocks, sizeMem);

    d_sums_a = new Buffer(numBlocks * sizeof(float));
    d_sums_b = new Buffer(numBlocks * sizeof(float));
    d_nums = new Buffer(max_n * sizeof(float));
    d_den1s = new Buffer(max_n * sizeof(float));
    d_den2s = new Buffer(max_n * sizeof(float));

    g_max_n = max_n;
  }
}

void ncc_deinit()
{
  delete d_sums_a;
  delete d_sums_b;
  delete d_nums;
  delete d_den1s;
  delete d_den2s;
  d_sums_a = NULL;
  d_sums_b = NULL;
  d_nums = NULL;
  d_den1s = NULL;
  d_den2s = NULL;

  g_max_n = 0;
}

float ncc(Buffer* f, Buffer* g, Buffer* mask, unsigned n)
{
  float nbPixel = ncc_sum(mask, n);
  float meanF = ncc_sum(f, n) / nbPixel;
  float meanG = ncc_sum(g, n) / nbPixel;

#if DEBUG
  std::cerr << "meanF: " << meanF << std::endl;
  std::cerr << "meanG: " << meanG << std::endl;
#endif

  size_t numThreads, numBlocks, sizeMem;
  get_device_params(n, numThreads, numBlocks, sizeMem);

  Kernel* kernel = ncc_kernel_.compile(Ncc_cl, "ncc_kernel");

  kernel->block1d(numThreads);
  kernel->grid1d(numBlocks);

  kernel->addBufferArg(f);
  kernel->addArg(meanF);
  kernel->addBufferArg(g);
  kernel->addArg(meanG);
  kernel->addBufferArg(mask);
  kernel->addBufferArg(d_nums);
  kernel->addBufferArg(d_den1s);
  kernel->addBufferArg(d_den2s);
  kernel->addArg(n);

  kernel->launch();

  delete kernel;

  float den = sqrt(ncc_sum(d_den1s, n) * ncc_sum(d_den2s, n));

  if (den < 1e-5) {
    return 1e5;
  }

  return ncc_sum(d_nums, n) / den;
}
} // namespace gpu
} // namespace xromm
