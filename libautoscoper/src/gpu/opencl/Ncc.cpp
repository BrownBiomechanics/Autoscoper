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

using namespace std;

namespace xromm { namespace gpu {

//////// Global variables ////////

static unsigned g_max_n = 0;
static size_t g_maxNumThreads = 0;

static Buffer* d_sums = NULL;
static Buffer* d_nums = NULL;
static Buffer* d_den1s = NULL;
static Buffer* d_den2s = NULL;

//////// Cuda kernels ////////
#include "gpu/opencl/kernel/Ncc.cl.h"
#include "gpu/opencl/kernel/NccSum.cl.h"

static Program ncc_kernel_;
static Program ncc_sum_kernel_;

//////// Helper functions ////////

static void get_max_threads()
{
	if (!g_maxNumThreads)
	{
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
		if (g_maxNumThreads*sizeof(float) > maxLocalMem) {
			g_maxNumThreads = maxLocalMem / sizeof(float);
		}

#if DEBUG
		cerr << "ncc: maxLocalMem = " << maxLocalMem << endl;
		cerr << "ncc: maxNumThreads = " << g_maxNumThreads << endl;
#endif
	}
}

static void get_device_params(unsigned n,
					   size_t& numThreads,
					   size_t& numBlocks,
					   size_t& sizeMem)
{
	numThreads = n < g_maxNumThreads ? n : g_maxNumThreads;
	numBlocks = (n+numThreads-1)/numThreads;
	sizeMem = numThreads*sizeof(float);
}

static float ncc_sum(Buffer* f, unsigned n)
{
	size_t numThreads, numBlocks, sizeMem;
	get_device_params(n, numThreads, numBlocks, sizeMem);

	Kernel* kernel = ncc_sum_kernel_.compile(NccSum_cl, "ncc_sum_kernel");

	while (n > 1)
	{
#if DEBUG
		cerr << "ncc_sum[" << n << "] numThreads = " << numThreads << endl;
		cerr << "ncc_sum[" << n << "] numBlocks = " << numBlocks << endl;
		cerr << "ncc_sum[" << n << "] sizeMem = " << sizeMem << endl;
#endif

		kernel->block2d(numThreads, 1);
		kernel->grid2d(1, numBlocks);

		kernel->addBufferArg(f);
		kernel->addBufferArg(d_sums);
		kernel->addLocalMem(sizeMem);
		kernel->addArg(n);

		kernel->launch();

#if DEBUG
		float *tmp = new float[numBlocks];
		d_sums->write(tmp, numBlocks*sizeof(float));
		for (unsigned j=0; j<numBlocks; j++) {
			cerr << " " << tmp[j];
		}
		cerr << endl;
		delete tmp;
#endif

		n = numBlocks;
		get_device_params(n, numThreads, numBlocks, sizeMem);
		f = d_sums;

		kernel->reset();
	}

	delete kernel;

	float h_sum;
	d_sums->write(&h_sum, sizeof(float));
	return h_sum;
}

//////// Interface Definitions ////////

void ncc_init(unsigned max_n)
{
	if (g_max_n != max_n)
	{
		ncc_deinit();
		get_max_threads();

		size_t numThreads, numBlocks, sizeMem;
		get_device_params(max_n, numThreads, numBlocks, sizeMem);

		d_sums = new Buffer(numBlocks*sizeof(float));
		d_nums = new Buffer(max_n*sizeof(float));
		d_den1s = new Buffer(max_n*sizeof(float));
		d_den2s = new Buffer(max_n*sizeof(float));

		g_max_n = max_n;
	}
}

void ncc_deinit()
{
	delete d_sums;
	delete d_nums;
	delete d_den1s;
	delete d_den2s;

	g_max_n = 0;
}

float ncc(Buffer* f, Buffer* g, Buffer* mask, unsigned n)
{
	float nbPixel = ncc_sum(mask, n);
	float meanF = ncc_sum(f, n) / nbPixel;
	float meanG = ncc_sum(g, n) / nbPixel;

#if DEBUG
	cerr << "meanF: " << meanF << endl;
	cerr << "meanG: " << meanG << endl;
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
	
	float den = sqrt(ncc_sum(d_den1s,n)*ncc_sum(d_den2s,n));

	if (den < 1e-5) {
		return 1e5;
	}
	
	return ncc_sum(d_nums,n)/den;
}

} // namespace gpu

} // namespace xromm

