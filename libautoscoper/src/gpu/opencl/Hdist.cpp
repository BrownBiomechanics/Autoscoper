/// \file Hdist.hpp
/// \author Anthony J. Lombardi

#include "Hdist.hpp"
#include <cmath>
#include <iostream>



namespace xromm {
	namespace gpu {

        using namespace std;

        static unsigned g_max_n_hdist = 0;

        static size_t g_maxNumThreads_hdist = 0;

        static Buffer* d_sums_ba = NULL;
        static Buffer* d_nums_ba = NULL;
        static Buffer* d_den1s_ba = NULL;
        static Buffer* d_den2s_ba = NULL;

        //////// OpenCL Kernels ////////
        #include "gpu/opencl/kernel/Hdist.cl.h"
        #include "gpu/opencl/kernel/HdistSum.cl.h"

        static Program hdist_kernel_;
        static Program hdist_sum_kernel_;

		//////// Helper functions - from NCC files ////////

        static void get_max_threads()
        {
            if (!g_maxNumThreads_hdist)
            {
                g_maxNumThreads_hdist = Kernel::getMaxGroup();
                size_t* max_items = Kernel::getMaxItems();
                if (max_items[0] < g_maxNumThreads_hdist)
                    g_maxNumThreads_hdist = max_items[0];
                delete max_items;

                // HACK: automatic detection above is not working on
                // Granoff iMac 10.7 (reports 1024, but throws
                // CL_INVALID_WORK_GROUP_SIZE). Hard set to 128 for now.
                g_maxNumThreads_hdist = 128;

                /* reduce threads to fit in local mem */
                size_t maxLocalMem = Kernel::getLocalMemSize();
                if (g_maxNumThreads_hdist * sizeof(float) > maxLocalMem) {
                    g_maxNumThreads_hdist = maxLocalMem / sizeof(float);
                }

            #if DEBUG
                cerr << "ncc: maxLocalMem = " << maxLocalMem << endl;
                cerr << "ncc: maxNumThreads = " << g_maxNumThreads_hdist << endl;
            #endif
            }
        }

        static void get_device_params(unsigned n,
            size_t& numThreads,
            size_t& numBlocks,
            size_t& sizeMem)
        {
            numThreads = n < g_maxNumThreads_hdist ? n : g_maxNumThreads_hdist;
            numBlocks = (n + numThreads - 1) / numThreads;
            sizeMem = numThreads * sizeof(float);
        }

        //////// Kernel Drivers ////////
        static float hdist_sum(Buffer * f, unsigned n) {
            size_t numThreads, numBlocks, sizeMem;
            get_device_params(n, numThreads, numBlocks, sizeMem);

            Kernel* kernel = hdist_sum_kernel_.compile(HdistSum_cl, "hdist_sum_kernel");
            while (n > 1) {
                kernel->block2d(numThreads, 1);
                kernel->grid2d(1, numBlocks);

                kernel->addBufferArg(f);
                kernel->addBufferArg(d_sums_ba);
                kernel->addLocalMem(sizeMem);
                kernel->addArg(n);

                kernel->launch();

                n = numBlocks;
                get_device_params(n, numThreads, numBlocks, sizeMem);
                f = d_sums_ba;

                kernel->reset();

            }

            delete kernel;

            float sum;
            d_sums_ba->write(&sum, sizeof(float));
            return sum;
        }

		//////// Interface Definitions ////////

		void hdist_init(unsigned max_n) {
			if (g_max_n_hdist != max_n) {
				hdist_deinit();
                get_max_threads();

				size_t numThreads, numBlocks, sizeMem;
                get_device_params(max_n, numThreads, numBlocks, sizeMem);

                d_sums_ba = new Buffer(numBlocks * sizeof(float));
                d_nums_ba = new Buffer(max_n * sizeof(float));
                d_den1s_ba = new Buffer(max_n * sizeof(float));
                d_den2s_ba = new Buffer(max_n * sizeof(float));

                g_max_n_hdist = max_n;
			}
		}

        void hdist_deinit() {
            delete d_sums_ba;
            delete d_nums_ba;
            delete d_den1s_ba;
            delete d_den2s_ba;

            g_max_n_hdist = 0;
        }

        float hdist(Buffer* f, Buffer* g, Buffer* mask, unsigned n) {
            float nbPixel = hdist_sum(mask, n);
            float meanF = hdist_sum(f, n) / nbPixel;
            float meanG = hdist_sum(g, n) / nbPixel;

            size_t numThreads, numBlocks, sizeMem;
            get_device_params(n, numThreads, numBlocks, sizeMem);

            Kernel* kernel = hdist_kernel_.compile(Hdist_cl, "hdist_kernel");

            kernel->block1d(numThreads);
            kernel->grid1d(numBlocks);

            kernel->addBufferArg(f);
            kernel->addArg(meanF);
            kernel->addBufferArg(g);
            kernel->addArg(meanG);
            kernel->addBufferArg(mask);
            kernel->addBufferArg(d_nums_ba);
            kernel->addArg(n);

            kernel->launch();

            delete kernel;

            float sad_const = hdist_sum(d_nums_ba, n);
            return sad_const;
        }

} } // namespace xromm::gpu