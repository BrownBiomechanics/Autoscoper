/// \file Hdist.hpp
/// \author Anthony J. Lombardi

#ifndef XROMM_HDIST_HPP
#define XROMM_HDIST_HPP
#include "OpenCL.hpp"
namespace xromm {
	namespace gpu {
		void hdist_init(unsigned max_n);

		void hdist_deinit();

		float hdist(Buffer* f, Buffer* g, Buffer* mask, unsigned n);

	} // namespace gpu
} // namespace xromm
#endif // XROMM_HDIST_HPP