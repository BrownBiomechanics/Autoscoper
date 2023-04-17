/// \file DistanceField.hpp
/// \author Anthony J. Lombardi
/// 
/// Code based off of code by Chris Saliba from the "dsx_planner" repository on Github
/// https://github.com/cmsaliba/dsx_planner
#pragma once
#ifndef XROMM_DISTANCEFIELD_H
#define XROMM_DISTANCEFIELD_H

#include "OpenCL.hpp"
#include "Mesh.hpp"

namespace xromm {
	namespace gpu {
		class DistanceField {
		public:
			DistanceField(Mesh* mesh,float desiredVoxelSize = 0.001f,float scale = 0.1f); // default the voxel size to 1mm and the scale to 0.1f
			//~DistanceField();
			Buffer* getDistanceFieldBuffer() { return dFieldBuffer_; };
			float* getVoxelSize() {return voxelSize_;};
			float* getOffset() {return offset_;};
			int* getNumPoints() {return nPts_;};
			float* getDistanceFieldSize() {return dFieldSize_;};
			size_t getDistanceFieldBufferSize() {return dFieldBufferSize_;};
		private:
			size_t dFieldBufferSize_;
			int nPts_[3];
			float dFieldSize_[3];
			float voxelSize_[3];
			float offset_[3];

			Buffer* dFieldBuffer_;
			
			void calculate_unsigned_dfield(Mesh* mesh);
			void sign_dfield(Mesh* mesh);

		};

} } // namespace xromm::gpu
#endif // !XROMM_DISTANCEFIELD_H
