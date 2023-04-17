/// \file DistanceField.cpp
/// \author Anthony J. Lombardi

#include "DistanceField.hpp"
#include <cmath>
#include <vector>
#include <iostream>

namespace xromm {
	namespace gpu {

#define KERNEL_X 10
#define KERNEL_Y 10
#define KERNEL_Z 10
#define KERNEL_CODE DistanceField_cl
#define KERNEL_UNSIGNED "calculate_unsigned_distance_kernel"
#define KERNEL_SIGNED "sign_distance_kernel"

static Program distance_program_;

#include "gpu/opencl/kernel/DistanceField.cl.h"

		DistanceField::DistanceField(Mesh* mesh, float desiredVoxelSize,float scale) {
			float* aabb = mesh->GetAABB();
			float bbox[3] = {aabb[1]-aabb[0],aabb[3]-aabb[2],aabb[5]-aabb[4]}; // size of the aabb

			float start[3] = {aabb[0] - bbox[0] * scale, aabb[2] - bbox[1] * scale, aabb[4] - bbox[2] * scale}; // start of the dfield bbox
			float end[3] = {aabb[1] + bbox[0] * scale, aabb[3] + bbox[1] * scale, aabb[5] + bbox[2] * scale}; // end of the dfield bbox

			offset_[0] = start[0]; offset_[1] = start[1]; offset_[2] = start[2]; // offset of the dfield bbox
			dFieldSize_[0] = end[0] - start[0]; dFieldSize_[1] = end[1] - start[1]; dFieldSize_[2] = end[2] - start[2]; // size of the dfield bbox

			// num of points in each dimension
			nPts_[0] = std::ceil(dFieldSize_[0] / desiredVoxelSize) + 1;
			nPts_[1] = std::ceil(dFieldSize_[1] / desiredVoxelSize) + 1;
			nPts_[2] = std::ceil(dFieldSize_[2] / desiredVoxelSize) + 1;
			
			// actual voxel size
			voxelSize_[0] = dFieldSize_[0] / (float)nPts_[0];
			voxelSize_[1] = dFieldSize_[1] / (float)nPts_[1];
			voxelSize_[2] = dFieldSize_[2] / (float)nPts_[2]; 

			dFieldBufferSize_ = nPts_[0] * nPts_[1] * nPts_[2] * sizeof(float); // size of the dfield buffer

			/*if (dFieldBuffer_ != NULL) {
				delete dFieldBuffer_;
			}*/
			dFieldBuffer_ = new Buffer(dFieldBufferSize_, CL_MEM_READ_WRITE); // create the dfield buffer

			calculate_unsigned_dfield(mesh); // calculate the unsigned distance field
			sign_dfield(mesh); // sign the distance field
		}

		/*DistanceField::~DistanceField() {
			if (dFieldBuffer_ != NULL) {
				delete dFieldBuffer_;
			}
		}*/

		void DistanceField::calculate_unsigned_dfield(Mesh* mesh) {
			int step = 1000;
			unsigned int nItr = (mesh->GetNumFacets() + mesh->GetNumFacets()*3) / step + 1; // number of iterations to run the dfield calculation (number of facets + number of vertices / 1000 + 1)
			// create a buffer of every vertex in the mesh
			std::vector<float> vertexBufferData;
			vertexBufferData.reserve(mesh->GetNumFacets() * 9);
			for (unsigned int i = 0; i < mesh->GetNumFacets(); i++) {
				const Facet& facet = mesh->GetFacet(i);
				vertexBufferData.push_back(facet.v1[0]);
				vertexBufferData.push_back(facet.v1[1]);
				vertexBufferData.push_back(facet.v1[2]);
				vertexBufferData.push_back(facet.v2[0]);
				vertexBufferData.push_back(facet.v2[1]);
				vertexBufferData.push_back(facet.v2[2]);
				vertexBufferData.push_back(facet.v3[0]);
				vertexBufferData.push_back(facet.v3[1]);
				vertexBufferData.push_back(facet.v3[2]);
			}
			// load all of the vertices into a buffer
			Buffer* vertexBuffer = new Buffer(mesh->GetNumFacets() * 9 * sizeof(float), CL_MEM_READ_ONLY);
			vertexBuffer->write(vertexBufferData.data(), mesh->GetNumFacets() * 9 * sizeof(float));
			
			std::cout << "Calculating distance field...";
			unsigned int mesh_start, nFacets = mesh->GetNumFacets();
			for (unsigned int i = 0; i < nItr; i++) {
				mesh_start = i * step;

				Kernel* kernel = distance_program_.compile(KERNEL_CODE, KERNEL_UNSIGNED);
				kernel->block3d(KERNEL_X, KERNEL_Y, KERNEL_Z);
				kernel->grid3d(
					(nPts_[0] + KERNEL_X - 1) / KERNEL_X,
					(nPts_[1] + KERNEL_Y - 1) / KERNEL_Y,
					(nPts_[2] + KERNEL_Z - 1) / KERNEL_Z
				);
				kernel->addBufferArg(dFieldBuffer_);
				kernel->addArg(offset_[0]);
				kernel->addArg(offset_[1]);
				kernel->addArg(offset_[2]);
				kernel->addArg(voxelSize_[0]);
				kernel->addArg(voxelSize_[1]);
				kernel->addArg(voxelSize_[2]);
				kernel->addArg(nPts_[0]); // gridSize
				kernel->addArg(nPts_[1]);
				kernel->addArg(nPts_[2]);
				kernel->addBufferArg(vertexBuffer);
				kernel->addArg(nFacets);
				kernel->addArg(mesh_start); // mesh start
				kernel->addArg(step); // step

				kernel->launch();

				delete kernel;	
			}
			std::cout << "done" << std::endl;

			delete vertexBuffer;
			vertexBufferData.clear();
		}

		void DistanceField::sign_dfield(Mesh *mesh) {
			// create three buffers one for all of the vertex1, vertex2, and vertex3
			std::vector<float> vertex1BufferData;
			std::vector<float> vertex2BufferData;
			std::vector<float> vertex3BufferData;
			vertex1BufferData.reserve(mesh->GetNumFacets() * 3);
			vertex2BufferData.reserve(mesh->GetNumFacets() * 3);
			vertex3BufferData.reserve(mesh->GetNumFacets() * 3);
			for (unsigned int i = 0; i < mesh->GetNumFacets(); i++) {
				const Facet& facet = mesh->GetFacet(i);
				vertex1BufferData.push_back(facet.v1[0]);
				vertex1BufferData.push_back(facet.v1[1]);
				vertex1BufferData.push_back(facet.v1[2]);
				vertex2BufferData.push_back(facet.v2[0]);
				vertex2BufferData.push_back(facet.v2[1]);
				vertex2BufferData.push_back(facet.v2[2]);
				vertex3BufferData.push_back(facet.v3[0]);
				vertex3BufferData.push_back(facet.v3[1]);
				vertex3BufferData.push_back(facet.v3[2]);
			}
			// load all of the vertices into a buffer
			Buffer* vertex1Buffer = new Buffer(mesh->GetNumFacets() * 3 * sizeof(float), CL_MEM_READ_ONLY);
			vertex1Buffer->write(vertex1BufferData.data(), mesh->GetNumFacets() * 3 * sizeof(float));
			Buffer* vertex2Buffer = new Buffer(mesh->GetNumFacets() * 3 * sizeof(float), CL_MEM_READ_ONLY);
			vertex2Buffer->write(vertex2BufferData.data(), mesh->GetNumFacets() * 3 * sizeof(float));
			Buffer* vertex3Buffer = new Buffer(mesh->GetNumFacets() * 3 * sizeof(float), CL_MEM_READ_ONLY);
			vertex3Buffer->write(vertex3BufferData.data(), mesh->GetNumFacets() * 3 * sizeof(float));

			int step = 100;
			unsigned int nItr = mesh->GetNumFacets() / step + 1;
			unsigned int face_start, nFacets = mesh->GetNumFacets();
			std::cout << "Signing distance field...";
			for (int i = 0; i < nItr; i++) {
				face_start = i * step;
				Kernel* kernel = distance_program_.compile(KERNEL_CODE, KERNEL_SIGNED);
				kernel->block3d(KERNEL_X, KERNEL_Y, KERNEL_Z);
				kernel->grid3d(
					(nPts_[0] + KERNEL_X - 1) / KERNEL_X,
					(nPts_[1] + KERNEL_Y - 1) / KERNEL_Y,
					(nPts_[2] + KERNEL_Z - 1) / KERNEL_Z
				);

				kernel->addBufferArg(dFieldBuffer_);
				kernel->addArg(offset_[0]);
				kernel->addArg(offset_[1]);
				kernel->addArg(offset_[2]);
				kernel->addArg(voxelSize_[0]);
				kernel->addArg(voxelSize_[1]);
				kernel->addArg(voxelSize_[2]);
				kernel->addArg(nPts_[0]); // gridSize
				kernel->addArg(nPts_[1]);
				kernel->addArg(nPts_[2]);
				kernel->addBufferArg(vertex1Buffer);
				kernel->addBufferArg(vertex2Buffer);
				kernel->addBufferArg(vertex3Buffer);
				kernel->addArg(nFacets);
				kernel->addArg(face_start); // face start
				kernel->addArg(step); // step

				kernel->launch();

				delete kernel;
			}
			std::cout << "done" << std::endl;

			delete vertex1Buffer;
			delete vertex2Buffer;
			delete vertex3Buffer;
			vertex1BufferData.clear();
			vertex2BufferData.clear();
			vertex3BufferData.clear();
		}

	} } // namespace xromm::gpu