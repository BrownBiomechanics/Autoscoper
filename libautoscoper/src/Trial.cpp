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

/// \file Trial.cpp
/// \author Andy Loomis, Benjamin Knorlein

#include "Trial.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "Video.hpp"
#include "Volume.hpp"
#include "Camera.hpp"

using namespace std;

namespace xromm
{

	Trial::Trial(const string& filename)
		: cameras(), videos(), volumes(), frame(0), num_frames(0), guess(0), current_volume(0), num_volumes(0)
	{
		if (filename.compare("") == 0) {
			return;
		}

		// Load the config file.
		ifstream file(filename.c_str());
		if (file.is_open() == false) {
			throw runtime_error("File not found: " + filename);
		}

		vector<string> mayaCams;
		vector<string> camRootDirs;
		vector<string> volumeFiles;
		vector<string> voxelSizes;
		vector<string> volumeFlips;
		vector<string> renderResolution;
		vector<string> optimizationOffsets;

		string line, key, value;
		while (getline(file, line)) {

			// Skip blank lines and commented lines.
			if (line.size() == 0 || line[0] == '\n' || line[0] == '#') {
				continue;
			}

			istringstream lineStream(line);
			getline(lineStream, key, ' ');
			if (key.compare("mayaCam_csv") == 0) {
				getline(lineStream, value);
				mayaCams.push_back(value);
			}
			else if (key.compare("CameraRootDir") == 0) {
				getline(lineStream, value);
				camRootDirs.push_back(value);
			}
			else if (key.compare("VolumeFile") == 0) {
				getline(lineStream, value);
				volumeFiles.push_back(value);
			}
			else if (key.compare("VolumeFlip") == 0) {
				getline(lineStream, value);
				volumeFlips.push_back(value);
			}
			else if (key.compare("VoxelSize") == 0) {
				getline(lineStream, value);
				voxelSizes.push_back(value);
			}
			else if (key.compare("RenderResolution") == 0) {
				getline(lineStream, value);
				renderResolution.push_back(value);
			}
			else if (key.compare("OptimizationOffsets") == 0) {
				getline(lineStream, value);
				optimizationOffsets.push_back(value);
			}
		}

		// Close the file.
		file.close();

		// Check that this is a valid trial
		if (mayaCams.size() < 1) {
			throw runtime_error("There must be at least one mayacam files.");
		}
		if (mayaCams.size() != camRootDirs.size()) {
			throw runtime_error("The number of cameras and videos must match.");
		}
		if (volumeFiles.size() < 1) {
			throw runtime_error("There must be at least one volume file.");
		}
		if (volumeFiles.size() != voxelSizes.size()) {
			throw runtime_error("You must sepcify a voxels size for each volume.");
		}

		cameras.clear();
		for (unsigned int i = 0; i < mayaCams.size(); ++i) {
			try {
				Camera camera(mayaCams[i]);
				cameras.push_back(camera);
			}
			catch (exception& e) {
				cerr << e.what() << endl;
			}
		}

		// First load the volumes as more continous memory is required than for the videos.
		volumes.clear();
		volumestransform.clear();
		for (unsigned int i = 0; i < volumeFiles.size(); ++i) {

			try {
				Volume volume(volumeFiles[i]);

				int flip_x = 0, flip_y = 0, flip_z = 0;
				if (i < volumeFlips.size()) {
					stringstream volume_flip(volumeFlips[i]);
					volume_flip >> flip_x >> flip_y >> flip_z;
				}

				volume.flipX(flip_x);
				volume.flipY(flip_y);
				volume.flipZ(flip_z);

				float scaleX, scaleY, scaleZ;
				stringstream voxelSize(voxelSizes[i]);
				voxelSize >> scaleX >> scaleY >> scaleZ;

				volume.scaleX(scaleX);
				volume.scaleY(scaleY);
				volume.scaleZ(scaleZ);

				volumes.push_back(volume);
				volumestransform.push_back(VolumeTransform());
				num_volumes++;
			}
			catch (exception& e) {
				throw e;
			}
		}

		int maxVideoFrames = 0;
		videos.clear();
		for (unsigned int i = 0; i < camRootDirs.size(); ++i) {
			try {
				Video video(camRootDirs[i]);
				if (video.num_frames() > maxVideoFrames) {
					maxVideoFrames = video.num_frames();
				}
				videos.push_back(video);
			}
			catch (exception& e) {
				cerr << e.what() << endl;
			}
		}

		// Read in the offsets, otherwise default to 0.1
		offsets[0] = 0.1; offsets[1] = 0.1; offsets[2] = 0.1;
		offsets[3] = 0.1; offsets[4] = 0.1; offsets[5] = 0.1;
		if (!optimizationOffsets.empty()) {
			stringstream offset_stream(optimizationOffsets.back());
			offset_stream >> offsets[0] >> offsets[1] >> offsets[2] >>
				offsets[3] >> offsets[4] >> offsets[5];
		}

		// Read in the rendering dimensions, default to 880
		render_width = 880;
		render_height = 880;
		if (!renderResolution.empty()) {
			stringstream resolution_stream(renderResolution.back());
			resolution_stream >> render_width >> render_height;
		}

		// Initialize the coordinate frames
		num_frames = maxVideoFrames;
	}

	void Trial::save(const std::string& filename)
	{
		ofstream file(filename.c_str());
		if (!file) {
			throw runtime_error("Failed to save to file: " + filename);
		}

		file.precision(12);

		for (unsigned i = 0; i < cameras.size(); ++i) {
			file << "mayaCam_csv " << cameras.at(i).mayacam() << endl;
		}

		for (unsigned i = 0; i < videos.size(); ++i) {
			file << "CameraRootDir " << videos.at(i).dirname() << endl;
		}

		for (unsigned i = 0; i < volumes.size(); ++i) {
			file << "VolumeFile " << volumes.at(i).name() << endl;
			file << "VolumeFlip " << volumes.at(i).flipX() << " "
				<< volumes.at(i).flipY() << " "
				<< volumes.at(i).flipZ() << endl;
			file << "VoxelSize " << volumes.at(i).scaleX() << " "
				<< volumes.at(i).scaleY() << " "
				<< volumes.at(i).scaleZ() << endl;
		}

		file << "RenderResolution " << render_width << " "
			<< render_height << endl;

		file << "OptimizationOffsets " << offsets[0] << " "
			<< offsets[1] << " "
			<< offsets[2] << " "
			<< offsets[3] << " "
			<< offsets[4] << " "
			<< offsets[5] << endl;

		file.close();
	}

	KeyCurve* Trial::getXCurve(int volumeID) {
		if (volumestransform.size() <= 0)
			return NULL;

		if (volumeID < volumestransform.size() &&
			volumeID >= 0) {
			return &volumestransform[volumeID].x_curve;
		}
		else {
			return &volumestransform[current_volume].x_curve;
		}
	}

	KeyCurve* Trial::getYCurve(int volumeID) {
		if (volumestransform.size() <= 0)
			return NULL;

		if (volumeID < volumestransform.size() &&
			volumeID >= 0) {
			return &volumestransform[volumeID].y_curve;
		}
		else {
			return &volumestransform[current_volume].y_curve;
		}
	}

	KeyCurve* Trial::getZCurve(int volumeID) {
		if (volumestransform.size() <= 0)
			return NULL;

		if (volumeID < volumestransform.size() &&
			volumeID >= 0) {
			return &volumestransform[volumeID].z_curve;
		}
		else {
			return &volumestransform[current_volume].z_curve;
		}
	}

	KeyCurve* Trial::getYawCurve(int volumeID) {
		if (volumestransform.size() <= 0)
			return NULL;

		if (volumeID < volumestransform.size() &&
			volumeID >= 0) {
			return &volumestransform[volumeID].yaw_curve;
		}
		else {
			return &volumestransform[current_volume].yaw_curve;
		}
	}

	KeyCurve* Trial::getPitchCurve(int volumeID) {
		if (volumestransform.size() <= 0)
			return NULL;

		if (volumeID < volumestransform.size() &&
			volumeID >= 0) {
			return &volumestransform[volumeID].pitch_curve;
		}
		else {
			return &volumestransform[current_volume].pitch_curve;
		}
	}

	KeyCurve* Trial::getRollCurve(int volumeID) {
		if (volumestransform.size() <= 0)
			return NULL;

		if (volumeID < volumestransform.size() &&
			volumeID >= 0) {
			return &volumestransform[volumeID].roll_curve;
		}
		else {
			return &volumestransform[current_volume].roll_curve;
		}
	}

	CoordFrame* Trial::getVolumeMatrix(int volumeID) {
		if (volumestransform.size() <= 0)
			return NULL;

		if (volumeID < volumestransform.size() &&
			volumeID >= 0) {
			return &volumestransform[volumeID].volumeMatrix;
		}
		else {
			return &volumestransform[current_volume].volumeMatrix;
		}
	}



} // namespace xromm
