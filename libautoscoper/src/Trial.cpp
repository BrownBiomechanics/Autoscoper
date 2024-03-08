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
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "asys/SystemTools.hxx"
#include "Video.hpp"
#include "Volume.hpp"
#include "Camera.hpp"

#include <filesystem_compat.hpp>

namespace fs = std::filesystem;

namespace xromm {
std::string trialReadingError(const std::string& filename, const std::string& message)
{
  return std::string("Invalid trial configuration file. ") + message
         + "\n"
           "See "
         + filename
         + ".\n"
           "\n"
           "Please check the trial configuration specification.\n"
           "See https://autoscoper.readthedocs.io/en/latest/file-specifications/config.html";
}

Trial::Trial(const std::string& filename)
  : cameras()
  , videos()
  , volumes()
  , frame(0)
  , num_frames(0)
  , guess(0)
  , current_volume(0)
  , num_volumes(0)
{
  if (filename.compare("") == 0) {
    return;
  }

  std::ifstream file(filename.c_str());
  if (file.is_open() == false) {
    throw std::runtime_error("File not found: " + filename);
  }

  std::vector<int> version{ 0, 0 };
  std::vector<std::string> mayaCams;
  std::vector<std::string> camRootDirs;
  std::vector<std::string> volumeFiles;
  std::vector<std::string> voxelSizes;
  std::vector<std::string> volumeFlips;
  std::vector<std::string> renderResolution;
  std::vector<std::string> optimizationOffsets;

  parse(
    file, version, mayaCams, camRootDirs, volumeFiles, voxelSizes, volumeFlips, renderResolution, optimizationOffsets);

  file.close();

  if (version[0] == 0 && version[1] == 0) {
    version[0] = 1;
    version[1] = 0;
    std::cerr << "Trial configuration file is missing the `Version` key. "
              << "Assuming version is " << version[0] << "." << version[1] << "\n"
              << "See " + filename + ".\n"
              << "\n"
              << "Please check the trial configuration specification.\n"
              << "See https://autoscoper.readthedocs.io/en/latest/file-specifications/config.html";
  }

  if (version[0] == 1 && version[1] >= 1) {
    std::string configLocation = fs::path(filename).parent_path().string();
    convertToAbsolutePaths(mayaCams, configLocation);
    convertToAbsolutePaths(camRootDirs, configLocation);
    convertToAbsolutePaths(volumeFiles, configLocation);
  }

  validate(mayaCams, camRootDirs, volumeFiles, voxelSizes, filename);

  loadCameras(mayaCams);

  loadVolumes(volumeFiles, voxelSizes, volumeFlips);

  loadVideos(camRootDirs);

  loadOffsets(optimizationOffsets);

  loadRenderResolution(renderResolution);
}

void Trial::convertToUnixSlashes(std::string& path)
{
  std::replace(path.begin(), path.end(), '\\', '/');
  std::string doubleSlash = "//";
  std::string singleSlash = "/";

  size_t pos = 0;
  while ((pos = path.find(doubleSlash, pos)) != std::string::npos) {
    path.replace(pos, doubleSlash.length(), singleSlash);
    pos += singleSlash.length();
  }

  // remove trailing slash if the path is more than a single /
  if (path.size() > 1 && path.back() == '/') {
    // if it is c:/ then do not remove the trailing slash
    if (!(path.size() == 3 && path[1] == ':')) {
      path.pop_back();
    }
  }
}

void Trial::convertToAbsolutePaths(std::vector<std::string>& paths, const std::string& basePath)
{
  for (size_t idx = 0; idx < paths.size(); ++idx) {
    paths[idx] = toAbsolutePath(paths[idx], basePath);
  }
}

std::string Trial::toAbsolutePath(const std::string& path, const std::string& basePath)
{
  fs::path fsPath = fs::path(path);
  fs::path fsBasePath = fs::path(basePath);
  if (!fsPath.is_absolute()) {
    return (fsBasePath / fsPath).string();
  }
  return path;
}

void Trial::convertToRelativePaths(std::vector<std::string>& paths, const std::string& basePath)
{
  for (size_t idx = 0; idx < paths.size(); ++idx) {
    paths[idx] = toRelativePath(paths[idx], basePath);
  }
}

std::string Trial::toRelativePath(const std::string& path, const std::string& basePath)
{
  return fs::relative(path, basePath).string();
}

void Trial::parse(std::ifstream& file,
                  std::vector<int>& version,
                  std::vector<std::string>& mayaCams,
                  std::vector<std::string>& camRootDirs,
                  std::vector<std::string>& volumeFiles,
                  std::vector<std::string>& voxelSizes,
                  std::vector<std::string>& volumeFlips,
                  std::vector<std::string>& renderResolution,
                  std::vector<std::string>& optimizationOffsets)
{

  std::string line, key, value;
  while (asys::SystemTools::GetLineFromStream(file, line)) {

    // Skip blank lines and commented lines.
    if (line.size() == 0 || line[0] == '\n' || line[0] == '#') {
      continue;
    }

    std::istringstream lineStream(line);
    std::getline(lineStream, key, ' ');
    if (key.compare("mayaCam_csv") == 0) {
      asys::SystemTools::GetLineFromStream(lineStream, value);
      convertToUnixSlashes(value);
      mayaCams.push_back(value);
    } else if (key.compare("CameraRootDir") == 0) {
      asys::SystemTools::GetLineFromStream(lineStream, value);
      convertToUnixSlashes(value);
      camRootDirs.push_back(value);
    } else if (key.compare("VolumeFile") == 0) {
      asys::SystemTools::GetLineFromStream(lineStream, value);
      convertToUnixSlashes(value);
      volumeFiles.push_back(value);
    } else if (key.compare("VolumeFlip") == 0) {
      asys::SystemTools::GetLineFromStream(lineStream, value);
      volumeFlips.push_back(value);
    } else if (key.compare("VoxelSize") == 0) {
      asys::SystemTools::GetLineFromStream(lineStream, value);
      voxelSizes.push_back(value);
    } else if (key.compare("RenderResolution") == 0) {
      asys::SystemTools::GetLineFromStream(lineStream, value);
      renderResolution.push_back(value);
    } else if (key.compare("OptimizationOffsets") == 0) {
      asys::SystemTools::GetLineFromStream(lineStream, value);
      optimizationOffsets.push_back(value);
    } else if (key.compare("Version") == 0) {
      asys::SystemTools::GetLineFromStream(lineStream, value);
      parseVersion(value, version);
      continue;
    }
  }
}

void Trial::parseVersion(const std::string& text, std::vector<int>& version)
{
  std::istringstream versionStream(text);
  for (int idx = 0; idx < 2; ++idx) {
    std::string versionNumber;
    std::getline(versionStream, versionNumber, '.');
    version[idx] = std::atoi(versionNumber.c_str());
  }
}

void Trial::validate(const std::vector<std::string>& mayaCams,
                     const std::vector<std::string>& camRootDirs,
                     const std::vector<std::string>& volumeFiles,
                     const std::vector<std::string>& voxelSizes,
                     const std::string& filename)
{

  // Check that this is a valid trial
  if (mayaCams.size() < 1) {
    throw std::runtime_error(trialReadingError(filename, "There must be at least one mayacam files."));
  }
  if (mayaCams.size() != camRootDirs.size()) {
    throw std::runtime_error(trialReadingError(filename,
                                               std::string("The number of cameras and videos must match.\n") + "Found "
                                                 + std::to_string(mayaCams.size())
                                                 + " cameras "
                                                   "and "
                                                 + std::to_string(camRootDirs.size()) + " videos."));
  }
  if (volumeFiles.size() < 1) {
    throw std::runtime_error(trialReadingError(filename, "There must be at least one volume file."));
  }
  if (volumeFiles.size() != voxelSizes.size()) {
    throw std::runtime_error(
      trialReadingError(filename,
                        std::string("Each volume must be associated with its corresponding voxel sizes.\n") + "Found "
                          + std::to_string(volumeFiles.size())
                          + " volumes "
                            "and "
                          + std::to_string(voxelSizes.size()) + " voxel sizes."));
  }
}

void Trial::loadCameras(std::vector<std::string>& mayaCams)
{
  cameras.clear();
  for (unsigned int i = 0; i < mayaCams.size(); ++i) {
    Camera camera(mayaCams[i]);
    cameras.push_back(camera);
  }
}

void Trial::loadVolumes(std::vector<std::string>& volumeFiles,
                        std::vector<std::string>& voxelSizes,
                        std::vector<std::string>& volumeFlips)
{
  // First load the volumes as more continuous memory is required than for the videos.
  volumes.clear();
  volumestransform.clear();
  for (unsigned int i = 0; i < volumeFiles.size(); ++i) {

    Volume volume(volumeFiles[i]);

    int flip_x = 0, flip_y = 0, flip_z = 0;
    if (i < volumeFlips.size()) {
      std::stringstream volume_flip(volumeFlips[i]);
      volume_flip >> flip_x >> flip_y >> flip_z;
    }

    volume.flipX(flip_x);
    volume.flipY(flip_y);
    volume.flipZ(flip_z);

    float scaleX, scaleY, scaleZ;
    std::stringstream voxelSize(voxelSizes[i]);
    voxelSize >> scaleX >> scaleY >> scaleZ;

    volume.scaleX(scaleX);
    volume.scaleY(scaleY);
    volume.scaleZ(scaleZ);

    volumes.push_back(volume);
    volumestransform.push_back(VolumeTransform());
    num_volumes++;
  }
}

void Trial::loadVideos(std::vector<std::string>& camRootDirs)
{
  int maxVideoFrames = 0;
  videos.clear();
  for (unsigned int i = 0; i < camRootDirs.size(); ++i) {
    Video video(camRootDirs[i]);
    if (video.num_frames() > maxVideoFrames) {
      maxVideoFrames = video.num_frames();
    }
    videos.push_back(video);
  }

  // Initialize the coordinate frames
  num_frames = maxVideoFrames;
}

void Trial::loadOffsets(std::vector<std::string>& optimizationOffsets)
{
  // Read in the offsets, otherwise default to 0.1
  offsets[0] = 0.1;
  offsets[1] = 0.1;
  offsets[2] = 0.1;
  offsets[3] = 0.1;
  offsets[4] = 0.1;
  offsets[5] = 0.1;
  if (!optimizationOffsets.empty()) {
    std::stringstream offset_stream(optimizationOffsets.back());
    offset_stream >> offsets[0] >> offsets[1] >> offsets[2] >> offsets[3] >> offsets[4] >> offsets[5];
  }
}

void Trial::loadRenderResolution(std::vector<std::string>& renderResolution)
{
  // Read in the rendering dimensions, default to 512
  render_width = 512;
  render_height = 512;
  if (!renderResolution.empty()) {
    std::stringstream resolution_stream(renderResolution.back());
    resolution_stream >> render_width >> render_height;
  }
}

void Trial::save(const std::string& filename)
{
  std::vector<std::string> mayaCamsFiles;
  for (const Camera& camera : cameras) {
    mayaCamsFiles.push_back(camera.mayacam());
  }

  std::vector<std::string> camRootDirs;
  for (const Video& video : videos) {
    camRootDirs.push_back(video.dirname());
  }

  std::vector<std::string> volumeFiles;
  for (const Volume& volume : volumes) {
    volumeFiles.push_back(volume.name());
  }

  // Convert to relative paths
  std::string configLocation = fs::path(filename).parent_path().string();
  convertToRelativePaths(mayaCamsFiles, configLocation);
  convertToRelativePaths(camRootDirs, configLocation);
  convertToRelativePaths(volumeFiles, configLocation);

  std::ofstream file(filename.c_str());
  if (!file) {
    throw std::runtime_error("Failed to save to file: " + filename);
  }

  file.precision(12);

  file << "Version 1.1" << std::endl;

  for (const std::string& mayaCamsFile : mayaCamsFiles) {
    file << "mayaCam_csv " << mayaCamsFile << std::endl;
  }

  for (const std::string& camRootDir : camRootDirs) {
    file << "CameraRootDir " << camRootDir << std::endl;
  }

  for (unsigned i = 0; i < volumes.size(); ++i) {
    file << "VolumeFile " << volumeFiles.at(i) << std::endl;
    file << "VolumeFlip " << volumes.at(i).flipX() << " " << volumes.at(i).flipY() << " " << volumes.at(i).flipZ()
         << std::endl;
    file << "VoxelSize " << volumes.at(i).scaleX() << " " << volumes.at(i).scaleY() << " " << volumes.at(i).scaleZ()
         << std::endl;
  }

  file << "RenderResolution " << render_width << " " << render_height << std::endl;

  file << "OptimizationOffsets " << offsets[0] << " " << offsets[1] << " " << offsets[2] << " " << offsets[3] << " "
       << offsets[4] << " " << offsets[5] << std::endl;

  file.close();
}

KeyCurve<float>* Trial::getXCurve(int volumeID)
{
  if (volumestransform.size() <= 0)
    return NULL;

  if (volumeID < volumestransform.size() && volumeID >= 0) {
    return &volumestransform[volumeID].x_curve;
  } else {
    return &volumestransform[current_volume].x_curve;
  }
}

KeyCurve<float>* Trial::getYCurve(int volumeID)
{
  if (volumestransform.size() <= 0)
    return NULL;

  if (volumeID < volumestransform.size() && volumeID >= 0) {
    return &volumestransform[volumeID].y_curve;
  } else {
    return &volumestransform[current_volume].y_curve;
  }
}

KeyCurve<float>* Trial::getZCurve(int volumeID)
{
  if (volumestransform.size() <= 0)
    return NULL;

  if (volumeID < volumestransform.size() && volumeID >= 0) {
    return &volumestransform[volumeID].z_curve;
  } else {
    return &volumestransform[current_volume].z_curve;
  }
}

KeyCurve<Quatf>* Trial::getQuatCurve(int volumeID)
{
  if (volumestransform.size() <= 0)
    return NULL;

  if (volumeID < volumestransform.size() && volumeID >= 0) {
    return &volumestransform[volumeID].quat_curve;
  } else {
    return &volumestransform[current_volume].quat_curve;
  }
}

CoordFrame* Trial::getVolumeMatrix(int volumeID)
{
  if (volumestransform.size() <= 0)
    return NULL;

  if (volumeID < volumestransform.size() && volumeID >= 0) {
    return &volumestransform[volumeID].volumeMatrix;
  } else {
    return &volumestransform[current_volume].volumeMatrix;
  }
}
} // namespace xromm
