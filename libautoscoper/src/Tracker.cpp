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

/// \file Tracker.hpp
/// \author Andy Loomis, Benjamin Knorlein

#include "Tracker.hpp"

#include <algorithm>
#include <limits>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#if defined(Autoscoper_RENDERING_USE_CUDA_BACKEND)
  #include "gpu/cuda/CudaWrap.hpp"
  #include "gpu/cuda/Ncc_kernels.h"
  #include "gpu/cuda/HDist_kernels.h"
  #include "gpu/cuda/Compositor_kernels.h"
  #include "gpu/cuda/Mult_kernels.h"
  #include <cuda_runtime_api.h>
#elif defined(Autoscoper_RENDERING_USE_OpenCL_BACKEND)
  #include "gpu/opencl/Ncc.hpp"
  #include "gpu/opencl/Mult.hpp"
#endif

#include "VolumeDescription.hpp"
#include "Video.hpp"
#include "View.hpp"
#include "DownhillSimplex.hpp"
#include "SimulatedAnnealing.hpp"
#include "Camera.hpp"
#include "CoordFrame.hpp"
#include "PSO.hpp"

static bool firstRun = true;

#define DEBUG 0

// XXX
// Set callback for Downhill Simplex. This is really a hack so that we can use
// define the global function pointer FUNC, which Downhill SImplex will
// optimize.
static xromm::Tracker* g_markerless = NULL;

// This is for Downhill Simplex
double FUNC(double* P) { return g_markerless->minimizationFunc(P+1); }
// This is for PSO. P is the 6-DOF manipulator handle.
double PSO_FUNC(double* P) { return g_markerless->minimizationFunc(P); }


namespace xromm {

#if DEBUG
#if defined(Autoscoper_RENDERING_USE_CUDA_BACKEND)
  void save_debug_image(const Buffer* dev_image, int width, int height)
#elif defined(Autoscoper_RENDERING_USE_OpenCL_BACKEND)
  void save_debug_image(const gpu::Buffer* dev_image, int width, int height)
#endif
{
  static int count = 0; // static, so we add to it whenever we run this
  float* host_image = new float[width*height];
  unsigned char* uchar_image = new unsigned char[width*height];

#if defined(Autoscoper_RENDERING_USE_CUDA_BACKEND)
  cudaMemcpy(host_image, dev_image, width*height*sizeof(float), cudaMemcpyDeviceToHost);
#elif defined(Autoscoper_RENDERING_USE_OpenCL_BACKEND)
  dev_image->write(host_image, width*height*sizeof(float));
#endif
#undef max
#undef min
  float minim = std::numeric_limits<float>::max();
  float maxim = std::numeric_limits<float>::min();

  // Copy to a char array
  for (int i = 0; i < width*height; i++) {
    if (host_image[i] > maxim) maxim = host_image[i];
    if (host_image[i] < minim) minim = host_image[i];
  }

  // Copy to a char array
  for (int i = 0; i < width*height; i++) {
    uchar_image[i] = (int)(255*(host_image[i] - minim)/(maxim - minim));
  }

    char filename[256];
    #ifdef __APPLE__
  sprintf(filename,"/Users/bardiya/autoscoper-v2/debug/image_cam%02d.pgm",count++);
    #elif _WIN32
    sprintf(filename,"C:/Users/anthony.lombardi/Desktop/viewport-clip-test/image_cam%02d.pgm",count++);
    #endif

    std::cout << filename << std::endl;
  std::ofstream file(filename, std::ios::out);
  file << "P2" << std::endl;
  file << width << " " << height << std::endl;
  file << 255 << std::endl;
  for (int i = 0; i < width*height; i++) {
    file << 255-(int)uchar_image[i] << " "; // (255-X) because we want white to be air
  }
    file.close(); // we have to flip this vertically for the actual image
  delete[] uchar_image;
  delete[] host_image;
}
#endif


#if defined(Autoscoper_RENDERING_USE_CUDA_BACKEND)
  void save_full_drr(const Buffer* dev_image, int width, int height)
#elif defined(Autoscoper_RENDERING_USE_OpenCL_BACKEND)
  void save_full_drr(const gpu::Buffer* dev_image, int width, int height)
#endif
  {
    static int count = 0; // static, so we add to it whenever we run this
    float* host_image = new float[width * height];
    unsigned char* uchar_image = new unsigned char[width * height];

#if defined(Autoscoper_RENDERING_USE_CUDA_BACKEND)
    cudaMemcpy(host_image, dev_image, width * height * sizeof(float), cudaMemcpyDeviceToHost);
#elif defined(Autoscoper_RENDERING_USE_OpenCL_BACKEND)
    dev_image->write(host_image, width * height * sizeof(float));
#endif
#undef max
#undef min
    float minim = std::numeric_limits<float>::max();
    float maxim = std::numeric_limits<float>::min();

    // Copy to a char array
    for (int i = 0; i < width * height; i++) {
      if (host_image[i] > maxim) maxim = host_image[i];
      if (host_image[i] < minim) minim = host_image[i];
    }

    // Copy to a char array
    for (int i = 0; i < width * height; i++) {
      uchar_image[i] = (int)(255 * (host_image[i] - minim) / (maxim - minim));
    }

    char filename[256];
#ifdef __APPLE__
    sprintf(filename, "/Users/bardiya/autoscoper-v2/my_drr/image_cam%02d.pgm", count++);
#elif _WIN32
    sprintf_s(filename, "C:/_MyDRRs/image_cam%02d.pgm", count++);
#endif

    std::cout << filename << std::endl;
    std::ofstream file(filename, std::ios::out);
    file << "P2" << std::endl;
    file << width << " " << height << std::endl;
    file << 255 << std::endl;
    for (int i = 0; i < width * height; i++) {
      file << 255 - (int)uchar_image[i] << " "; // (255-X) because we want white to be air
    }
    file.close(); // we have to flip this vertically for the actual image
    delete[] uchar_image;
    delete[] host_image;
  }


Tracker::Tracker()
    : rendered_drr_(NULL),
      rendered_rad_(NULL),
    drr_mask_(NULL),
    background_mask_(NULL)
{
    g_markerless = this;
  optimization_method = 0; // initialize cost function
  cf_model_select = 0; //cost function selector
  intializeRandom();
}

Tracker::~Tracker()
{
  for (int i = 0; i < volumeDescription_.size(); i++){
    delete volumeDescription_[i];
  }
  volumeDescription_.clear();

  for (int i = 0; i < views_.size(); i++) {
    delete views_[i];
  }
  views_.clear();
}

//void Tracker::init()
//{
//#if defined(Autoscoper_RENDERING_USE_CUDA_BACKEND)
//    gpu::cudaInitWrap();
//#endif
//}

void Tracker::load(const Trial& trial)
{
    trial_ = trial;

    std::vector<gpu::View*>::iterator viewIter;
    for (viewIter = views_.begin(); viewIter != views_.end(); ++viewIter) {
        delete *viewIter;
    }
    views_.clear();

  for (int i = 0; i < volumeDescription_.size(); i++){
    delete volumeDescription_[i];
  }
  volumeDescription_.clear();
  for (int i = 0; i < trial_.volumes.size(); i++){
    gpu::VolumeDescription * v_desc = new gpu::VolumeDescription(trial_.volumes[i]);
    volumeDescription_.push_back(v_desc);
    //center pivot
    trial_.getVolumeMatrix(i)->translate(v_desc->transCenter());
  }

  unsigned npixels = trial_.render_width*trial_.render_height;
#if defined(Autoscoper_RENDERING_USE_CUDA_BACKEND)
  gpu::cudaMallocWrap(rendered_drr_,trial_.render_width*trial_.render_height*sizeof(float));
    gpu::cudaMallocWrap(rendered_rad_,trial_.render_width*trial_.render_height*sizeof(float));
  gpu::cudaMallocWrap(drr_mask_, trial_.render_width*trial_.render_height*sizeof(float));
  gpu::cudaMallocWrap(background_mask_, trial_.render_width*trial_.render_height*sizeof(float));
  gpu::fill(drr_mask_, trial_.render_width*trial_.render_height, 1.0f);
  gpu::fill(background_mask_, trial_.render_width*trial_.render_height, 1.0f);
#elif defined(Autoscoper_RENDERING_USE_OpenCL_BACKEND)
  rendered_drr_ = new gpu::Buffer(npixels*sizeof(float));
  rendered_rad_ = new gpu::Buffer(npixels*sizeof(float));
  drr_mask_ = new gpu::Buffer(npixels*sizeof(float));
  background_mask_ = new gpu::Buffer(npixels*sizeof(float));
  drr_mask_->fill(1.0f);
  background_mask_->fill(1.0f);
#endif

    gpu::ncc_init(npixels);
  #if defined(Autoscoper_RENDERING_USE_CUDA_BACKEND) // trying another cost function (Housdorff)
    gpu::hdist_init(npixels);
  #endif

    for (unsigned int i = 0; i < trial_.cameras.size(); ++i) {

        Camera& camera = trial_.cameras.at(i);
        Video& video  = trial_.videos.at(i);

        video.set_frame(trial_.frame);

        gpu::View* view = new gpu::View(camera);

    for (int i = 0; i < volumeDescription_.size(); i++){
      view->addDrrRenderer();
      view->drrRenderer(i)->setVolume(*volumeDescription_[i]);
    }

        view->radRenderer()->set_image_plane(camera.viewport()[0],
                                             camera.viewport()[1],
                                             camera.viewport()[2],
                                             camera.viewport()[3]);
        view->radRenderer()->set_rad(video.data(),
                                     video.width(),
                                     video.height(),
                                     video.bps());
    view->backgroundRenderer()->set_image_plane(camera.viewport()[0],
      camera.viewport()[1],
      camera.viewport()[2],
      camera.viewport()[3]);

        views_.push_back(view);
    }

}

void Tracker::optimize(int frame, int dFrame, int repeats, int opt_method, unsigned int max_iter, double min_limit, double max_limit, int cf_model, unsigned int max_stall_iter)
{

  // Upon a call to the Tracker::optimize method, the following steps are performed:
  //
  // 1) Set the heuristic to determine the initial guess for the volume pose
  // 2) Start the particle swarm optimization (PSO) based on the initial guess
  // 3) For each iteration of the PSO:
  //  a) Generate 100 random particles around the current best guess
  //  b) Evaluate the fitness of each particle using normalized cross correlation (NCC)
  //  c) Update the best guess based on the best particle
  //  d) Repeat until the maximum number of iterations without change (MAX_STALL) is
  //     reached and return the best guess
  // 4) Run Downhill Simplex (Nelder-Mead) on the best guess from the PSO
  // 5) Update the volume pose based on the best guess

  intializeRandom();

  optimization_method = opt_method;
  cf_model_select = cf_model;

    if (frame < 0 || frame >= trial_.num_frames) {
        std::cerr << "Tracker::optimize(): Invalid frame." << std::endl;
        return;
    }

    int NDIM = 6;       // Number of dimensions to optimize over.
    double FTOL = 1e-3; // Tolerance for the optimization.
    MAT P;              // Matrix of points to initialize the routine.
    double Y[MP];       // The values of the minimization function at the
                        // initial points.

    int ITER = 0;

    trial_.frame = frame;

    for (unsigned int i = 0; i < trial_.videos.size(); ++i) {

        trial_.videos.at(i).set_frame(trial_.frame);
        views_[i]->radRenderer()->set_rad(trial_.videos.at(i).data(),
                                          trial_.videos.at(i).width(),
                                          trial_.videos.at(i).height(),
                                          trial_.videos.at(i).bps());
    }

    int framesBehind = (dFrame > 0)?
                       (int)trial_.frame:
                       (int)trial_.num_frames-trial_.frame-1;

    if (trial_.guess == 2 && framesBehind > 1) {
    double xyzypr1[6] = { (*trial_.getXCurve(-1))(trial_.frame - 2 * dFrame),
      (*trial_.getYCurve(-1))(trial_.frame - 2 * dFrame),
      (*trial_.getZCurve(-1))(trial_.frame - 2 * dFrame),
      (*trial_.getYawCurve(-1))(trial_.frame - 2 * dFrame),
      (*trial_.getPitchCurve(-1))(trial_.frame - 2 * dFrame),
      (*trial_.getRollCurve(-1))(trial_.frame - 2 * dFrame) };
    double xyzypr2[6] = { (*trial_.getXCurve(-1))(trial_.frame - dFrame),
      (*trial_.getYCurve(-1))(trial_.frame - dFrame),
      (*trial_.getZCurve(-1))(trial_.frame - dFrame),
      (*trial_.getYawCurve(-1))(trial_.frame - dFrame),
      (*trial_.getPitchCurve(-1))(trial_.frame - dFrame),
      (*trial_.getRollCurve(-1))(trial_.frame - dFrame) };


        CoordFrame xcframe = CoordFrame::from_xyzypr(xyzypr1).linear_extrap(
                             CoordFrame::from_xyzypr(xyzypr2));

        xcframe.to_xyzypr(xyzypr1);
    trial_.getXCurve(-1)->insert(trial_.frame, xyzypr1[0]);
    trial_.getYCurve(-1)->insert(trial_.frame, xyzypr1[1]);
    trial_.getZCurve(-1)->insert(trial_.frame, xyzypr1[2]);
    trial_.getYawCurve(-1)->insert(trial_.frame, xyzypr1[3]);
    trial_.getPitchCurve(-1)->insert(trial_.frame, xyzypr1[4]);
    trial_.getRollCurve(-1)->insert(trial_.frame, xyzypr1[5]);
    }
    else if (trial_.guess == 1 && framesBehind > 0) {
    trial_.getXCurve(-1)->insert(trial_.frame, (*trial_.getXCurve(-1))(trial_.frame - dFrame));
    trial_.getYCurve(-1)->insert(trial_.frame, (*trial_.getYCurve(-1))(trial_.frame - dFrame));
    trial_.getZCurve(-1)->insert(trial_.frame, (*trial_.getZCurve(-1))(trial_.frame - dFrame));
    trial_.getYawCurve(-1)->insert(trial_.frame, (*trial_.getYawCurve(-1))(trial_.frame - dFrame));
    trial_.getPitchCurve(-1)->insert(trial_.frame, (*trial_.getPitchCurve(-1))(trial_.frame - dFrame));
    trial_.getRollCurve(-1)->insert(trial_.frame, (*trial_.getRollCurve(-1))(trial_.frame - dFrame));
    }

    int totalIter = 0;
    for (int j = 0; j < repeats; j++) {

    // Get Current Pose
    double xyzypr[6] = { (*trial_.getXCurve(-1))(trial_.frame),
      (*trial_.getYCurve(-1))(trial_.frame),
      (*trial_.getZCurve(-1))(trial_.frame),
      (*trial_.getYawCurve(-1))(trial_.frame),
      (*trial_.getPitchCurve(-1))(trial_.frame),
      (*trial_.getRollCurve(-1))(trial_.frame) };

    // Init Manip for saving final optimum
    double init_manip[6] = { 0 };
    double init_ncc = minimizationFunc(init_manip);

    CoordFrame manip = CoordFrame::from_xyzAxis_angle(init_manip);

    if (optimization_method == 0) // 0: PSO + Downhill Simplex, 1: Downhill Simplex
    {
      // PSO Algorithm

      //int gBest = 0;
      //int gBestTest = 1000;
      int stall_iter = 0;
      bool done = false;
      float START_RANGE_MIN = (float)min_limit;
      float START_RANGE_MAX = (float)max_limit;
      unsigned int MAX_EPOCHS = max_iter;
      unsigned int MAX_STALL = max_stall_iter;

      clock_t cpu_begin = clock();
      Particle gBest = pso(START_RANGE_MIN, START_RANGE_MAX, MAX_EPOCHS, MAX_STALL);
      clock_t cpu_end = clock();

      printf("Time elapsed:%10.3lf s\n", (double)(cpu_end - cpu_begin) / CLOCKS_PER_SEC);

      using ::operator<<; // Access the stream operator from the global namespace
      std::cout << "Pose change from initial position: " << gBest.Position << std::endl;

      printf("Minimum NCC from PSO = %f\n", gBest.NCC);

      double xyzypr_manip[NUM_OF_DIMENSIONS] = { 0 };
      std::copy(gBest.Position.begin(), gBest.Position.begin() + NUM_OF_DIMENSIONS, xyzypr_manip);

      manip = CoordFrame::from_xyzAxis_angle(xyzypr_manip);
      // PSO End

      // Additional DOWNHILL SIMPLEX AT THE END:
      // Move the pose to the optimized pose
      // Convert Current Pose to its Coordinate System Frame
      CoordFrame xcframe = CoordFrame::from_xyzypr(xyzypr);

      xcframe = xcframe * trial_.getVolumeMatrix(-1)->inverse() * manip * *trial_.getVolumeMatrix(-1);
      xcframe.to_xyzypr(xyzypr);

      xcframe = xcframe * trial_.getVolumeMatrix(-1)->inverse() * manip * *trial_.getVolumeMatrix(-1);

      trial_.getXCurve(-1)->insert(trial_.frame, xyzypr[0]);
      trial_.getYCurve(-1)->insert(trial_.frame, xyzypr[1]);
      trial_.getZCurve(-1)->insert(trial_.frame, xyzypr[2]);
      trial_.getYawCurve(-1)->insert(trial_.frame, xyzypr[3]);
      trial_.getPitchCurve(-1)->insert(trial_.frame, xyzypr[4]);
      trial_.getRollCurve(-1)->insert(trial_.frame, xyzypr[5]);

      // DOWNHILL SIMPLEX
      // Generate the 7 vertices that form the initial simplex. Because
      // the independent variables of the function we are optimizing over
      // are relative to the initial guess, the same vertices can be used
      // to form the initial simplex for every frame.
      for (int i = 0; i < 7; ++i) {
        P[i + 1][1] = (i == 1) ? trial_.offsets[0] : 0.0;
        P[i + 1][2] = (i == 2) ? trial_.offsets[1] : 0.0;
        P[i + 1][3] = (i == 3) ? trial_.offsets[2] : 0.0;
        P[i + 1][4] = (i == 4) ? trial_.offsets[3] : 0.0;
        P[i + 1][5] = (i == 5) ? trial_.offsets[4] : 0.0;
        P[i + 1][6] = (i == 6) ? trial_.offsets[5] : 0.0;
      }

      // Determine the function values at the vertices of the initial
      // simplex
      for (int i = 0; i < 7; ++i) {
        Y[i + 1] = FUNC(P[i + 1]);
      }

      // Downhill Simplex Optimization
      // Optimize the frame
      AMOEBA(P, Y, NDIM, FTOL, &ITER);

      // get the final ncc value to print out for user
      double final_ncc = minimizationFunc((P[1] + 1));

      if (final_ncc < init_ncc) {
        std::cout << "Downhill Simplex Optimized Final NCC: " << minimizationFunc((P[1] + 1)) << std::endl;
        // For Downhill Simplex Method (Final)
        manip = CoordFrame::from_xyzAxis_angle(P[1] + 1);
      }
      else {
        std::cout << "The initial position was optimized." << std::endl;
        manip = CoordFrame::from_xyzAxis_angle(init_manip);
      }

    }
    else {
      // DOWNHILL SIMPLEX Only
      // Generate the 7 vertices that form the initial simplex. Because
      // the independent variables of the function we are optimizing over
      // are relative to the initial guess, the same vertices can be used
      // to form the initial simplex for every frame.
      for (int i = 0; i < 7; ++i) {
        P[i + 1][1] = (i == 1) ? trial_.offsets[0] : 0.0;
        P[i + 1][2] = (i == 2) ? trial_.offsets[1] : 0.0;
        P[i + 1][3] = (i == 3) ? trial_.offsets[2] : 0.0;
        P[i + 1][4] = (i == 4) ? trial_.offsets[3] : 0.0;
        P[i + 1][5] = (i == 5) ? trial_.offsets[4] : 0.0;
        P[i + 1][6] = (i == 6) ? trial_.offsets[5] : 0.0;
      }

      // Determine the function values at the vertices of the initial
      // simplex
      for (int i = 0; i < 7; ++i) {
        Y[i + 1] = FUNC(P[i + 1]);
      }

      // Downhill Simplex Optimization
      // Optimize the frame
      AMOEBA(P, Y, NDIM, FTOL, &ITER);

      std::cout << "Optimized Final NCC: " << minimizationFunc((P[1] + 1)) << std::endl;

      // For Downhill Simplex Method
      manip = CoordFrame::from_xyzAxis_angle(P[1] + 1);
    }

    // Move the pose to the optimized pose
    // Convert Current Pose to its Coordinate System Frame
        CoordFrame xcframe = CoordFrame::from_xyzypr(xyzypr);

    xcframe = xcframe * trial_.getVolumeMatrix(-1)->inverse() * manip * *trial_.getVolumeMatrix(-1);
        xcframe.to_xyzypr(xyzypr);

    xcframe = xcframe * trial_.getVolumeMatrix(-1)->inverse() * manip * *trial_.getVolumeMatrix(-1);


    trial_.getXCurve(-1)->insert(trial_.frame, xyzypr[0]);
    trial_.getYCurve(-1)->insert(trial_.frame, xyzypr[1]);
    trial_.getZCurve(-1)->insert(trial_.frame, xyzypr[2]);
    trial_.getYawCurve(-1)->insert(trial_.frame, xyzypr[3]);
    trial_.getPitchCurve(-1)->insert(trial_.frame, xyzypr[4]);
    trial_.getRollCurve(-1)->insert(trial_.frame, xyzypr[5]);
    }
  totalIter += ITER;

    std::cerr << "Tracker::optimize(): Frame " << trial_.frame
         << " done in " << totalIter << " total iterations" << std::endl;
}


// Calculate Correlation for Bone Matching
std::vector <double> Tracker::trackFrame(unsigned int volumeID, double* xyzypr) const
{

    // Upon a call to the Tracker::trackFrame() function, the following steps are performed:
    //
    // 1) Given a new pose, for each camera view:
    //  a) Compute the modelView matrix
    //  b) Project the volume onto the film plane and calculate the 2D bounding box top left corner location,
    //    and pixel dimension
    //  c) Compute the size of the bounding box in pixels, based on the render resolution
    //  d) Pass the bounding box (image space) and pixel size to the DRR, radiograph, mask, and background renderers
    //  e) Render the DRR, radiograph, mask, and background images
    //  f) Use the DRR mask to "blank out" the areas of the radiograph not covered by the DRR
    //  g) Compute the normalized cross correlation (NCC) between the radiograph and DRR
    //  h) Compute (1.0 - NCC) value, this turns the problem from a maximization to a minimization one
    // 2) Return an NCC value for each camera view (used in minimizationFunc())

    std::vector<double> correlations;
    CoordFrame xcframe = CoordFrame::from_xyzypr(xyzypr);

    for (unsigned int i = 0; i < views_.size(); ++i) {
      // Set the modelview matrix for DRR rendering
      // (1a)
      CoordFrame modelview = views_[i]->camera()->coord_frame().inverse()*xcframe;
      double imv[16]; modelview.inverse().to_matrix_row_order(imv);
      views_[i]->drrRenderer(volumeID)->setInvModelView(imv);

      // Calculate the viewport surrounding the volume
      // (1b)
      double viewport[4];
      if (!this->calculate_viewport(modelview, *views_[i]->camera(), viewport)) {
        std::cerr << "Tracker::trackFrame(): Volume " << volumeID << " is not in view of camera " << i << std::endl;
        correlations.push_back(0.0);
        continue;
      }

      // Calculate the size of the image to render based on the viewport
      // (1c)
      unsigned render_width = viewport[2] * trial_.render_width / views_[i]->camera()->viewport()[2];
      unsigned render_height = viewport[3] * trial_.render_height / views_[i]->camera()->viewport()[3];

      if (render_width > trial_.render_width || render_height > trial_.render_height) {
        throw std::runtime_error("Tracker::trackFrame(): Rendered image is larger than the viewport buffer!\n" + std::to_string(render_width) + " > " + std::to_string(trial_.render_width) + " || " + std::to_string(render_height) + " > " + std::to_string(trial_.render_height));
      }

      // Set the viewports
      // (1d)
      views_[i]->drrRenderer(volumeID)->setViewport(viewport[0], viewport[1],
        viewport[2], viewport[3]);
      views_[i]->radRenderer()->set_viewport(viewport[0], viewport[1],
        viewport[2], viewport[3]);

      // DRR projection
      // Performed by the RayCaster class
      // called by the Tracker::trackFrame() method
      // @ = DRR pixel with some intensity value
      // -------------------------------
      // |                             |
      // |             ^               |
      // |            /@\              |
      // |           /@@@\             |
      // |          /@@@@@\            |
      // |          \@@@@@/            |
      // |           \@@@/             |
      // |            \@/              |
      // |             v               |
      // |                             |
      // -------------------------------
      //
      // Crop the radiograph to the bounds of the DRR
      // Performed by the RadiographRenderer class
      // called by the Tracker::trackFrame() method
      //
      // Radiograph before cropping:
      // $ = some pixel with some intensity value, inside the region of interest
      // & = some pixel with some intensity value, outside the region of interest
      // ! = some pixel with some intensity value, inside the object of interset
      // -----------------------------------------
      // |&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&|
      // |&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&|
      // |&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&|
      // |&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&|
      // |&&&&&&&&$$$$$$$$$$$$$$$$$$$$$$$$$$$$&&&|
      // |&&&&&&&&$$$$$$$$$$$$$^$$$$$$$$$$$$$$&&&|
      // |&&&&&&&&$$$$$$$$$$$$/!\$$$$$$$$$$$$$&&&|
      // |&&&&&&&&$$$$$$$$$$$/!!!\$$$$$$$$$$$$&&&|
      // |&&&&&&&&$$$$$$$$$$/!!!!!\$$$$$$$$$$$&&&|
      // |&&&&&&&&$$$$$$$$$$\!!!!!/$$$$$$$$$$$&&&|
      // |&&&&&&&&$$$$$$$$$$$\!!!/$$$$$$$$$$$$&&&|
      // |&&&&&&&&$$$$$$$$$$$$\!/$$$$$$$$$$$$$&&&|
      // |&&&&&&&&$$$$$$$$$$$$$v$$$$$$$$$$$$$$&&&|
      // |&&&&&&&&$$$$$$$$$$$$$$$$$$$$$$$$$$$$&&&|
      // |&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&|
      // |&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&|
      // |&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&|
      // -----------------------------------------
      //
      // Radiograph after cropping:
      // -------------------------------
      // |$$$$$$$$$$$$$$$$$$$$$$$$$$$$$|
      // |$$$$$$$$$$$$$^$$$$$$$$$$$$$$$|
      // |$$$$$$$$$$$$/!\$$$$$$$$$$$$$$|
      // |$$$$$$$$$$$/!!!\$$$$$$$$$$$$$|
      // |$$$$$$$$$$/!!!!!\$$$$$$$$$$$$|
      // |$$$$$$$$$$\!!!!!/$$$$$$$$$$$$|
      // |$$$$$$$$$$$\!!!/$$$$$$$$$$$$$|
      // |$$$$$$$$$$$$\!/$$$$$$$$$$$$$$|
      // |$$$$$$$$$$$$$v$$$$$$$$$$$$$$$|
      // |$$$$$$$$$$$$$$$$$$$$$$$$$$$$$|
      // -------------------------------
      //
      // Create the background mask
      // Performed by the backgoundRenderer class
      // called by the Tracker::trackFrame() method
      // This just contains the background pixels
      // 255 / white
      //
      // Create the binary DRR mask
      // -------------------------------
      // |11111111111111111111111111111|
      // |11111111111110111111111111111|
      // |11111111111100011111111111111|
      // |11111111111000001111111111111|
      // |11111111110000000111111111111|
      // |11111111110000000111111111111|
      // |11111111111000001111111111111|
      // |11111111111100011111111111111|
      // |11111111111110111111111111111|
      // |11111111111111111111111111111|
      // -------------------------------
      //
      // Mask the radiograph with the DRR mask
      // Performed by multiplying the DRR mask by the radiograph
      // See the gpu::multiply function in the gpu namespace
      // called by the Tracker::trackFrame() method
      // We are assigning 1 to be 255 and 0 to
      // be the value of the radiograph pixel
      // -------------------------------
      // |                             |
      // |             ^               |
      // |            /!\              |
      // |           /!!!\             |
      // |          /!!!!!\            |
      // |          \!!!!!/            |
      // |           \!!!/             |
      // |            \!/              |
      // |             v               |
      // |                             |
      // -------------------------------
      //
      // Calculate the NCC between the DRR and the masked radiograph
      //

      // (1e)

      // Render the DRR and Radiograph
      views_[i]->renderDrrSingle(volumeID, rendered_drr_, render_width, render_height);
      views_[i]->renderRad(rendered_rad_, render_width, render_height);

      //render masks
      views_[i]->backgroundRenderer()->set_viewport(viewport[0], viewport[1],
        viewport[2], viewport[3]);

      views_[i]->renderBackground(background_mask_, render_width, render_height);
      views_[i]->renderDRRMask(rendered_drr_, drr_mask_, render_width, render_height);

      // (1f)
      gpu::multiply(background_mask_, drr_mask_, drr_mask_, render_width, render_height);
      gpu::multiply(rendered_rad_, drr_mask_, rendered_rad_, render_width, render_height);
      gpu::multiply(rendered_drr_, drr_mask_, rendered_drr_, render_width, render_height);

#if DEBUG
      save_debug_image(rendered_drr_, render_width, render_height);
      //save_debug_image(rendered_rad_, render_width, render_height);
      //save_debug_image(drr_mask_, render_width, render_height);
      //save_debug_image(background_mask_, render_width, render_height);
#endif
      if (cf_model_select) // If 1, we do Implant
      {
        // Calculate the correlation for implant
        // Calculate Hausdorff Distance for Implant Matching _ FUTURE
        #if defined(Autoscoper_RENDERING_USE_CUDA_BACKEND)
          correlations.push_back(gpu::hdist(rendered_drr_, rendered_rad_, drr_mask_, render_width*render_height));
        #endif
      }
      else { // If 0, we do bone model
        // Calculate the correlation for ncc
        // (1g) and (1h)
        correlations.push_back(1.0 - gpu::ncc(rendered_drr_, rendered_rad_, drr_mask_, render_width*render_height));
      }

    }
    return correlations;
  }
// Minimizing Function for Bone Matching
double Tracker::minimizationFunc(const double* values) const
{
  // Construct a coordinate frame from the given values

  double xyzypr[6] = { (*(const_cast<Trial&>(trial_)).getXCurve(-1))(trial_.frame),
    (*(const_cast<Trial&>(trial_)).getYCurve(-1))(trial_.frame),
    (*(const_cast<Trial&>(trial_)).getZCurve(-1))(trial_.frame),
    (*(const_cast<Trial&>(trial_)).getYawCurve(-1))(trial_.frame),
    (*(const_cast<Trial&>(trial_)).getPitchCurve(-1))(trial_.frame),
    (*(const_cast<Trial&>(trial_)).getRollCurve(-1))(trial_.frame) };
    CoordFrame xcframe = CoordFrame::from_xyzypr(xyzypr);


  CoordFrame manip = CoordFrame::from_xyzAxis_angle(values);
  xcframe = xcframe * (const_cast<Trial&>(trial_)).getVolumeMatrix(-1)->inverse() * manip * *(const_cast<Trial&>(trial_)).getVolumeMatrix(-1);

  unsigned int idx = trial_.current_volume;
  xcframe.to_xyzypr(xyzypr);
  std::vector <double> correlations = trackFrame(idx, &xyzypr[0]);

  double correlation = correlations[0];
  //printf("Cam 0: %4.5f", correlation);
  for (unsigned int i = 1; i < trial_.cameras.size(); ++i) {
    correlation += correlations[i];
  //  printf("\tCam %d: %4.5f", i, correlations[i]);
  }
  //printf("\tFinal NCC: %4.5f\n", correlation);
  if (correlation < 0) { correlation = 9999; } // In case we have a really bad filters and correlation ends up negative... This should not happen...
  return correlation;
}

void Tracker::updateBackground()
{
  for (unsigned int i = 0; i < views_.size(); ++i) {
    views_[i]->updateBackground(trial_.videos[i].background(), trial_.videos[i].width(), trial_.videos[i].height());
  }
}

void Tracker::setBackgroundThreshold(float threshold)
{
  for (unsigned int i = 0; i < views_.size(); ++i) {
    views_[i]->setBackgroundThreshold(threshold);
  }
}


#if defined(Autoscoper_RENDERING_USE_CUDA_BACKEND)
  void get_image(const Buffer* dev_image, int width, int height, std::vector<unsigned char> &data)
#elif defined(Autoscoper_RENDERING_USE_OpenCL_BACKEND)
  void get_image(const gpu::Buffer* dev_image, int width, int height, std::vector<unsigned char> &data)
#endif
  {
    static int count = 0;
    float* host_image = new float[width*height];

#if defined(Autoscoper_RENDERING_USE_CUDA_BACKEND)
    cudaMemcpy(host_image, dev_image, width*height*sizeof(float), cudaMemcpyDeviceToHost);
#elif defined(Autoscoper_RENDERING_USE_OpenCL_BACKEND)
    dev_image->write(host_image, width*height*sizeof(float));
#endif
    // Copy to a char array
    for (int i = 0; i < width*height; i++) {
      data.push_back((unsigned char)(255 * host_image[i]));
    }

    delete[] host_image;
  }


std::vector<unsigned char> Tracker::getImageData(unsigned volumeID, unsigned camera, double* xyzypr, unsigned& width, unsigned& height)
{
    trial()->current_volume = volumeID;

    CoordFrame xcframe = CoordFrame::from_xyzypr(xyzypr);

    CoordFrame modelview = views_[camera]->camera()->coord_frame().inverse()*xcframe;
    double imv[16]; modelview.inverse().to_matrix_row_order(imv);
    views_[camera]->drrRenderer(volumeID)->setInvModelView(imv);

    // Calculate the viewport surrounding the volume
    double viewport[4];
    //this->calculate_viewport(modelview, viewport);

        // Export Full Images
        viewport[0] = -views_[camera]->camera()->viewport()[2]/2;
        viewport[1] = -views_[camera]->camera()->viewport()[3]/2;
        viewport[2] = views_[camera]->camera()->viewport()[2];
        viewport[3] = views_[camera]->camera()->viewport()[3];

    // Calculate the size of the image to render
    unsigned render_width = viewport[2] * trial_.render_width / views_[camera]->camera()->viewport()[2];
    unsigned render_height = viewport[3] * trial_.render_height / views_[camera]->camera()->viewport()[3];

    // Set the viewports
    views_[camera]->drrRenderer(volumeID)->setViewport(viewport[0], viewport[1],
      viewport[2], viewport[3]);
    views_[camera]->radRenderer()->set_viewport(viewport[0], viewport[1],
      viewport[2], viewport[3]);

    // Render the DRR and Radiograph
    views_[camera]->renderDrrSingle(volumeID, rendered_drr_, render_width, render_height);
    views_[camera]->renderRad(rendered_rad_, render_width, render_height);

    //render masks
    views_[camera]->backgroundRenderer()->set_viewport(viewport[0], viewport[1],
      viewport[2], viewport[3]);

    views_[camera]->renderBackground(background_mask_, render_width, render_height);
    views_[camera]->renderDRRMask(rendered_drr_, drr_mask_, render_width, render_height);

    gpu::multiply(background_mask_, drr_mask_, drr_mask_, render_width, render_height);
    gpu::multiply(rendered_rad_, drr_mask_, rendered_rad_, render_width, render_height);
    gpu::multiply(rendered_drr_, drr_mask_, rendered_drr_, render_width, render_height);

    width = render_width;
    height = render_height;


    std::vector<unsigned char> out_data;
    get_image(rendered_rad_, width, height, out_data);
    get_image(rendered_drr_, width, height, out_data);
    get_image(drr_mask_, width, height, out_data);

    return out_data;
}

bool Tracker::calculate_viewport(const CoordFrame& modelview, const Camera& camera, double* viewport) const
{
  //
  // Despite being called a viewport, the output is actually the 2D bounding box of
  // the volume after it has been projected onto the film/view plane.
  //
  // The term viewport refers to the coordinate system the bounding box is returned in.
  //
  // The film plane sits at z = -2.
  // The bounding box (viewport) is returned as {x_min, y_min, x_max, y_max}
  //
  // Calculate the minimum and maximum values of the bounding box
  // corners after they have been projected onto the view plane
  //

    double min_max[4] = {1.0,1.0,-1.0,-1.0};
    double corners[24] = {0,0,-1,0,0,0, 0,1,-1,0,1,0, 1,0,-1,1,0,0,1,1,-1,1,1,0};

  int idx = trial_.current_volume;

    for (int j = 0; j < 8; j++) {

        // Calculate the location of the corner in object space
    corners[3 * j + 0] = (corners[3 * j + 0] - volumeDescription_[idx]->invTrans()[0]) / volumeDescription_[idx]->invScale()[0];
    corners[3 * j + 1] = (corners[3 * j + 1] - volumeDescription_[idx]->invTrans()[1]) / volumeDescription_[idx]->invScale()[1];
    corners[3 * j + 2] = (corners[3 * j + 2] - volumeDescription_[idx]->invTrans()[2]) / volumeDescription_[idx]->invScale()[2];

        // Calculate the location of the corner in camera space
        double corner[3];
        modelview.point_to_world_space(&corners[3*j],corner);

        // Calculate its projection onto the film plane, where z = -2
        // xfp = - f * xc / zc
        // yfp = - f * yc / zc
        // Where, (f)ocal length = 2, *c refers to camera coordinates, and *fp refers to film plane coordinates
        double film_plane[3];
        film_plane[0] = -2*corner[0]/corner[2];
        film_plane[1] = -2*corner[1]/corner[2];

        // Update the min and max values
        if (min_max[0] > film_plane[0]) {
            min_max[0] = film_plane[0];
        }
        if (min_max[1] > film_plane[1]) {
            min_max[1] = film_plane[1];
        }
        if (min_max[2] < film_plane[0]) {
            min_max[2] = film_plane[0];
        }
        if (min_max[3] < film_plane[1]) {
            min_max[3] = film_plane[1];
        }
    }

    double rad_min_x_y_cam_coords[3] = { 0.0, 0.0, 0.0 };
    camera.coord_frame().inverse().point_to_world_space(&(camera.image_plane()[3]), rad_min_x_y_cam_coords);
    double rad_max_x_y_cam_coords[3] = { 0.0, 0.0, 0.0 };
    camera.coord_frame().inverse().point_to_world_space(&(camera.image_plane()[9]), rad_max_x_y_cam_coords);
    double rad_min_max_film[4]{
      -2 * rad_min_x_y_cam_coords[0] / rad_min_x_y_cam_coords[2],
      -2 * rad_min_x_y_cam_coords[1] / rad_min_x_y_cam_coords[2],
      -2 * rad_max_x_y_cam_coords[0] / rad_max_x_y_cam_coords[2],
      -2 * rad_max_x_y_cam_coords[1] / rad_max_x_y_cam_coords[2]
    };

    // Need to make sure that the bounding box falls within, at least part of, the rad image bounds
    if (min_max[0] > rad_min_max_film[2] && min_max[2] > rad_min_max_film[2]) {
      // This means that the min_max bbox is greater than the rad image bounds in the x direction
      return false;
     }
    if (min_max[1] > rad_min_max_film[3] && min_max[3] > rad_min_max_film[3]) {
      // This means that the min_max bbox is greater than the rad image bounds in the y direction
      return false;
    }
    if (min_max[0] < rad_min_max_film[0] && min_max[2] < rad_min_max_film[0]) {
      // This means that the min_max bbox is less than the rad image bounds in the x direction
      return false;
    }
    if (min_max[1] < rad_min_max_film[1] && min_max[3] < rad_min_max_film[1]) {
      // This means that the min_max bbox is less than the rad image bounds in the y direction
      return false;
    }

    // clip min_max to rad_min_max_film
    if (min_max[0] < rad_min_max_film[0]) {
        min_max[0] = rad_min_max_film[0];
    }
    if (min_max[1] < rad_min_max_film[1]) {
        min_max[1] = rad_min_max_film[1];
    }
    if (min_max[2] > rad_min_max_film[2]) {
        min_max[2] = rad_min_max_film[2];
    }
    if (min_max[3] > rad_min_max_film[3]) {
        min_max[3] = rad_min_max_film[3];
    }

    viewport[0] = min_max[0];
    viewport[1] = min_max[1];
    viewport[2] = min_max[2]-min_max[0];
    viewport[3] = min_max[3]-min_max[1];
    return true;
}


// Save Full DRR Image
void Tracker::getFullDRR(unsigned int volumeID) const
{
    double xyzypr[6] = { (*(const_cast<Trial&>(trial_)).getXCurve(-1))(trial_.frame),
        (*(const_cast<Trial&>(trial_)).getYCurve(-1))(trial_.frame),
        (*(const_cast<Trial&>(trial_)).getZCurve(-1))(trial_.frame),
        (*(const_cast<Trial&>(trial_)).getYawCurve(-1))(trial_.frame),
        (*(const_cast<Trial&>(trial_)).getPitchCurve(-1))(trial_.frame),
        (*(const_cast<Trial&>(trial_)).getRollCurve(-1))(trial_.frame) };

  CoordFrame xcframe = CoordFrame::from_xyzypr(xyzypr);

  for (unsigned int i = 0; i < views_.size(); ++i) {
    // Set the modelview matrix for DRR rendering
    CoordFrame modelview = views_[i]->camera()->coord_frame().inverse() * xcframe;
    double imv[16]; modelview.inverse().to_matrix_row_order(imv);
    views_[i]->drrRenderer(volumeID)->setInvModelView(imv);

    // Calculate the viewport surrounding the volume
    double viewport[4];
    //this->calculate_viewport(modelview, viewport);

    // Export Full Images
    viewport[0] = -views_[i]->camera()->viewport()[2]/2;
    viewport[1] = -views_[i]->camera()->viewport()[3]/2;
    viewport[2] = views_[i]->camera()->viewport()[2];
    viewport[3] = views_[i]->camera()->viewport()[3];

    // Calculate the size of the image to render
    unsigned render_width = viewport[2] * trial_.render_width / views_[i]->camera()->viewport()[2];
    unsigned render_height = viewport[3] * trial_.render_height / views_[i]->camera()->viewport()[3];

    // Set the viewports
    views_[i]->drrRenderer(volumeID)->setViewport(viewport[0], viewport[1],
      viewport[2], viewport[3]);
    views_[i]->radRenderer()->set_viewport(viewport[0], viewport[1],
      viewport[2], viewport[3]);

    // Render the DRR and Radiograph
    views_[i]->renderDrrSingle(volumeID, rendered_drr_, render_width, render_height);
    views_[i]->renderRad(rendered_rad_, render_width, render_height);

    //render masks
    views_[i]->backgroundRenderer()->set_viewport(viewport[0], viewport[1],
      viewport[2], viewport[3]);

    views_[i]->renderBackground(background_mask_, render_width, render_height);
    views_[i]->renderDRRMask(rendered_drr_, drr_mask_, render_width, render_height);

    gpu::multiply(background_mask_, drr_mask_, drr_mask_, render_width, render_height);
    gpu::multiply(rendered_rad_, drr_mask_, rendered_rad_, render_width, render_height);
    gpu::multiply(rendered_drr_, drr_mask_, rendered_drr_, render_width, render_height);

    save_full_drr(rendered_drr_, render_width, render_height);
  }
}





} // namespace XROMM
