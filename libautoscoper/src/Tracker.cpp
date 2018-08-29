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
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef WITH_CUDA
#include "gpu/cuda/CudaWrap.hpp"
#include "gpu/cuda/Ncc_kernels.h"
#include "gpu/cuda/HDist_kernels.h"
#include "gpu/cuda/Compositor_kernels.h"
#include "gpu/cuda/Mult_kernels.h"
#else
#include "gpu/opencl/Ncc.hpp"
#endif

#include "VolumeDescription.hpp"
#include "Video.hpp"
#include "View.hpp"
#include "DownhillSimplex.hpp"
#include "SimulatedAnnealing.hpp"
#include "Camera.hpp"
#include "CoordFrame.hpp"
#include <cuda_runtime_api.h>
#include "gpu/opencl/Mult.hpp"




using namespace std;

static bool firstRun = true;

#define DEBUG 0

// XXX
// Set callback for Downhill Simplex. This is really a hack so that we can use
// define the global function pointer FUNC, which Downhill SImplex will
// optimize.

static xromm::Tracker* g_markerless = NULL;
 
// Bardiya: I commented this out to play with the implant cost function
double FUNC(double* P) { return g_markerless->minimizationFunc(P+1); }


// For Particle Swarm Optimization
cParticle particles[MAX_PARTICLES];

namespace xromm {

#if DEBUG
#ifdef WITH_CUDA
	void save_debug_image(const Buffer* dev_image, int width, int height)
#else
	void save_debug_image(const gpu::Buffer* dev_image, int width, int height)
#endif
{
	static int count = 0;
	float* host_image = new float[width*height];
	unsigned char* uchar_image = new unsigned char[width*height];

#ifdef WITH_CUDA
	cudaMemcpy(host_image, dev_image, width*height*sizeof(float), cudaMemcpyDeviceToHost);
#else
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
	sprintf(filename,"pgm//image_%02d.pgm",count++);
	ofstream file(filename,ios::out);
	file << "P2" << endl;
	file << width << " " << height << endl;
	file << 255 << endl;
	for (int i = 0; i < width*height; i++) {
		file << (int)uchar_image[i] << " ";
	}

	delete[] uchar_image;
	delete[] host_image;
}
#endif

Tracker::Tracker()
    : rendered_drr_(NULL),
      rendered_rad_(NULL),
	  drr_mask_(NULL),
	  background_mask_(NULL)
{
    g_markerless = this;
	optimization_method = 0; // initialize cost function
	cf_model_select = 0;
}

Tracker::~Tracker()
{
	for (int i = 0; i < volumeDescription_.size(); i++){
		delete volumeDescription_[i];
	}
	volumeDescription_.clear();
}

void Tracker::init()
{
#ifdef WITH_CUDA
    gpu::cudaInitWrap();
#endif
}

void Tracker::load(const Trial& trial)
{
    trial_ = trial;

    vector<gpu::View*>::iterator viewIter;
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
#ifdef WITH_CUDA
	gpu::cudaMallocWrap(rendered_drr_,trial_.render_width*trial_.render_height*sizeof(float));
    gpu::cudaMallocWrap(rendered_rad_,trial_.render_width*trial_.render_height*sizeof(float));
	gpu::cudaMallocWrap(drr_mask_, trial_.render_width*trial_.render_height*sizeof(float));
	gpu::cudaMallocWrap(background_mask_, trial_.render_width*trial_.render_height*sizeof(float));
	gpu::fill(drr_mask_, trial_.render_width*trial_.render_height, 1.0f);
	gpu::fill(background_mask_, trial_.render_width*trial_.render_height, 1.0f);
#else
	rendered_drr_ = new gpu::Buffer(npixels*sizeof(float));
	rendered_rad_ = new gpu::Buffer(npixels*sizeof(float));
	drr_mask_ = new gpu::Buffer(npixels*sizeof(float));
	background_mask_ = new gpu::Buffer(npixels*sizeof(float));
	drr_mask_->fill(1.0f);
	background_mask_->fill(1.0f);
#endif

    gpu::ncc_init(npixels);
	gpu::hdist_init(npixels); 
	
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

void Tracker::optimize(int frame, int dFrame, int repeats, double nm_opt_alpha, double nm_opt_gamma, double nm_opt_beta, int opt_method, unsigned int inner_iter, double rot_limit, double trans_limit, int cf_model)
{

	optimization_method = opt_method;
	cf_model_select = cf_model;

    if (frame < 0 || frame >= trial_.num_frames) {
        cerr << "Tracker::optimize(): Invalid frame." << endl;
        return;
    }

    int NDIM = 6;       // Number of dimensions to optimize over.
    double FTOL = 1e-5; // Tolerance for the optimization.
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
		CoordFrame manip = CoordFrame::from_xyzAxis_angle(init_manip);

		if (optimization_method == 0) // HAVE TO CHANGE THIS TO ANOTHER RADIO BUTTON. NOW, IMPLANT MEANS DOWNHILL SIMPLEX
		{
			/*
			// My Try for Simulated Annealing
			double TEMP_INIT = 10;
			double TEMP_FINAL = 0.0001;
			double N_CYCLE = inner_iter;
			//double MAX_ITER = 20;
			double xyzypr_manip[6] = { 0 };
			double rot_lim_opt = rot_limit;
			double trans_lim_opt = trans_limit;
			double xym[6] = { xyzypr_manip[0], xyzypr_manip[1], xyzypr_manip[2] , xyzypr_manip[3] , xyzypr_manip[4] , xyzypr_manip[5] };
			double a = .9;// Reduce Temp with this
			double d = 10000;
			//double T = TEMP_INIT;
			double best_cost = minimizationFunc(xyzypr_manip);
			double new_cost = 99999;
			//while (T > TEMP_FINAL)
			for (double T = TEMP_INIT; T>TEMP_FINAL; T *= a) {
				int i = 1;
				while (i <= N_CYCLE) {
					xyzypr_manip[0] = xyzypr_manip[0] + SA_fRand(-trans_lim_opt, trans_lim_opt);
					xyzypr_manip[1] = xyzypr_manip[1] + SA_fRand(-trans_lim_opt, trans_lim_opt);
					xyzypr_manip[2] = xyzypr_manip[2] + SA_fRand(-trans_lim_opt, trans_lim_opt);
					xyzypr_manip[3] = xyzypr_manip[3] + SA_fRand(-rot_lim_opt, rot_lim_opt);
					xyzypr_manip[4] = xyzypr_manip[4] + SA_fRand(-rot_lim_opt, rot_lim_opt);
					xyzypr_manip[5] = xyzypr_manip[5] + SA_fRand(-rot_lim_opt, rot_lim_opt);

					new_cost = minimizationFunc(xyzypr_manip);
					//|| SA_accept(best_cost, new_cost, T, d) > SA_fRand(0, 1)
					if (new_cost <= best_cost ) {
						best_cost = new_cost;
						xym[0] = xyzypr_manip[0];
						xym[1] = xyzypr_manip[1];
						xym[2] = xyzypr_manip[2];
						xym[3] = xyzypr_manip[3];
						xym[4] = xyzypr_manip[4];
						xym[5] = xyzypr_manip[5];
					}
					i = i + 1;
					ITER += 1;
				}
				T = T * a;
				//DEBUGGING: cout << rot_lim_opt << " " << trans_lim_opt << endl;
				rot_lim_opt = rot_lim_opt;
				trans_lim_opt = trans_lim_opt; // Translation shrinks slower
				xyzypr_manip[0] = xym[0];
				xyzypr_manip[1] = xym[1];
				xyzypr_manip[2] = xym[2];
				xyzypr_manip[3] = xym[3];
				xyzypr_manip[4] = xym[4];
				xyzypr_manip[5] = xym[5];
			}*/
			// PSO Algorithm
			double xyzypr_manip[6] = { 0 };
			int gBest = 999;
			int gBestTest = 1000;
			int stall_iter = 0;
			bool done = false;
			int START_RANGE_MIN = rot_limit;
			int START_RANGE_MAX = trans_limit;
			int MAX_EPOCHS = inner_iter;
			initialize(START_RANGE_MIN, START_RANGE_MAX);

			do
			{
				/* Two conditions can end this loop:
				if the maximum number of epochs allowed has been reached, or,
				if the Target value has been found.
				*/
				if (ITER < MAX_EPOCHS) {

					for (int i = 0; i <= MAX_PARTICLES - 1; i++)
					{
						if (testProblem(i) == TARGET)
						{
							done = true;
						}
					} // i

					gBestTest = minimum();

					//cout << testProblem(gBestTest) << endl;

					// Check if we are stalled
					if (abs(testProblem(gBestTest) - testProblem(gBest)) < 1e-5) {
						stall_iter += 1;
					}
					if (stall_iter == 20) {
						done = true;
						cout << "Maximum Stall Iteration Reached..." << endl;
					}

					//If any particle's pBest value is better than the gBest value,
					//make it the new gBest Value.
					if (abs(TARGET - testProblem(gBestTest)) < abs(TARGET - testProblem(gBest)))
					{
						gBest = gBestTest;
					}

					getVelocity(gBest);

					updateParticles(gBest);

					ITER += 1;

				}
				else {
					done = true;
				}

			} while (!done);

			cout << ITER << " epochs completed." << endl;

			cout << "Best Case:" << endl;
			for (int j = 0; j <= MAX_INPUTS - 1; j++)
			{
				if (j < MAX_INPUTS - 1) {
					cout << particles[gBest].getData(j) << " , ";
				}
				else {
					cout << particles[gBest].getData(j) << " = ";
				}
			} // j

			//cout << testProblem(gBest) << endl;

			xyzypr_manip[0] = particles[gBest].getData(0);
			xyzypr_manip[1] = particles[gBest].getData(1);
			xyzypr_manip[2] = particles[gBest].getData(2);
			xyzypr_manip[3] = particles[gBest].getData(3);
			xyzypr_manip[4] = particles[gBest].getData(4);
			xyzypr_manip[5] = particles[gBest].getData(5);
			//

			cout << "Optimized Final NCC: " << testProblem(gBest) << endl;

			manip = CoordFrame::from_xyzAxis_angle(xyzypr_manip);
			// SA End

			// ADD DOWNHILL AT THE END:
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


			// Get Current Pose
			double xyzypr[6] = { (*trial_.getXCurve(-1))(trial_.frame),
				(*trial_.getYCurve(-1))(trial_.frame),
				(*trial_.getZCurve(-1))(trial_.frame),
				(*trial_.getYawCurve(-1))(trial_.frame),
				(*trial_.getPitchCurve(-1))(trial_.frame),
				(*trial_.getRollCurve(-1))(trial_.frame) };

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
			AMOEBA(P, Y, NDIM, FTOL, &ITER, nm_opt_alpha, nm_opt_gamma, nm_opt_beta);

			cout << "Optimized Final NCC: " << minimizationFunc((P[1] + 1)) << endl;

			// For Downhill Simplex Method
			manip = CoordFrame::from_xyzAxis_angle(P[1] + 1);


		}
		else {

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
			AMOEBA(P, Y, NDIM, FTOL, &ITER, nm_opt_alpha, nm_opt_gamma, nm_opt_beta);

			cout << "Optimized Final NCC: " << minimizationFunc((P[1] + 1)) << endl;

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

    cerr << "Tracker::optimize(): Frame " << trial_.frame
         << " done in " << totalIter << " total iterations" << endl;
}


// Calculate Correlation for Bone Matching
std::vector <double> Tracker::trackFrame(unsigned int volumeID, double* xyzypr) const
	{
		std::vector<double> correlations;
		CoordFrame xcframe = CoordFrame::from_xyzypr(xyzypr);

		for (unsigned int i = 0; i < views_.size(); ++i) {
			// Set the modelview matrix for DRR rendering
			CoordFrame modelview = views_[i]->camera()->coord_frame().inverse()*xcframe;
			double imv[16]; modelview.inverse().to_matrix_row_order(imv);
			views_[i]->drrRenderer(volumeID)->setInvModelView(imv);

			// Calculate the viewport surrounding the volume
			double viewport[4];
			this->calculate_viewport(modelview, viewport);

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

#if DEBUG
			save_debug_image(rendered_drr_, render_width, render_height);
			save_debug_image(rendered_rad_, render_width, render_height);
			save_debug_image(drr_mask_, render_width, render_height);
			save_debug_image(background_mask_, render_width, render_height);
#endif
			if (cf_model_select) // If 1, we do Implant
			{
				// Calculate the correlation for implant
				// Calculate Hausdorff Distance for Implant Matching _ FUTURE
				correlations.push_back(gpu::hdist(rendered_drr_, rendered_rad_, drr_mask_, render_width*render_height));

			}
			else { // If 0, we do bone model
				// Calculate the correlation for ncc
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
		correlation *= correlations[i];
	//	printf("\tCam %d: %4.5f", i, correlations[i]);
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


#ifdef WITH_CUDA
	void get_image(const Buffer* dev_image, int width, int height, std::vector<unsigned char> &data)
#else
	void get_image(const gpu::Buffer* dev_image, int width, int height, std::vector<unsigned char> &data)
#endif
	{
		static int count = 0;
		float* host_image = new float[width*height];

#ifdef WITH_CUDA
		cudaMemcpy(host_image, dev_image, width*height*sizeof(float), cudaMemcpyDeviceToHost);
#else
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
		this->calculate_viewport(modelview, viewport);

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

void Tracker::calculate_viewport(const CoordFrame& modelview,double* viewport) const
{
    // Calculate the minimum and maximum values of the bounding box
    // corners after they have been projected onto the view plane
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

    viewport[0] = min_max[0];
    viewport[1] = min_max[1];
    viewport[2] = min_max[2]-min_max[0];
    viewport[3] = min_max[3]-min_max[1];
}


double Tracker::SA_accept(double e, double ep, double T, double d)
{
	double p = exp(-(ep - e)* d/ T);
	cout << p << endl;
	return p;
}

double Tracker::SA_fRand(double fMin, double fMax)
{
	double f = (double)rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}




// Particle Swarm Optimization
/*void Tracker::psoAlgorithm()
{

	return;
}*/

void Tracker::initialize(int START_RANGE_MIN, int START_RANGE_MAX)
{

	double total;

	cout << "Initialized PSO with: " << MAX_PARTICLES << " Particles." << endl;
	for (int i = 0; i <= MAX_PARTICLES - 1; i++)
	{
		total = 0;

		for (int j = 0; j <= MAX_INPUTS - 1; j++)
		{
			particles[i].setData(j, getRandomNumber(START_RANGE_MIN, START_RANGE_MAX));
			if (i == 0) {
				particles[i].setData(j, 0);
			}

		}



		double manip_temp[6] = { 0 };
	//	cout << "First Init Point: ";
		for (int j = 0; j <= MAX_INPUTS - 1; j++)
		{
			manip_temp[j] = particles[i].getData(j);
			
			//cout << manip_temp[j] << ", ";
		} // i
	//	cout << endl;
		total = minimizationFunc(manip_temp);
	//	cout << "Check initialize: " << total << endl;
		particles[i].setpBest(total);

	} // i

	return;
}

void Tracker::getVelocity(int gBestIndex)
{
	/* from Kennedy & Eberhart(1995).
	vx[][] = vx[][] + 2 * rand() * (pbestx[][] - presentx[][]) +
	2 * rand() * (pbestx[][gbest] - presentx[][])
	*/
	int testResults, bestResults;
	float vValue;

	bestResults = testProblem(gBestIndex);

	for (int i = 0; i <= MAX_PARTICLES - 1; i++)
	{
		testResults = testProblem(i);
		vValue = particles[i].getVelocity() +
			2 * gRand() * (particles[i].getpBest() - testResults) + 2 * gRand() *
			(bestResults - testResults);

		// BA Addition
		vValue = 0.1;

		//cout << "For Particle #" << i << ", Velocity is: " << vValue << endl;
		if (vValue > V_MAX) {
			particles[i].setVelocity(V_MAX);
		}
		else if (vValue < -V_MAX) {
			particles[i].setVelocity(-V_MAX);
		}
		else if (vValue < 1e-3 & vValue > 0) {
			particles[i].setVelocity(0.005);
		}
		else if (vValue > -1e-3 & vValue < 0) {
			particles[i].setVelocity(-0.005);
		}
		else {
			particles[i].setVelocity(vValue);
		}
	} // i
}

void Tracker::updateParticles(int gBestIndex)
{
	double total;
	double tempData;

	for (int i = 0; i <= MAX_PARTICLES - 1; i++)
	{
		for (int j = 0; j <= MAX_INPUTS - 1; j++)
		{
			if (particles[i].getData(j) != particles[gBestIndex].getData(j))
			{
				tempData = particles[i].getData(j);
				particles[i].setData(j, tempData + static_cast<int>(particles[i].getVelocity()));
			}
		} // j

		  //Check pBest value.
		total = testProblem(i);
		if (abs(TARGET - total) < particles[i].getpBest())
		{
			particles[i].setpBest(total);
		}

	} // i

}

double Tracker::testProblem(int index)
{
	double xyzypr_manip[6] = { 0 };
	double total = 0;
	for (int i = 0; i <= MAX_INPUTS - 1; i++)
	{
		xyzypr_manip[i] = particles[index].getData(i);
	} // i
	//double x1 = particles[index].getData(0);
	//double x2 = particles[index].getData(1);
	total = minimizationFunc(xyzypr_manip);
	
	//cout << "Check testProblem function: " << total << endl;

	return total;
}

float Tracker::gRand()
{
	// Returns a pseudo-random float between 0.0 and 1.0
	return float(rand() / (RAND_MAX + 1.0));
}

double Tracker::getRandomNumber(int low, int high)
{
	// Returns a pseudo-random integer between low and high.
	double f = (double)rand() / RAND_MAX;
	return  low + (high - low) * f;
}

int Tracker::minimum()
{
	//Returns an array index.
	int winner = 0;
	bool foundNewWinner;
	bool done = false;

	do
	{
		foundNewWinner = false;
		for (int i = 0; i <= MAX_PARTICLES - 1; i++)
		{
			if (i != winner) {             //Avoid self-comparison.
										   //The minimum has to be in relation to the Target.
				if (abs(TARGET - testProblem(i)) < abs(TARGET - testProblem(winner)))
				{
					winner = i;
					foundNewWinner = true;
				}
			}
		} // i

		if (foundNewWinner == false)
		{
			done = true;
		}

	} while (!done);

	return winner;
}














} // namespace XROMM


