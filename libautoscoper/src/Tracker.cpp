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
	tracker_cost_function = 0; // initialize cost function
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

void Tracker::optimize(int frame, int dFrame, int repeats, double nm_opt_alpha, double nm_opt_gamma, double nm_opt_beta, int cost_function_index)
{

	tracker_cost_function = cost_function_index;

    if (frame < 0 || frame >= trial_.num_frames) {
        cerr << "Tracker::optimize(): Invalid frame." << endl;
        return;
    }

    int NDIM = 6;       // Number of dimensions to optimize over.
    double FTOL = 1e-6; // Tolerance for the optimization.
    MAT P;              // Matrix of points to initialize the routine.
    double Y[MP];       // The values of the minimization function at the
                        // initial points.
    int ITER;

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

        // Generate the 7 vertices that form the initial simplex. Because
        // the independent variables of the function we are optimizing over
        // are relative to the initial guess, the same vertices can be used
        // to form the initial simplex for every frame.
        for (int i = 0; i < 7; ++i) {
            P[i+1][1] = (i == 1)? trial_.offsets[0]: 0.0;
            P[i+1][2] = (i == 2)? trial_.offsets[1]: 0.0;
            P[i+1][3] = (i == 3)? trial_.offsets[2]: 0.0;
            P[i+1][4] = (i == 4)? trial_.offsets[3]: 0.0;
            P[i+1][5] = (i == 5)? trial_.offsets[4]: 0.0;
            P[i+1][6] = (i == 6)? trial_.offsets[5]: 0.0;
        }

        // Determine the function values at the vertices of the initial
        // simplex
        for (int i = 0; i < 7; ++i) {
            Y[i+1] = FUNC(P[i+1]);
        }

        // Optimize the frame
        ITER = 0;

		// Downhill Simplex Optimization
        AMOEBA(P, Y, NDIM, FTOL, &ITER, nm_opt_alpha, nm_opt_gamma, nm_opt_beta);


		// Get Current Pose
		double xyzypr[6] = { (*trial_.getXCurve(-1))(trial_.frame),
			(*trial_.getYCurve(-1))(trial_.frame),
			(*trial_.getZCurve(-1))(trial_.frame),
			(*trial_.getYawCurve(-1))(trial_.frame),
			(*trial_.getPitchCurve(-1))(trial_.frame),
			(*trial_.getRollCurve(-1))(trial_.frame) };



		// My Try for Simulated Annealing
		//double MAX_TEMP = 50;
		//double MAX_ITER = 1;
		// Writing Simulated Annealing Code Here
		double x = SA_fRand(-30, 30);
		double y = SA_fRand(-30, 30);
		double xm = x, ym = y;
		double tI = 100000;
		double tF = 0.000001;
		double a = 0.99;
		double d = 1e-5;// (1.6*(pow(10, -23)));
		double T = tI;
		double minim = SA_func(x,y);
		double z;
		double counter = 0;

		while (T > tF) {
			int i = 1;
			while (i <= 30) {
				x = x + SA_fRand(-0.5, 0.5);
				y = y + SA_fRand(-0.5, 0.5);
				z = SA_func(x, y);
				if (z < minim || (SA_accept(z, minim, T, d) > (SA_fRand(0, 1)))) {
					minim = z;
					xm = x;
					ym = y;
				}
				i = i + 1;
			}
			counter = counter + 1;
			T = T * a;
			x = xm;
			y = ym;
		}

		cout << "min: " << minim << " x: " << xm << " y: " << ym << endl;

		// SA End


		// Convert Current Pose to its Coordinate System Frame
        CoordFrame xcframe = CoordFrame::from_xyzypr(xyzypr);


		CoordFrame manip = CoordFrame::from_xyzAxis_angle(P[1] + 1);
		xcframe = xcframe * trial_.getVolumeMatrix(-1)->inverse() * manip * *trial_.getVolumeMatrix(-1);
        xcframe.to_xyzypr(xyzypr);

		xcframe = xcframe * trial_.getVolumeMatrix(-1)->inverse() * manip * *trial_.getVolumeMatrix(-1);


		trial_.getXCurve(-1)->insert(trial_.frame, xyzypr[0]);
		trial_.getYCurve(-1)->insert(trial_.frame, xyzypr[1]);
		trial_.getZCurve(-1)->insert(trial_.frame, xyzypr[2]);
		trial_.getYawCurve(-1)->insert(trial_.frame, xyzypr[3]);
		trial_.getPitchCurve(-1)->insert(trial_.frame, xyzypr[4]);
		trial_.getRollCurve(-1)->insert(trial_.frame, xyzypr[5]);

        totalIter += ITER;
    }

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
			if (tracker_cost_function)
			{
				// Calculate the correlation for implant
				// Calculate Hausdorff Distance for Implant Matching _ FUTURE
				correlations.push_back(1.0 - gpu::hdist(rendered_drr_, rendered_rad_, drr_mask_, render_width*render_height));
			}
			else {
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
	printf("Cam 0: %4.5f", correlation);
	for (unsigned int i = 1; i < trial_.cameras.size(); ++i) {
		correlation *= correlations[i];
		printf("\tCam %d: %4.5f", i, correlations[i]);
	}
	printf("\tFinal NCC: %4.5f\n", correlation);

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


double Tracker::SA_accept(double z, double minim, double T, double d)
{
	double p = -(z - minim) / (d * T);
	return pow(exp(1), p);
}

double Tracker::SA_fRand(double fMin, double fMax)
{
	double f = (double)rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}

double Tracker::SA_func(double x, double y)
{
	return (pow(x - 2, 2) + pow(y - 1, 2));
}

} // namespace XROMM


