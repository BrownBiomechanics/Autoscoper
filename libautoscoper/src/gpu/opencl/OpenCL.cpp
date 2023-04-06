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

/// \file OpenCL.cpp
/// \author Mark Howison, Benjamin Knorlein

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>      // std::stringstream, std::stringbuf

#include "OpenCL.hpp"
#include "Backtrace.hpp"

/* OpenCL-OpenGL interoperability */
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenGL/CGLDevice.h>
#include <OpenCL/cl_gl_ext.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <CL/cl_gl.h>
static clGetGLContextInfoKHR_fn pfn_clGetGLContextInfoKHR;
#else
#include <GL/glx.h>
#include <CL/cl_gl.h>
static clGetGLContextInfoKHR_fn pfn_clGetGLContextInfoKHR;
#endif

#define TYPE CL_DEVICE_TYPE_GPU

#define ERROR(msg) do{\
  cerr << "Error at " << __FILE__ << ':' << __LINE__ \
       << "\n  " << msg << endl; \
  xromm::bt(); \
  exit(1); \
  }while(0)

#define CHECK_CL \
  if (err_ != CL_SUCCESS) {\
    cerr << "OpenCL error at " << __FILE__ << ':' << __LINE__ \
           << "\n  " << err_ << ' ' << opencl_error(err_) << endl; \
    xromm::bt(); \
    exit(1); \
  }



using namespace std;

static bool inited_ = false;
static bool gl_inited_ = false;
static int used_platform = 0;
static int used_device = 0;
std::vector <std::pair<int,int> > platform_device_keys;

#if defined(__APPLE__) || defined(__MACOSX)
static CGLShareGroupObj share_group_;
static CGLContextObj glContext;
#elif defined(_WIN32)
// TODO: implement this
#else
static GLXContext glx_context_;
static Display* glx_display_;
#endif

static cl_int err_;
static cl_context context_;
static cl_device_id devices_[10];
static cl_command_queue queue_;

static const char* opencl_error(cl_int err)
{
  switch (err) {
    case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
#ifdef CL_VERSION_1_2
    case CL_COMPILE_PROGRAM_FAILURE: return "CL_COMPILE_PROGRAM_FAILURE";
    case CL_LINKER_NOT_AVAILABLE: return "CL_LINKER_NOT_AVAILABLE";
    case CL_LINK_PROGRAM_FAILURE: return "CL_LINK_PROGRAM_FAILURE";
    case CL_DEVICE_PARTITION_FAILED: return "CL_DEVICE_PARTITION_FAILED";
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
#endif
    case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_INVALID_PROPERTY: return "CL_INVALID_PROPERTY";
#ifdef CL_VERSION_1_2
    case CL_INVALID_IMAGE_DESCRIPTOR: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case CL_INVALID_COMPILER_OPTIONS: return "CL_INVALID_COMPILER_OPTIONS";
    case CL_INVALID_LINKER_OPTIONS: return "CL_INVALID_LINKER_OPTIONS";
    case CL_INVALID_DEVICE_PARTITION_COUNT: return "CL_INVALID_DEVICE_PARTITION_COUNT";
#endif
#ifdef CL_VERSION_2_0
    case CL_INVALID_PIPE_SIZE: return "CL_INVALID_PIPE_SIZE";
    case CL_INVALID_DEVICE_QUEUE: return "CL_INVALID_DEVICE_QUEUE";
#endif
#ifdef CL_VERSION_2_2
    case CL_INVALID_SPEC_ID: return "CL_INVALID_SPEC_ID";
    case CL_MAX_SIZE_RESTRICTION_EXCEEDED: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
#endif
    default: return "Unknown";
    }
}

static void print_platform(cl_platform_id platform)
{
  cerr << "# OpenCL Platform" << endl;

  char buffer[1024];

  err_ = clGetPlatformInfo(
        platform, CL_PLATFORM_VERSION, sizeof(buffer), buffer, NULL);
  CHECK_CL
  cerr << "# Version    : " << buffer << endl;

  err_ = clGetPlatformInfo(
        platform, CL_PLATFORM_NAME, sizeof(buffer), buffer, NULL);
  CHECK_CL
  cerr << "# Name       : " << buffer << endl;

  err_ = clGetPlatformInfo(
        platform, CL_PLATFORM_VENDOR, sizeof(buffer), buffer, NULL);
  CHECK_CL
  cerr << "# Vendor     : " << buffer << endl;
}


static void print_device(cl_device_id device)
{
  char buffer[1024];
  cl_bool b;
  cl_device_type t;
  cl_ulong ul;
  cl_uint ui;
  size_t s[3];

  cerr << "# OpenCL Device" << "\n";

  err_ = clGetDeviceInfo(
        device, CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
  CHECK_CL
  cerr << "# Name          : " << buffer << "\n";

  err_ = clGetDeviceInfo(
        device, CL_DEVICE_TYPE, sizeof(t), &t, NULL);
  cerr << "# Type          : ";
  switch (t) {
    case CL_DEVICE_TYPE_CPU: cerr << "CPU\n"; break;
    case CL_DEVICE_TYPE_GPU: cerr << "GPU\n"; break;
    case CL_DEVICE_TYPE_ACCELERATOR: cerr << "Accelerator\n"; break;
    case CL_DEVICE_TYPE_DEFAULT: cerr << "Default\n"; break;
    default: cerr << "Unknown\n";
  }

  err_ = clGetDeviceInfo(
    device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(ui), &ui, NULL);
  CHECK_CL
  cerr << "# Compute Cores : " << ui << "\n";

  err_ = clGetDeviceInfo(
    device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(ui), &ui, NULL);
  CHECK_CL
  cerr << "# Core Freq.    : " << ui << " Mhz\n";

  err_ = clGetDeviceInfo(
    device, CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL);
  CHECK_CL
  cerr << "# Vendor        : " << buffer << "\n";

  err_ = clGetDeviceInfo(
    device, CL_DEVICE_VENDOR_ID, sizeof(ui), &ui, NULL);
  CHECK_CL
  cerr << "# Vendor ID     : " << ui << "\n";

  err_ = clGetDeviceInfo(
    device, CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL);
  CHECK_CL
  cerr << "# Version       : " << buffer << "\n";

  err_ = clGetDeviceInfo(
    device, CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL);
  CHECK_CL
  cerr << "# Driver Ver.   : " << buffer << "\n";

  err_ = clGetDeviceInfo(
    device, CL_DEVICE_AVAILABLE, sizeof(b), &b, NULL);
  CHECK_CL
  cerr << "# Available     : ";
  switch (b) {
    case CL_TRUE: cerr << "Yes\n"; break;
    case CL_FALSE: cerr << "No\n"; break;
    default: cerr << "Unknown\n";
  }

  err_ = clGetDeviceInfo(
    device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(s), s, NULL);
  CHECK_CL
  cerr << "# Max Items     : ("
     << s[0] << ',' << s[1] << ',' << s[2] << ")\n";

  err_ = clGetDeviceInfo(
    device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), s, NULL);
  CHECK_CL
  cerr << "# Max Group     : " << s[0] << "\n";

  err_ = clGetDeviceInfo(
    device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(ul), &ul, NULL);
  CHECK_CL
  cerr << "# Max Constant  : " << ul << " kB\n";

  err_ = clGetDeviceInfo(
    device, CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(ui), &ui, NULL);
  CHECK_CL
  cerr << "# Max Constants : " << ui << "\n";

  err_ = clGetDeviceInfo(
    device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(ul), &ul, NULL);
  CHECK_CL
  cerr << "# Local Mem.    : " << (ul/1024) << " kB\n";

  err_ = clGetDeviceInfo(
    device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(ul), &ul, NULL);
  CHECK_CL
  cerr << "# Global Mem.   : " << (ul/1024) << " kB\n";

  err_ = clGetDeviceInfo(
    device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(ul), &ul, NULL);
  CHECK_CL
  cerr << "# Global Cache  : " << ul << " B\n";

  err_ = clGetDeviceInfo(
    device, CL_DEVICE_IMAGE_SUPPORT, sizeof(b), &b, NULL);
  CHECK_CL
  cerr << "# Image Support : " << b << "\n";

  err_ = clGetDeviceInfo(
    device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), s+0, NULL);
  CHECK_CL
  err_ = clGetDeviceInfo(
    device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), s+1, NULL);
  CHECK_CL
  cerr << "# Max 2D Image  : (" << s[0] << ',' << s[1] << ")\n";

  err_ = clGetDeviceInfo(
    device, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(size_t), s+0, NULL);
  CHECK_CL
  err_ = clGetDeviceInfo(
    device, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(size_t), s+1, NULL);
  CHECK_CL
  err_ = clGetDeviceInfo(
    device, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(size_t), s+2, NULL);
  CHECK_CL
  cerr << "# Max 3D Image  : ("

     << s[0] << ',' << s[1] << ',' << s[2] << ")\n";

  err_ = clGetDeviceInfo(
    device, CL_DEVICE_EXTENSIONS, sizeof(buffer), buffer, NULL);
  CHECK_CL
  cerr << "# Extensions    :  "  << buffer << " \n";
}

namespace xromm { namespace gpu {


std::vector< std::vector<std::string> > get_platforms(){
  cl_uint num_platforms;
  cl_platform_id platforms[10];
  err_ = clGetPlatformIDs(10, platforms, &num_platforms);
  CHECK_CL

  if (num_platforms < 1) ERROR("no OpenCL platforms found");

  std::vector< std::vector<std::string> > platforms_desc;
  char buffer[1024];

  for (int i = 0 ; i < num_platforms; i ++){
    cl_uint num_devices;
    err_ = clGetDeviceIDs(platforms[i], TYPE, 1, devices_, &num_devices);
    for(int d = 0 ; d <num_devices; d++){
      bool isvalid = true;

      std::vector<std::string> platform_desc;

      std::string description = "";
      err_ = clGetDeviceInfo(devices_[d], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
      description.append(buffer);
      err_ = clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(buffer), buffer, NULL);
      description.append(" - ");
      description.append(buffer);
      platform_desc.push_back(description);

      /* find GPU device */
      err_ = clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(buffer), buffer, NULL);
      isvalid = (err_ == CL_SUCCESS) ? isvalid : false;
      std::string version = "# Version    : ";
      version.append(buffer);
      platform_desc.push_back(version);

      err_ = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(buffer), buffer, NULL);
      isvalid = (err_ == CL_SUCCESS) ? isvalid : false;
      std::string name = "# Name    : ";
      name.append(buffer);
      platform_desc.push_back(name);

      err_ = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(buffer), buffer, NULL);
      isvalid = (err_ == CL_SUCCESS) ? isvalid : false;
      std::string vendor = "# Vendor    : ";
      vendor.append(buffer);
      platform_desc.push_back(vendor);

      char buffer[1024];
      cl_bool b;
      cl_device_type t;
      cl_ulong ul;
      cl_uint ui;
      size_t s[3];
      stringstream ss;

      ss << "# OpenCL Device";
      platform_desc.push_back(ss.str());
      ss.str("");
      ss.clear(); // Clear state flags.

      err_ = clGetDeviceInfo(devices_[d], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
      isvalid = (err_ == CL_SUCCESS) ? isvalid : false;
      ss << "# Name          : " << buffer;
      platform_desc.push_back(ss.str());
      ss.str("");
      ss.clear(); // Clear state flags.

      err_ = clGetDeviceInfo(devices_[d], CL_DEVICE_TYPE, sizeof(t), &t, NULL);
      ss << "# Type          : ";
      switch (t) {
        case CL_DEVICE_TYPE_CPU: ss << "CPU"; break;
        case CL_DEVICE_TYPE_GPU: ss << "GPU"; break;
        case CL_DEVICE_TYPE_ACCELERATOR: ss << "Accelerator"; break;
        case CL_DEVICE_TYPE_DEFAULT: ss << "Default"; break;
        default: ss << "Unknown";
      }
      platform_desc.push_back(ss.str());
      ss.str("");
      ss.clear(); // Clear state flags.

      err_ = clGetDeviceInfo(devices_[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(ui), &ui, NULL);
      isvalid = (err_ == CL_SUCCESS) ? isvalid : false;
      ss << "# Compute Cores : " << ui;
      platform_desc.push_back(ss.str());
      ss.str("");
      ss.clear(); // Clear state flags.

      err_ = clGetDeviceInfo(devices_[d], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(ui), &ui, NULL);
      isvalid = (err_ == CL_SUCCESS) ? isvalid : false;
      ss << "# Core Freq.    : " << ui << " Mhz";
      platform_desc.push_back(ss.str());
      ss.str("");
      ss.clear(); // Clear state flags.

      err_ = clGetDeviceInfo(devices_[d], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL);
      isvalid = (err_ == CL_SUCCESS) ? isvalid : false;
      ss << "# Vendor        : " << buffer;
      platform_desc.push_back(ss.str());
      ss.str("");
      ss.clear(); // Clear state flags.

      err_ = clGetDeviceInfo(devices_[d], CL_DEVICE_VENDOR_ID, sizeof(ui), &ui, NULL);
      isvalid = (err_ == CL_SUCCESS) ? isvalid : false;
      ss << "# Vendor ID     : " << ui;
      platform_desc.push_back(ss.str());
      ss.str("");
      ss.clear(); // Clear state flags.

      err_ = clGetDeviceInfo(devices_[d], CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL);
      isvalid = (err_ == CL_SUCCESS) ? isvalid : false;
      ss << "# Version       : " << buffer;
      platform_desc.push_back(ss.str());
      ss.str("");
      ss.clear(); // Clear state flags.

      err_ = clGetDeviceInfo(devices_[d], CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL);
      isvalid = (err_ == CL_SUCCESS) ? isvalid : false;
      ss << "# Driver Ver.   : " << buffer;
      platform_desc.push_back(ss.str());
      ss.str("");
      ss.clear(); // Clear state flags.

      err_ = clGetDeviceInfo(devices_[d], CL_DEVICE_AVAILABLE, sizeof(b), &b, NULL);
      isvalid = (err_ == CL_SUCCESS) ? isvalid : false;
      ss << "# Available     : ";
      switch (b) {
        case CL_TRUE: ss << "Yes"; break;
        case CL_FALSE: ss << "No"; break;
        default: ss << "Unknown";
      }
      platform_desc.push_back(ss.str());
      ss.str("");
      ss.clear(); // Clear state flags.

      err_ = clGetDeviceInfo(devices_[d], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(s), s, NULL);
      isvalid = (err_ == CL_SUCCESS) ? isvalid : false;
      ss << "# Max Items     : ("<< s[0] << ',' << s[1] << ',' << s[2] << ")";
      platform_desc.push_back(ss.str());
      ss.str("");
      ss.clear(); // Clear state flags.

      err_ = clGetDeviceInfo(devices_[d], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), s, NULL);
      isvalid = (err_ == CL_SUCCESS) ? isvalid : false;
      ss << "# Max Group     : " << s[0];
      platform_desc.push_back(ss.str());
      ss.str("");
      ss.clear(); // Clear state flags.

      err_ = clGetDeviceInfo(devices_[d], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(ul), &ul, NULL);
      isvalid = (err_ == CL_SUCCESS) ? isvalid : false;
      ss << "# Max Constant  : " << ul << " kB";
      platform_desc.push_back(ss.str());
      ss.str("");
      ss.clear(); // Clear state flags.;

      err_ = clGetDeviceInfo(devices_[d], CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(ui), &ui, NULL);
      isvalid = (err_ == CL_SUCCESS) ? isvalid : false;
      ss << "# Max Constants : " << ui;
      platform_desc.push_back(ss.str());
      ss.str("");
      ss.clear(); // Clear state flags.

      err_ = clGetDeviceInfo(devices_[d], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(ul), &ul, NULL);
      isvalid = (err_ == CL_SUCCESS) ? isvalid : false;
      ss << "# Local Mem.    : " << (ul/1024) << " kB";
      platform_desc.push_back(ss.str());
      ss.str("");
      ss.clear(); // Clear state flags.

      err_ = clGetDeviceInfo(devices_[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(ul), &ul, NULL);
      isvalid = (err_ == CL_SUCCESS) ? isvalid : false;
      ss << "# Global Mem.   : " << (ul/1024) << " kB";
      platform_desc.push_back(ss.str());
      ss.str("");
      ss.clear(); // Clear state flags.

      err_ = clGetDeviceInfo(devices_[d], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(ul), &ul, NULL);
      isvalid = (err_ == CL_SUCCESS) ? isvalid : false;
      ss << "# Global Cache  : " << ul << " B";
      platform_desc.push_back(ss.str());
      ss.str("");
      ss.clear(); // Clear state flags.

      err_ = clGetDeviceInfo(devices_[d], CL_DEVICE_IMAGE_SUPPORT, sizeof(b), &b, NULL);
      isvalid = (err_ == CL_SUCCESS) ? isvalid : false;
      ss << "# Image Support : " << b;
      platform_desc.push_back(ss.str());
      ss.str("");
      ss.clear(); // Clear state flags.

      err_ = clGetDeviceInfo(devices_[d], CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), s+0, NULL);
      isvalid = (err_ == CL_SUCCESS) ? isvalid : false;
      err_ = clGetDeviceInfo(devices_[d], CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), s+1, NULL);
      isvalid = (err_ == CL_SUCCESS) ? isvalid : false;
      ss << "# Max 2D Image  : (" << s[0] << ',' << s[1] << ")";
      platform_desc.push_back(ss.str());
      ss.str("");
      ss.clear(); // Clear state flags.

      err_ = clGetDeviceInfo(devices_[d], CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(size_t), s+0, NULL);
      isvalid = (err_ == CL_SUCCESS) ? isvalid : false;
      err_ = clGetDeviceInfo(devices_[d], CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(size_t), s+1, NULL);
      isvalid = (err_ == CL_SUCCESS) ? isvalid : false;
      err_ = clGetDeviceInfo(devices_[d], CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(size_t), s+2, NULL);
      isvalid = (err_ == CL_SUCCESS) ? isvalid : false;
      ss << "# Max 3D Image  : (" << s[0] << ',' << s[1] << ',' << s[2] << ")";
      platform_desc.push_back(ss.str());
      ss.str("");
      ss.clear(); // Clear state flags.

      err_ = clGetDeviceInfo(devices_[d], CL_DEVICE_EXTENSIONS, sizeof(buffer), buffer, NULL);
      isvalid = (err_ == CL_SUCCESS) ? isvalid : false;
      ss << "# Extensions    :  "  << buffer;
      platform_desc.push_back(ss.str());
      ss.str("");
      ss.clear(); // Clear state flags.

      if(isvalid){
        platforms_desc.push_back(platform_desc);
        platform_device_keys.push_back(std::make_pair(i,d));
      }
    }
  }
  return platforms_desc;
}

void setUsedPlatform(int platform_idx){
  used_platform = platform_device_keys[platform_idx].first;
  used_device = platform_device_keys[platform_idx].second;
}

void setUsedPlatform(int platform, int device){
  used_platform = platform;
  used_device = device;
}

std::pair<int,int> getUsedPlatform(){
  return std::make_pair(used_platform,used_device);
}

void opencl_global_gl_context()
{
#if defined(__APPLE__) || defined(__MACOSX)
  glContext = CGLGetCurrentContext();
  share_group_ = CGLGetShareGroup(glContext);
  if (!share_group_) ERROR("invalid CGL sharegroup");
#elif defined(_WIN32)
#else
  glx_context_ = glXGetCurrentContext();
  if (!glx_context_) ERROR("invalid GLX context");
  glx_display_ = glXGetCurrentDisplay();
  if (!glx_display_) ERROR("invalid GLX display");
#endif
  gl_inited_ = true;
}

cl_int opencl_global_context()
{
  if (!inited_)
  {
    if (!gl_inited_) return CL_INVALID_CONTEXT;

    /* find platform */

    cl_uint num_platforms;
    cl_platform_id platforms[10];
    err_ = clGetPlatformIDs(10, platforms, &num_platforms);
    CHECK_CL
    if (num_platforms < used_platform) used_platform = 0;
    if (num_platforms < 1) ERROR("no OpenCL platforms found");

    /* create context */

#if defined(__APPLE__) || defined(__MACOSX)
#pragma OPENCL EXTENSION cl_APPLE_gl_sharing : enable
    cl_context_properties prop[] = {
      CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (intptr_t)share_group_,
      CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[used_platform]),
      0 };

    /* omit the device, according to this:
       http://www.khronos.org/message_boards/viewtopic.php?f=28&t=2548 */
    context_ = clCreateContext(prop, 0, NULL, 0, 0, &err_);
    CHECK_CL

    size_t num_gl_devices;
    err_ = clGetGLContextInfoAPPLE(
          context_, glContext   ,
                    CL_CGL_DEVICE_FOR_CURRENT_VIRTUAL_SCREEN_APPLE,
          sizeof(devices_), devices_, &num_gl_devices);
        int  _count = num_gl_devices / sizeof(cl_device_id);
    if(used_device >= _count) used_device = 0;
    CHECK_CL

#elif defined(_WIN32)
#pragma OPENCL EXTENSION cl_khr_gl_sharing : enable
    /* TODO: test this */
    cl_context_properties prop[] = {
      CL_GL_CONTEXT_KHR, (cl_context_properties) wglGetCurrentContext(),
      CL_WGL_HDC_KHR, (cl_context_properties) wglGetCurrentDC(),
      CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[used_platform]),
      0 };

      if (!pfn_clGetGLContextInfoKHR)
      {
#ifdef CL_VERSION_1_2
        pfn_clGetGLContextInfoKHR = (clGetGLContextInfoKHR_fn)clGetExtensionFunctionAddressForPlatform(platforms[used_platform], "clGetGLContextInfoKHR");
#else
        pfn_clGetGLContextInfoKHR = (clGetGLContextInfoKHR_fn)clGetExtensionFunctionAddress("clGetGLContextInfoKHR");
#endif
        if (!pfn_clGetGLContextInfoKHR)
        {
           std::cout << "Failed to query proc address for clGetGLContextInfoKHR." << std::endl;
          exit(EXIT_FAILURE);
        }
      }
      size_t size;
      pfn_clGetGLContextInfoKHR(prop, CL_DEVICES_FOR_GL_CONTEXT_KHR, 10 * sizeof(cl_device_id), devices_, &size);
      // Create a context using the supported devices
      int _count = size / sizeof(cl_device_id);
      if(used_device >= _count) used_device = 0;
      //fprintf(stderr,"%d Devices \n",_count);
      context_ = clCreateContext(prop, 1, &devices_[used_device], NULL, NULL, &err_);
      CHECK_CL
#else
#pragma OPENCL EXTENSION cl_khr_gl_sharing : enable
    cl_context_properties prop[] = {
      CL_GL_CONTEXT_KHR,
      (cl_context_properties)glXGetCurrentContext(),
      CL_GLX_DISPLAY_KHR,
      (cl_context_properties)glXGetCurrentDisplay(),
      CL_CONTEXT_PLATFORM,
      (cl_context_properties)(platforms[used_platform]),
      0 };

      if (!pfn_clGetGLContextInfoKHR)
      {
#ifdef CL_VERSION_1_2
        pfn_clGetGLContextInfoKHR = (clGetGLContextInfoKHR_fn)clGetExtensionFunctionAddressForPlatform(platforms[used_platform], "clGetGLContextInfoKHR");
#else
        pfn_clGetGLContextInfoKHR = (clGetGLContextInfoKHR_fn)clGetExtensionFunctionAddress("clGetGLContextInfoKHR");
#endif
        if (!pfn_clGetGLContextInfoKHR)
        {
          std::cout << "Failed to query proc address for clGetGLContextInfoKHR." << std::endl;
          exit(EXIT_FAILURE);
        }
      }

      size_t size;
      pfn_clGetGLContextInfoKHR(prop, CL_DEVICES_FOR_GL_CONTEXT_KHR, 10 * sizeof(cl_device_id), devices_, &size);

      int _count = size / sizeof(cl_device_id);
      if(used_device >= _count) used_device = 0;

      context_ = clCreateContext(prop, 1, &devices_[used_device], NULL, NULL, &err_);
    CHECK_CL
#endif

    /* create command queue */

    queue_ = clCreateCommandQueue(context_, devices_[used_device], 0, &err_);
    CHECK_CL

    inited_ = true;
  }
  return CL_SUCCESS;
}

Kernel::Kernel(cl_program program, const char* func)
{
  err_ = opencl_global_context();
  CHECK_CL
  reset();
  kernel_ = clCreateKernel(program, func, &err_);
  CHECK_CL
}
Kernel::~Kernel() {
    err_ = clReleaseKernel(kernel_);
    CHECK_CL
}

void Kernel::reset()
{
  arg_index_ = 0;
  grid_dim_ = 0;
  block_dim_ = 0;
}

size_t Kernel::getLocalMemSize()
{
  err_ = opencl_global_context();
  CHECK_CL
  cl_ulong s;
  err_ = clGetDeviceInfo(devices_[used_device],
          CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &s, NULL);
  CHECK_CL
  return s;
}

size_t* Kernel::getMaxItems()
{
  err_ = opencl_global_context();
  CHECK_CL
  size_t* s = new size_t[3];
  err_ = clGetDeviceInfo(devices_[used_device],
          CL_DEVICE_MAX_WORK_ITEM_SIZES, 3*sizeof(s), s, NULL);
  CHECK_CL
  return s;
}

size_t Kernel::getMaxGroup()
{
  err_ = opencl_global_context();
  CHECK_CL
  size_t s;
  err_ = clGetDeviceInfo(devices_[used_device],
          CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &s, NULL);
  CHECK_CL
  return s;
}

void Kernel::grid1d(size_t X)
{
  if (grid_dim_ && (grid_dim_ != 1)) {
    ERROR("Grid dimension was already set and is not 1");
  } else if (!block_dim_) {
    ERROR("Must set block dimension before grid");
  } else {
    grid_dim_ = 1;
  }
  grid_[0] = X * block_[0];
}

void Kernel::grid2d(size_t X, size_t Y)
{
  if (grid_dim_ && (grid_dim_ != 2)) {
    ERROR("Grid dimension was already set and is not 2");
  } else if (!block_dim_) {
    ERROR("Must set block dimension before grid");
  } else {
    grid_dim_ = 2;
  }
  grid_[0] = X * block_[0];
  grid_[1] = Y * block_[1];
}

void Kernel::grid3d(size_t X, size_t Y, size_t Z) {
  if (grid_dim_ && (grid_dim_ != 3)) {
	ERROR("Grid dimension was already set and is not 3");
  } else if (!block_dim_) {
	ERROR("Must set block dimension before grid");
  } else {
	grid_dim_ = 3;
  }
  grid_[0] = X * block_[0];
  grid_[1] = Y * block_[1];
  grid_[2] = Z * block_[2];
}

void Kernel::block1d(size_t X)
{
  if (block_dim_ && (block_dim_ != 1)) {
    ERROR("Block dimension was already set and is not 1");
  } else {
    block_dim_ = 1;
  }
  block_[0] = X;
}

void Kernel::block2d(size_t X, size_t Y)
{
  if (block_dim_ && (block_dim_ != 2)) {
    ERROR("Block dimension was already set and is not 2");
  } else {
    block_dim_ = 2;
  }
  block_[0] = X;
  block_[1] = Y;
}

void Kernel::block3d(size_t X, size_t Y, size_t Z)
{
  if (block_dim_ && (block_dim_ != 3)) {
    ERROR("Block dimension was already set and is not 3");
  } else {
	block_dim_ = 3;
  }
  block_[0] = X;
  block_[1] = Y;
  block_[2] = Z;
}

void Kernel::addBufferArg(const Buffer* buf)
{
  err_ = clSetKernelArg(kernel_, arg_index_++, sizeof(cl_mem), &buf->buffer_);
  CHECK_CL
}

void Kernel::addGLBufferArg(const GLBuffer* buf)
{
  gl_buffers.push_back(buf);
  err_ = clSetKernelArg(kernel_, arg_index_++, sizeof(cl_mem), &buf->buffer_);
  CHECK_CL
}

void Kernel::addImageArg(const Image* img)
{
  err_ = clSetKernelArg(kernel_, arg_index_++, sizeof(cl_mem), &img->image_);
  CHECK_CL
}

/* Dynamically allocated local memory can be added to the kernel by passing
   a NULL argument with the size of the buffer, e.g.
   http://stackoverflow.com/questions/8888718/how-to-declare-local-memory-in-opencl
*/
void Kernel::addLocalMem(size_t size)
{
  err_ = clSetKernelArg(kernel_, arg_index_++, size, NULL);
}

void Kernel::launch()
{
#if DEBUG
  cerr << "block:";
  for (unsigned i=0; i<grid_dim_; i++) cerr << ' ' << block_[i];
  cerr << endl;
  cerr << "grid:";
  for (unsigned i=0; i<grid_dim_; i++) cerr << ' ' << grid_[i];
  cerr << endl;
#endif
  if (!block_dim_) {
    ERROR("Block dimension is unset");
  } else if (!grid_dim_) {
    ERROR("Grid dimension is unset");
  } else if (block_dim_ != grid_dim_) {
    ERROR("Block dimension doesn't match grid dimension");
  }

  unsigned n_gl_buffers = gl_buffers.size();
  cl_mem* gl_mem = NULL;
  if (n_gl_buffers)
  {
    gl_mem = new cl_mem[n_gl_buffers];

    for (unsigned i=0; i<n_gl_buffers; i++) {
      gl_mem[i] = gl_buffers[i]->buffer_;
    }

    err_ = clEnqueueAcquireGLObjects(
        queue_, n_gl_buffers, gl_mem, 0, NULL, NULL);
    CHECK_CL
  }

  err_ = clEnqueueNDRangeKernel(
      queue_, kernel_, grid_dim_, NULL,
      grid_, block_, 0, NULL, NULL);
  CHECK_CL

  if (n_gl_buffers)
  {
    err_ = clEnqueueReleaseGLObjects(
        queue_, n_gl_buffers, gl_mem, 0, NULL, NULL);
    CHECK_CL

    delete gl_mem;
  }

  err_ = clFinish(queue_);
  CHECK_CL
}

void Kernel::setArg(cl_uint i, size_t size, const void* value)
{
  err_ = clSetKernelArg(kernel_, i, size, value);
  CHECK_CL
}

Program::Program() { compiled_ = false; }

Kernel* Program::compile(const char* code, const char* func)
{
  if (!compiled_)
  {
    err_ = opencl_global_context();
    CHECK_CL

    size_t len = strlen(code);
    program_ = clCreateProgramWithSource(context_, 1, &code, &len, &err_);
    CHECK_CL

    err_ = clBuildProgram(program_, 1, devices_, NULL, NULL, NULL);
    if (err_ == CL_BUILD_PROGRAM_FAILURE) {
      size_t log_size;
      err_ = clGetProgramBuildInfo(
          program_, devices_[0], CL_PROGRAM_BUILD_LOG,
          0, NULL, &log_size);
      CHECK_CL
      char* build_log = (char*)malloc(log_size+1);
      if (!build_log) ERROR("malloc for build log");
      err_ = clGetProgramBuildInfo(
          program_, devices_[0], CL_PROGRAM_BUILD_LOG,
          log_size, build_log, NULL);
      CHECK_CL
      build_log[log_size] = '\0';
      cerr << "OpenCL build failure for kernel function '" << func
           << "':\n" << build_log << endl;
      free(build_log);
      exit(1);
    } else {
      CHECK_CL
    }

    compiled_ = true;
  }

  return new Kernel(program_, func);
}

Buffer::Buffer(size_t size, cl_mem_flags access)
{
  err_ = opencl_global_context();
  CHECK_CL
  size_ = size;
  access_ = access;
  buffer_ = clCreateBuffer(context_, access, size, NULL, &err_);
  CHECK_CL
}

Buffer::~Buffer()
{
  err_ = clReleaseMemObject(buffer_);
  CHECK_CL
}

void Buffer::read(const void* buf, size_t size) const
{
  if (size == 0) size = size_;
  err_ = clEnqueueWriteBuffer(
      queue_, buffer_, CL_TRUE, 0, size, buf, 0, NULL, NULL);
  CHECK_CL
}

void Buffer::write(void* buf, size_t size) const
{
  if (size == 0) size = size_;
  err_ = clEnqueueReadBuffer(
      queue_, buffer_, CL_TRUE, 0, size, buf, 0, NULL, NULL);
  CHECK_CL
}

void Buffer::copy(const Buffer* dst, size_t size) const
{
  if (size == 0) size = size_;
  if (size > dst->size_)
    ERROR("Destination buffer does not have enough room!");
  err_ = clEnqueueCopyBuffer(
      queue_, buffer_, dst->buffer_, 0, 0, size, 0, NULL, NULL);
  CHECK_CL
}

void Buffer::fill(const char c) const
{
#ifdef CL_VERSION_1_2
  err_ = clEnqueueFillBuffer(queue_, buffer_, &c, 1, 0, size_, 0, NULL, NULL);
  CHECK_CL
#else
  char* tmp = (char*)new char[size_];
  memset(tmp, c, size_);
  err_ = clEnqueueWriteBuffer(
      queue_, buffer_, CL_TRUE, 0, size_, (void*)tmp, 0, NULL, NULL);
  CHECK_CL
  delete tmp;
#endif
}

void Buffer::fill(const float val) const
{
#ifdef CL_VERSION_1_2
  err_ = clEnqueueFillBuffer(queue_, buffer_, &val, sizeof(float), 0, size_, 0, NULL, NULL);
  CHECK_CL
#else
  float* tmp = (float*)new char[size_];
  memset(tmp, val, size_);
  err_ = clEnqueueWriteBuffer(
    queue_, buffer_, CL_TRUE, 0, size_, (void*)tmp, 0, NULL, NULL);
  CHECK_CL
    delete tmp;
#endif
}

GLBuffer::GLBuffer(cl_GLuint pbo, cl_mem_flags access)
{
  err_ = opencl_global_context();
  CHECK_CL
  access_ = access;
  buffer_ = clCreateFromGLBuffer(context_, access, pbo, &err_);
  CHECK_CL
}

GLBuffer::~GLBuffer()
{
  err_ = clReleaseMemObject(buffer_);
  CHECK_CL
}

Image::Image(size_t* dims, cl_image_format *format, cl_mem_flags access)
{
  err_ = opencl_global_context();
  CHECK_CL
  dims_[0] = dims[0];
  dims_[1] = dims[1];
  dims_[2] = dims[2];
  access_ = access;

#if DEBUG
  cerr << "OpenCL::Image dims (" << dims[0] << ',' << dims[1] << ','
     << dims[2] << ')' << endl;
#endif

  if (dims[0] == 0 || dims[1] == 0 || dims[2] == 0)
    ERROR("Image object must have non-zero dimensions");

#ifdef CL_VERSION_1_2
  cl_image_desc desc;

  if (dims[2] == 1) {
    desc.image_type  = CL_MEM_OBJECT_IMAGE2D;
    desc.image_depth = 0;
  } else {
    desc.image_type  = CL_MEM_OBJECT_IMAGE3D;
    desc.image_depth = dims[2];
  }

  desc.image_width       = dims[0];
  desc.image_height      = dims[1];
  desc.image_row_pitch   = 0;
  desc.image_slice_pitch = 0;
  desc.num_mip_levels    = 0;
  desc.num_samples       = 0;
  desc.buffer            = NULL;

  image_ = clCreateImage(context_, access, format, &desc, NULL, &err_);
#else
  if (dims[2] == 1) {
    image_ = clCreateImage2D(context_,
          access, format, dims[0], dims[1], 0, NULL, &err_);
  } else {
    image_ = clCreateImage3D(
          context_, access, format,
          dims[0], dims[1], dims[2],
          0, 0, NULL, &err_);
  }
#endif

  CHECK_CL
}

Image::~Image()
{
  err_ = clReleaseMemObject(image_);
  CHECK_CL
}

void Image::read(const void* buf) const
{
  size_t origin[3] = {0,0,0};
  err_ = clEnqueueWriteImage(
      queue_, image_, CL_TRUE, origin, dims_, 0, 0, buf, 0, NULL, NULL);
  CHECK_CL
}

void Image::write(void* buf) const
{
  size_t origin[3] = {0,0,0};
  err_ = clEnqueueReadImage(
      queue_, image_, CL_TRUE, origin, dims_, 0, 0, buf, 0, NULL, NULL);
  CHECK_CL
}

} } // namespace xromm::opencl
