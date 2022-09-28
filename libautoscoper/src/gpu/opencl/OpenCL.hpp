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

/// \file OpenCL.h
/// \author Mark Howison, Benjamin Knorlein

#ifndef XROMM_HPP
#define XROMM_HPP

#include <iostream>
#include <vector>

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#include <OpenGL/OpenGL.h>
#else
#include <windows.h>
#include <CL/opencl.h>
#include <GL/gl.h>
#endif

namespace xromm { namespace gpu {

void opencl_global_gl_context();
cl_int opencl_global_context();
std::vector< std::vector<std::string> > get_platforms();
void setUsedPlatform(int platform_idx);
void setUsedPlatform(int platform, int device);
std::pair<int,int> getUsedPlatform();



class Buffer;
class GLBuffer;
class Image;

class Kernel
{
public:
  Kernel(cl_program program, const char* func);
  void reset();

  static size_t getLocalMemSize();
  static size_t* getMaxItems();
  static size_t getMaxGroup();

  void grid1d(size_t X);
  void block1d(size_t X);
  void grid2d(size_t X, size_t Y);
  void block2d(size_t X, size_t Y);
  void launch();

  void addBufferArg(const Buffer* buf);
  void addGLBufferArg(const GLBuffer* buf);
  void addImageArg(const Image* img);
  void addLocalMem(size_t size);

  template<typename T> void addArg(T& value)
  {
    setArg(arg_index_++, sizeof(T), (const void*)(&value));
  }

protected:
  void setArg(cl_uint i, size_t size, const void* value);
  cl_kernel kernel_;
  cl_uint arg_index_;
  size_t grid_[3];
  cl_uint grid_dim_;
  size_t block_[3];
  cl_uint block_dim_;
  std::vector<const GLBuffer*> gl_buffers;
};

class Program
{
public:
  Program();
    Kernel* compile(const char* code, const char* func);
protected:
  cl_program program_;
  bool compiled_;
};

class Buffer
{
public:
  Buffer(size_t size, cl_mem_flags access=CL_MEM_READ_WRITE);
  ~Buffer();

  void read(const void* buf, size_t size=0) const;
  void write(void* buf, size_t size=0) const;
  void copy(const Buffer* dst, size_t size=0) const;
  void fill(const char c) const;
  void fill(float val) const;
  friend class Kernel;

protected:
  size_t size_;
  cl_mem buffer_;
  cl_mem_flags access_;
};

class GLBuffer
{
public:
  GLBuffer(GLuint pbo, cl_mem_flags access=CL_MEM_READ_WRITE);
  ~GLBuffer();
  friend class Kernel;
protected:
  cl_mem buffer_;
  cl_mem_flags access_;
};

class Image
{
public:
  Image(size_t* dims, cl_image_format *format,
        cl_mem_flags access=CL_MEM_READ_WRITE);
  ~Image();

  void read(const void* buf) const;
  void write(void* buf) const;

  friend class Kernel;

protected:
  size_t dims_[3];
  cl_mem image_;
  cl_mem_flags access_;
};

} } // namespace xromm::opencl

#endif // XROMM_HPP
