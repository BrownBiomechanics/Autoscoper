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
int getUsedPlatform();

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
