#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageToVTKImageFilter.h>

constexpr unsigned int Dimension = 3;
using PixelType = unsigned char;
using ImageType = itk::Image<PixelType, Dimension>;

itk::SmartPointer<ImageType> loadtiffstackITK(std::string filename);
vtkImageData* itk2vtk(itk::SmartPointer<ImageType> image);