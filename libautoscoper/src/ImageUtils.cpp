#include "ImageUtils.hpp"

using namespace std;

itk::SmartPointer<ImageType> loadtiffstackITK(string filename) {


    auto image = itk::ReadImage<ImageType>(filename);
    image->Print(cout);
    return image;
}

vtkImageData* itk2vtk(itk::SmartPointer<ImageType> image) {
    using FilterType = itk::ImageToVTKImageFilter<ImageType>;
    auto filter = FilterType::New();
    filter->SetInput(image);
    try
    {
        filter->Update();
    }
    catch (const itk::ExceptionObject& error)
    {
        cerr << "Error: " << error << endl;
    }

    vtkImageData* myvtkImageData = filter->GetOutput();
    //myvtkImageData->Print(cout);
    return myvtkImageData;
}