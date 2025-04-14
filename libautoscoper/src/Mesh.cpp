#include "Mesh.hpp"
#include <vtkSTLReader.h>
#include <vtkSTLWriter.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>

Mesh::Mesh(const std::string& filename)
{
  this->filename = filename;

  vtkNew<vtkSTLReader> reader;
  this->polyData = vtkPolyData::New();

  reader->SetFileName(filename.c_str());
  reader->Update();
  this->polyData = reader->GetOutput();

  // Register so it exists after deleting the reader
  this->polyData->Register(nullptr);

  // Compute bounding radius
  double centerA[3];
  this->polyData->GetCenter(centerA);

  double bounds[6];
  this->polyData->GetBounds(bounds);

  boundingRadius = (bounds[1] - centerA[0]) * (bounds[1] - centerA[0])
                   + (bounds[3] - centerA[1]) * (bounds[3] - centerA[1])
                   + (bounds[5] - centerA[2]) * (bounds[5] - centerA[2]);

  boundingRadius = sqrt(boundingRadius);

  std::cout << "Mesh " << filename << ", has a bounding radius of: " << boundingRadius << std::endl;
}

// Helper function to transform mesh after reading it in.  Physically moves the mesh to the new transformed position.
void Mesh::Transform(double xAngle, double yAngle, double zAngle, double shiftX, double shiftY, double shiftZ)
{
  vtkNew<vtkTransform> transform;
  // Shift in Y to overlay tiff (double bounding box center
  transform->Translate(shiftX, shiftY, shiftZ);
  transform->RotateX(xAngle);
  transform->RotateY(yAngle);
  transform->RotateZ(zAngle);

  vtkNew<vtkTransformPolyDataFilter> transformFilter;
  transformFilter->SetInputData(this->polyData);
  transformFilter->SetTransform(transform);
  transformFilter->Update();

  this->polyData = transformFilter->GetOutput();

  // prevent premature deletion of the polyData object
  this->polyData->Register(nullptr);
}

void Mesh::Write(const std::string& filename) const
{
  vtkNew<vtkSTLWriter> writer;
  writer->SetFileName(filename.c_str());
  writer->SetInputData(this->polyData);
  writer->Write();
}
