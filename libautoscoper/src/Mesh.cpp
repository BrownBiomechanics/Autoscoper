#include "Mesh.hpp"
#include <vtkSTLReader.h>
#include <vtkSTLWriter.h>

Mesh::Mesh(const std::string& filename)
{
  vtkSTLReader* reader = vtkSTLReader::New();
  reader->SetFileName(filename.c_str());
  reader->Update();
  this->polyData = reader->GetOutput();
  reader->Delete();
}

Mesh::Mesh(const Mesh& other)
{
  this->polyData = vtkPolyData::New();
  this->polyData->DeepCopy(other.polyData);
}

void Mesh::Write(const std::string& filename) const
{
  vtkSTLWriter* writer = vtkSTLWriter::New();
  writer->SetFileName(filename.c_str());
  writer->SetInputData(this->polyData);
  writer->Write();
  writer->Delete();
}
