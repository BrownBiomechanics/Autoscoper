#include "Mesh.hpp"
#include <vtkSTLReader.h>
#include <vtkSTLWriter.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkCenterOfMass.h>

Mesh::Mesh(const std::string& filename) {

	fileName = filename; 

	vtkSTLReader* reader = vtkSTLReader::New();
	this->polyData = vtkPolyData::New();


	reader->SetFileName(filename.c_str());
	reader->Update();
	this->polyData = reader->GetOutput();

	// Apply transfom to overlay tif
	auto center = this->polyData->GetCenter();
	// std::cout << "Mesh " << filename << " center = " << center[0] << ", " << center[1] << ", " << center[2] << std::endl;

	

	// Register so it exists after deleting the reader
	this->polyData->Register(nullptr);

	auto newCenter = this->polyData->GetCenter();
	reader->Delete();
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
	this->polyData->Register(nullptr);
}

//Mesh::Mesh(const Mesh& other) {
//	this->polyData = vtkPolyData::New();
//	this->polyData->DeepCopy(other.polyData);
//}

void Mesh::Write(const std::string& filename) const {
  vtkSTLWriter* writer = vtkSTLWriter::New();
  writer->SetFileName(filename.c_str());
  writer->SetInputData(this->polyData);
  writer->Write();
  writer->Delete();
}
