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

	// Create OBB Tree
	this->meshOBB = vtkOBBTree::New();
	// this->meshOBB->SetDataSet(this->polyData);
	this->meshOBB->SetMaxLevel(1);
	
	double corner[3];
	double maxAxis[3];
	double midAxis[3];
	double minAxis[3];
	double size[3];
	
	this->meshOBB->ComputeOBB(this->polyData, corner, maxAxis, midAxis, minAxis, size);

	/*std::cout << "corner = " << corner[0] << ", " << corner[1] << ", " << corner[2] << std::endl;
	std::cout << "maxAxis = " << maxAxis[0] << ", " << maxAxis[1] << ", " << maxAxis[2] << std::endl;
	std::cout << "midAxis = " << midAxis[0] << ", " << midAxis[1] << ", " << midAxis[2] << std::endl;
	std::cout << "minAxis = " << minAxis[0] << ", " << minAxis[1] << ", " << minAxis[2] << std::endl;
	std::cout << "size = " << size[0] << ", " << size[1] << ", " << size[2] << std::endl;*/

	
	/*this->meshOBB->Update();
	this->meshOBB->PrintSelf(std::cout, vtkIndent(2));*/

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
