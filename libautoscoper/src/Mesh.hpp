#pragma once
#include <string>
#include <vtkPolyData.h>

class Mesh {
public:
	Mesh(const std::string& filename);
	//Mesh(const Mesh&);

	vtkPolyData* GetPolyData() const { return this->polyData; }

	void Write(const std::string& filename) const;

private:

	vtkPolyData* polyData;
};