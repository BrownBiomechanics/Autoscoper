#pragma once
#include <string>
#include <vtkPolyData.h>
#include <vtkOBBTree.h>

class Mesh
{
public:
  Mesh(const std::string& filename);

  vtkPolyData* GetPolyData() const { return this->polyData; }

  void Write(const std::string& filename) const;

  void Transform(double xAngle, double yAngle, double zAngle, double shiftX, double shiftY, double shiftZ);

  double getBoundingRadius() const { return boundingRadius; }

private:
  double boundingRadius = 0.0;
  std::string filename;

  vtkPolyData* polyData;
  vtkOBBTree* meshOBB;
};
