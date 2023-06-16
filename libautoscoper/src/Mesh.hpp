#pragma once
#include <string>
#include <vtkPolyData.h>
#include <vtkOBBTree.h>


class Mesh {
public:
    Mesh(const std::string& filename);
    // Mesh(const Mesh&);

    vtkPolyData* GetPolyData() const { return this->polyData; }

    void Write(const std::string& filename) const;

    void Mesh::Transform(double xAngle, double yAngle, double zAngle, double shiftX, double shiftY, double shiftZ);

    std::string fileName;

private:

    vtkPolyData* polyData;
    vtkOBBTree* meshOBB;

};