#pragma once
#include <string>
#include <vector>

struct Facet {
	std::vector<float> v1;
	std::vector<float> v2;
	std::vector<float> v3;
	std::vector<float> normal;
	std::vector<float> centroid;
};

class Mesh {
public:
	Mesh(const std::string& filename);
	Mesh(const Mesh&);
	
	void TransformMesh(double xyzypr[]);
	void WriteSTL(const std::string& filename);

	unsigned int GetNumFacets() const { return numfacets_; }
	const Facet& GetFacet(unsigned int i) const { return facets_[i]; }
	float* GetAABB() { return aabb_; }
	void SetPosition(double xyzypr[]) { for (int i = 0; i < 6; i++) position_[i] = (float)xyzypr[i]; }
private:
	std::vector<Facet> facets_;
	unsigned int numfacets_;
	float aabb_[6];
	float position_[6]; // current position of the mesh in world coordinates

	void ReadSTL(const std::string& filename);
	void ReadSTLASCII(const std::string& filename);
	void ReadSTLBinary(const std::string& filename);
	void ReadVRML(const std::string& filename);

	void CalculateAABB();

	void StringToVector(std::string& str, std::vector<float>& v);
	void ReadCoordIndex(std::string& str, std::vector<int>& i);
};