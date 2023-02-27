#include "Mesh.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>

Mesh::Mesh(const std::string& filename) {
	// check the file extension
	std::string extension = filename.substr(filename.find_last_of(".") + 1);
	if (extension == "stl") {
		ReadSTL(filename);
	}
	else if (extension == "wrl" || extension == "iv") {
		ReadVRML(filename);
	}
	else {
		throw std::runtime_error("Mesh: File extension, " + extension + ", is not supported");
	}
}

Mesh::Mesh(const Mesh& mesh) {
	facets_ = mesh.facets_;
	numfacets_ = mesh.numfacets_;
	for (int i = 0; i < 6; i++) {
		aabb_[i] = mesh.aabb_[i];
		position_[i] = mesh.position_[i];
	}
}

void Mesh::TransformMesh(double new_position[]) {
	// xyzypr is a 6 element array of doubles that contains new x, y, z, yaw, pitch, and roll values for the mesh in world coordinates

	// calculate the movement from the current position to the new position
	double movement[6];
	for (int i = 0; i < 6; i++) {
		movement[i] = new_position[i] - position_[i];
	}

	// calculate the rotation matrix
	double rotation[9];
	rotation[0] = cos(movement[5]) * cos(movement[4]);
	rotation[1] = cos(movement[5]) * sin(movement[4]) * sin(movement[3]) - sin(movement[5]) * cos(movement[3]);
	rotation[2] = cos(movement[5]) * sin(movement[4]) * cos(movement[3]) + sin(movement[5]) * sin(movement[3]);
	rotation[3] = sin(movement[5]) * cos(movement[4]);
	rotation[4] = sin(movement[5]) * sin(movement[4]) * sin(movement[3]) + cos(movement[5]) * cos(movement[3]);
	rotation[5] = sin(movement[5]) * sin(movement[4]) * cos(movement[3]) - cos(movement[5]) * sin(movement[3]);
	rotation[6] = -sin(movement[4]);
	rotation[7] = cos(movement[4]) * sin(movement[3]);
	rotation[8] = cos(movement[4]) * cos(movement[3]);

	// calculate the translation vector
	double translation[3];
	translation[0] = movement[0];
	translation[1] = movement[1];
	translation[2] = movement[2];

	// transform the mesh (the mesh points are in the model coordinate system)
	// convert the translation vector and rotation matrix to the model coordinate system
	double model_translation[3];
	double model_rotation[9];
	model_translation[0] = translation[0] * rotation[0] + translation[1] * rotation[1] + translation[2] * rotation[2];
	model_translation[1] = translation[0] * rotation[3] + translation[1] * rotation[4] + translation[2] * rotation[5];
	model_translation[2] = translation[0] * rotation[6] + translation[1] * rotation[7] + translation[2] * rotation[8];
	model_rotation[0] = rotation[0];
	model_rotation[1] = rotation[1];
	model_rotation[2] = rotation[2];
	model_rotation[3] = rotation[3];
	model_rotation[4] = rotation[4];
	model_rotation[5] = rotation[5];
	model_rotation[6] = rotation[6];
	model_rotation[7] = rotation[7];
	model_rotation[8] = rotation[8];
	// transform the mesh
	for (Facet& facet : facets_) {
		facet.v1[0] = facet.v1[0] * model_rotation[0] + facet.v1[1] * model_rotation[1] + facet.v1[2] * model_rotation[2] + model_translation[0];
		facet.v1[1] = facet.v1[0] * model_rotation[3] + facet.v1[1] * model_rotation[4] + facet.v1[2] * model_rotation[5] + model_translation[1];
		facet.v1[2] = facet.v1[0] * model_rotation[6] + facet.v1[1] * model_rotation[7] + facet.v1[2] * model_rotation[8] + model_translation[2];
		facet.v2[0] = facet.v2[0] * model_rotation[0] + facet.v2[1] * model_rotation[1] + facet.v2[2] * model_rotation[2] + model_translation[0];
		facet.v2[1] = facet.v2[0] * model_rotation[3] + facet.v2[1] * model_rotation[4] + facet.v2[2] * model_rotation[5] + model_translation[1];
		facet.v2[2] = facet.v2[0] * model_rotation[6] + facet.v2[1] * model_rotation[7] + facet.v2[2] * model_rotation[8] + model_translation[2];
		facet.v3[0] = facet.v3[0] * model_rotation[0] + facet.v3[1] * model_rotation[1] + facet.v3[2] * model_rotation[2] + model_translation[0];
		facet.v3[1] = facet.v3[0] * model_rotation[3] + facet.v3[1] * model_rotation[4] + facet.v3[2] * model_rotation[5] + model_translation[1];
		facet.v3[2] = facet.v3[0] * model_rotation[6] + facet.v3[1] * model_rotation[7] + facet.v3[2] * model_rotation[8] + model_translation[2];
		facet.centroid[0] = facet.centroid[0] * model_rotation[0] + facet.centroid[1] * model_rotation[1] + facet.centroid[2] * model_rotation[2] + model_translation[0];
		facet.centroid[1] = facet.centroid[0] * model_rotation[3] + facet.centroid[1] * model_rotation[4] + facet.centroid[2] * model_rotation[5] + model_translation[1];
		facet.centroid[2] = facet.centroid[0] * model_rotation[6] + facet.centroid[1] * model_rotation[7] + facet.centroid[2] * model_rotation[8] + model_translation[2];
		facet.normal[0] = facet.normal[0] * model_rotation[0] + facet.normal[1] * model_rotation[1] + facet.normal[2] * model_rotation[2];
		facet.normal[1] = facet.normal[0] * model_rotation[3] + facet.normal[1] * model_rotation[4] + facet.normal[2] * model_rotation[5];
		facet.normal[2] = facet.normal[0] * model_rotation[6] + facet.normal[1] * model_rotation[7] + facet.normal[2] * model_rotation[8];
	}
	// update the bounding box and position
	CalculateAABB();
	for (int i = 0; i < 3; i++) {
		position_[i] = new_position[i];
	}

}

void Mesh::WriteSTL(const std::string& filename) {
	// Open the file in write mode
	std::ofstream file(filename, std::ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error("Mesh: Could not open file");
	}
	// write the STL in Binary
	// write the header -> use 80 blank chars since the header is generally ignored
	char header[81];
	for (int i = 0; i < 80; i++) {
		header[i] = ' ';
	}
	header[80] = '\0';
	file.write(header, 80);
	// write the number of facets
	file.write((char*)&numfacets_, 4);
	// write the facets
	for (Facet& facet : facets_) {
		// write the normal
		file.write((char*)&facet.normal[0], 4);
		file.write((char*)&facet.normal[1], 4);
		file.write((char*)&facet.normal[2], 4);
		// write the vertices
		file.write((char*)&facet.v1[0], 4);
		file.write((char*)&facet.v1[1], 4);
		file.write((char*)&facet.v1[2], 4);
		file.write((char*)&facet.v2[0], 4);
		file.write((char*)&facet.v2[1], 4);
		file.write((char*)&facet.v2[2], 4);
		file.write((char*)&facet.v3[0], 4);
		file.write((char*)&facet.v3[1], 4);
		file.write((char*)&facet.v3[2], 4);
		// write the attribute byte count
		unsigned short attribute = 0;
		file.write((char*)&attribute, 2);
	}
	file.close();
}


void Mesh::ReadSTL(const std::string& filename) {
		// open the file
		std::fstream file(filename, std::ios::in);
		if (!file.is_open()) {
				throw std::runtime_error("Mesh: Could not open file");
		}

		// determine if the file is binary or ascii
		std::string line;
		std::getline(file, line);
		file.close();
		if (line.find("solid") != std::string::npos) {
				ReadSTLASCII(filename);
		}
		else {
				ReadSTLBinary(filename);
		}

		CalculateAABB();
}

void Mesh::ReadSTLASCII(const std::string& filename) {
		// open the file
		std::fstream file(filename, std::ios::in);
		if (!file.is_open()) {
				throw std::runtime_error("Mesh: Could not open file");
		}

		// read the file
		std::string line;
		std::getline(file, line);
		// get the name from the first line
		std::string name = line.substr(6);
		Facet f{};
		while (line.find("endsolid") == std::string::npos) {
				std::getline(file, line); // get the next line
				if (line.find("facet normal") != std::string::npos) {
						StringToVector(line.substr(13), f.normal);
				}
				else if (line.find("outer loop") != std::string::npos) {
						std::getline(file, line);
						StringToVector(line.substr(7), f.v1); 
						std::getline(file, line);
						StringToVector(line.substr(7), f.v2);
						std::getline(file, line);
						StringToVector(line.substr(7), f.v3);
						f.centroid = { (f.v1[0] + f.v2[0] + f.v3[0]) / 3.0f, (f.v1[1] + f.v2[1] + f.v3[1]) / 3.0f, (f.v1[2] + f.v2[2] + f.v3[2]) / 3.0f };
						facets_.push_back(f);
				} 
		}
		file.close();
		numfacets_ = facets_.size();
}

void Mesh::ReadSTLBinary(const std::string& filename) {
		// open the file
		std::fstream file(filename, std::ios::in | std::ios::binary);
		if (!file.is_open()) {
				throw std::runtime_error("Mesh: Could not open file");
		}

		// read the 80 char header
		char header[81];
		size_t num_bytes = sizeof(char) * 80;
		file.read(reinterpret_cast<char*>(header), num_bytes);
		header[80] = '\0';

		// get the number of triangles
		file.read(reinterpret_cast<char*>(numfacets_), sizeof(numfacets_));

		for (int i = 0; i < numfacets_; ++i) {
				// read the normal
				float normal[3];
				num_bytes = sizeof(float) * 3;
				file.read(reinterpret_cast<char*>(normal), num_bytes);

				// read the vertices
				float v1[3];
				float v2[3];
				float v3[3];
				num_bytes = sizeof(float) * 3;
				file.read(reinterpret_cast<char*>(v1), num_bytes);
				file.read(reinterpret_cast<char*>(v2), num_bytes);
				file.read(reinterpret_cast<char*>(v3), num_bytes);

				// read the attribute byte count
				unsigned short attribute_byte_count;
				num_bytes = sizeof(unsigned short);
				file.read(reinterpret_cast<char*>(attribute_byte_count), num_bytes);

				// Move data into facet struct
				Facet f{};
				f.normal = { normal[0],normal[1],normal[2] };
				f.v1 = { v1[0],v1[1],v1[2] };
				f.v2 = { v2[0],v2[1],v2[2] };
				f.v3 = { v3[0],v3[1],v3[2] };
				f.centroid = { (f.v1[0] + f.v2[0] + f.v3[0]) / 3.0f, (f.v1[1] + f.v2[1] + f.v3[1]) / 3.0f, (f.v1[2] + f.v2[2] + f.v3[2]) / 3.0f };
				// add the facet to the mesh
				facets_.push_back(f);
		}
		file.close();
		numfacets_ = facets_.size();
}

void Mesh::ReadVRML(const std::string& filename) {
		std::string MIMICS_10_KEY = "#coordinates written in 1mm / 10000";
		std::string MIMICS_13_KEY = "#coordinates written in 1mm / 0";
		std::string MIMICS_22_KEY = "#resulted coordinates are measured in units, where 1 unit = 1000.000000 mm";

		std::vector<std::vector<float>> points;
		std::vector<std::vector<int>> indices;

		// open the file
		std::fstream file(filename, std::ios::in);
		if (!file.is_open()) {
				throw std::runtime_error("Mesh: Could not open file");
		}

		// read the file
		std::string line;
		bool isMimics = false;
		std::vector<float> scale = { 1.0f,1.0f,1.0f };
		while (std::getline(file, line)) {
				// check if the line contains one of the keys
				if (line.find(MIMICS_10_KEY) != std::string::npos || line.find(MIMICS_13_KEY) != std::string::npos || line.find(MIMICS_22_KEY) != std::string::npos) {
						isMimics = true;
				}
				// read the points
				else if (line.find("point") != std::string::npos && line.find("Viewpoint") == std::string::npos) {
						std::vector<float> v;
						std::getline(file, line); // next line could either be a point or a [
						while (line.find("]") == std::string::npos) {
								if (line.find("[") != std::string::npos) 
										std::getline(file, line);
								StringToVector(line, v);
								points.push_back(v);
								std::getline(file, line);
						}
				}
				// read the coord index
				else if (line.find("coordIndex") != std::string::npos) {
						std::vector<int> i;
						std::getline(file, line); // next line could either be an index group or a [
						while (line.find("]") == std::string::npos) {
								if (line.find("[") != std::string::npos)
										std::getline(file, line);
								ReadCoordIndex(line, i);
								indices.push_back(i);
								std::getline(file, line);
						}
				}
				else if (line.find("scale") != std::string::npos) {
						StringToVector(line.substr(6), scale);
				}
		}
		file.close();
		// scale the points
		for (int i = 0; i < points.size(); ++i) {
				points[i][0] *= scale[0];
				points[i][1] *= scale[1];
				points[i][2] *= scale[2];
		}
		if (isMimics) { // if this flag is true then the points are in m and need to be converted to mm 
				for (int i = 0; i < points.size(); ++i) {
						points[i][0] *= 1000.0f;
						points[i][1] *= 1000.0f;
						points[i][2] *= 1000.0f;
				}
		}
		// convert the points and indices to facets
		Facet f{};
		for (int i = 0; i < indices.size(); ++i) {
				f.v1 = points[indices[i][0]];
				f.v2 = points[indices[i][1]];
				f.v3 = points[indices[i][2]];
				f.centroid = { (f.v1[0] + f.v2[0] + f.v3[0]) / 3.0f, (f.v1[1] + f.v2[1] + f.v3[1]) / 3.0f, (f.v1[2] + f.v2[2] + f.v3[2]) / 3.0f };
				// calculate the normal
				std::vector<float> U = { (f.v2[0] - f.v1[0]), (f.v2[1] - f.v1[1]), (f.v2[2] - f.v1[2] )};
				std::vector<float> V = { (f.v3[0] - f.v1[0]), (f.v3[1] - f.v1[1]), (f.v3[2] - f.v1[2] )};
				float x = (U[1] * V[2]) - (U[2] * V[1]);
				float y = (U[2] * V[0]) - (U[0] * V[2]);
				float z = (U[0] * V[1]) - (U[1] * V[0]);
				f.normal = { x, y, z };
				facets_.push_back(f);
		}
		file.close();
		numfacets_ = facets_.size();
}

void Mesh::CalculateAABB() {
	// find the min and max for all xyz values of the mesh
	float minx, miny, minz, maxx, maxy, maxz;
	minx = miny = minz = std::numeric_limits<float>::max();
	maxx = maxy = maxz = std::numeric_limits<float>::min();
	for (Facet f : facets_) {
		minx = std::min(minx, std::min(f.v1[0], std::min(f.v2[0], f.v3[0])));
		miny = std::min(miny, std::min(f.v1[1], std::min(f.v2[1], f.v3[1])));
		minz = std::min(minz, std::min(f.v1[2], std::min(f.v2[2], f.v3[2])));
		maxx = std::max(maxx, std::max(f.v1[0], std::max(f.v2[0], f.v3[0])));
		maxy = std::max(maxy, std::max(f.v1[1], std::max(f.v2[1], f.v3[1])));
		maxz = std::max(maxz, std::max(f.v1[2], std::max(f.v2[2], f.v3[2])));
	}
	aabb_[0] = minx;
	aabb_[1] = miny;
	aabb_[2] = minz;
	aabb_[3] = maxx;
	aabb_[4] = maxy;
	aabb_[5] = maxz;
}

void Mesh::StringToVector(std::string& str, std::vector<float>& v) {
		size_t spacePos;
		float v1 = std::stof(str, &spacePos);
		str = str.substr(spacePos + 1);
		float v2 = std::stof(str, &spacePos);
		str = str.substr(spacePos + 1);
		v = { v1,v2,std::stof(str) };
}

void Mesh::ReadCoordIndex(std::string& str, std::vector<int>& i) {
		size_t spacePos;
		int x = std::stoi(str, &spacePos);
		str = str.substr(spacePos + 1);
		while (str[spacePos] == ' ' || str[spacePos] == ',') {
				spacePos++;
		}
		int y = std::stoi(str, &spacePos);
		str = str.substr(spacePos+1);
		while (str[spacePos] == ' ' || str[spacePos] == ',') {
				spacePos++;
		}
		i = { x,y,std::stoi(str) };
}

