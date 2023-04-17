float moller_trumbore_intersection(
	float3 rayOrigin, float3 rayDirection,
	float3 v0, float3 v1, float3 v2) {
	// OpenCL Moller-Trumbore intersection algorithm
	// Based on the algorithm description from https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
	// A return value of -1.0f indicates that the ray is inside the triangle
	// A return value of 1.0f indicates that the ray does not intersect the triangle

	// calculate the triangle normal
	float3 e1 = v1 - v0;
	float3 e2 = v2 - v0;
	float3 normal = cross(e1, e2);

	// calculate the determinant
	float det = dot(normal, rayDirection);

	// check if the ray is parallel to the triangle
	if (det > -FLT_EPSILON && det < FLT_EPSILON) {
		return 1.0f;
	}

	// determine if the ray intersects the triangle
	float invDet = 1.0f / det;
	float3 t = rayOrigin - v0;
	float u = dot(t, rayDirection) * invDet;
	if (u < 0.0f || u > 1.0f) {
		return 1.0f;
	}

	float3 q = cross(t, e1);
	float v = dot(rayDirection, q) * invDet;
	if (v < 0.0f || u + v > 1.0f) {
		return 1.0f;
	}

	float t2 = dot(e2, q) * invDet;
	if (t2 < 0.0f) {
		return 1.0f;
	}

	return -1.0f;
}

__kernel
void sign_distance_kernel(__global float* dField,
	float dFieldOffsetX, float dFieldOffsetY, float dFieldOffsetZ,
	float voxelSizeX, float voxelSizeY, float voxelSizeZ,
	int gridSizeX, int gridSizeY, int gridSizeZ,
	__global float* vertex0, __global float* vertex1, __global float* vertex2,
	unsigned int nTriangles, int faceStart, int step) {

	int gidx = get_global_id(0);
	int gidy = get_global_id(1);
	int gidz = get_global_id(2);

	int gid = gridSizeX * gridSizeY * gidz + gridSizeX * gidy + gidx;

	if (gid < gridSizeX * gridSizeY * gridSizeZ) {
		// dField grid point
		float3 fPoint;
		fPoint.x = dFieldOffsetX + (float)gidx * voxelSizeX;
		fPoint.y = dFieldOffsetY + (float)gidy * voxelSizeY;
		fPoint.z = dFieldOffsetZ + (float)gidz * voxelSizeZ;

		// cast towards the origin
		float3 dir = -fPoint;

		float3 v0, v1, v2;
		float sign = 1.0f;
		for (int i = faceStart; (i < (faceStart + step)) && i < nTriangles; i++) {
			// get the vertecies of the triangle
			v0.x = vertex0[i * 3];
			v0.y = vertex0[i * 3 + 1];
			v0.z = vertex0[i * 3 + 2];

			v1.x = vertex1[i * 3];
			v1.y = vertex1[i * 3 + 1];
			v1.z = vertex1[i * 3 + 2];

			v2.x = vertex2[i * 3];
			v2.y = vertex2[i * 3 + 1];
			v2.z = vertex2[i * 3 + 2];

			// determine the intersection
			sign *= moller_trumbore_intersection(fPoint, dir, v0, v1, v2);
		}

		dField[gid] *= sign;
	}
}

__kernel
void calculate_unsigned_distance_kernel(__global float* dField,
	float dFieldOffsetX, float dFieldOffsetY, float dFieldOffsetZ,
	float voxelSizeX, float voxelSizeY, float voxelSizeZ,
	int gridSizeX, int gridSizeY, int gridSizeZ,
	__global float* meshPoints, unsigned int nMeshPoints, unsigned int meshStart, int step) {

	int gidx = get_global_id(0);
	int gidy = get_global_id(1);
	int gidz = get_global_id(2);

	int gid = gridSizeX * gridSizeY * gidz + gridSizeX * gidy + gidx;

	if (gid < gridSizeX * gridSizeY * gridSizeZ) {
		// dField grid point
		float3 fPoint;
		fPoint.x = dFieldOffsetX + (float)gidx * voxelSizeX;
		fPoint.y = dFieldOffsetY + (float)gidy * voxelSizeY;
		fPoint.z = dFieldOffsetZ + (float)gidz * voxelSizeZ;

		// init min dist
		float minDist = (meshStart == 0) ? FLT_MAX : dField[gid];

		// find the min dist from all points in the mesh
		float tmpDist;
		for (int i = meshStart; (i < (meshStart + step)) && i < nMeshPoints; i++) {
			float3 mPoint;
			mPoint.x = meshPoints[i * 3];
			mPoint.y = meshPoints[i * 3 + 1];
			mPoint.z = meshPoints[i * 3 + 2];
			tmpDist = length(fPoint - mPoint);
			if (tmpDist < minDist) {
				minDist = tmpDist;
			}
		}
		dField[gid] = minDist;
	}
}