__kernel
void hdist_kernel(
	__global const float* f,
	float meanF,
	__global const float* g,
	float meanG,
	__global const float* mask,
	__global float* nums,
	unsigned n
) {
	unsigned i = get_global_id(0);

	if (i < n && mask[i] > 0.5f) {
		float fMinusMean = f[i]-meanF;
		float gMinusMean = g[i]-meanG;

		nums[i] = fabs(fMinusMean-gMinusMean);
	}
	else {
		nums[i] = 0.0f;
	}

}