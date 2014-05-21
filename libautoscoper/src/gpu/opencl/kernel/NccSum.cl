__kernel
void ncc_sum_kernel(
		__global const float* f,
		__global float* sums,
		__local float* buffer,
		unsigned n)
{
	unsigned i = get_global_id(0) + get_global_id(1)*get_global_size(1); // global index
	unsigned t = get_local_id(0); // thread index

	buffer[t] = (i < n) ? f[i] : 0.0f;

	barrier(CLK_LOCAL_MEM_FENCE);
	for(unsigned s = get_local_size(0)/2; s > 0; s >>= 1) {
		if (t < s) {
			buffer[t] += buffer[t + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (t == 0) {
		sums[get_global_id(1)] = buffer[0];
	}
}

// vim: ts=4 syntax=cpp noexpandtab
