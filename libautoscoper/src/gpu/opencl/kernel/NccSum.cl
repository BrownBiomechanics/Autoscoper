// Adds up the n values of f in blocks: each work-group sums
// get_local_size(0) consecutive values and writes the total to
// sums[group index]. The work-group size must be a power of two.
__kernel
void ncc_sum_kernel(
  __global const float* f,
  __global float* sums,
  __local float* buffer,
  unsigned n)
{
  const unsigned i = get_global_id(0) + get_global_id(1) * get_global_size(0); // global index
  const unsigned t = get_local_id(0); // thread index

  buffer[t] = (i < n) ? f[i] : 0.0f;

  barrier(CLK_LOCAL_MEM_FENCE);
  for (unsigned s = get_local_size(0) / 2; s > 0; s >>= 1) {
    if (t < s) {
      buffer[t] += buffer[t + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (t == 0) {
    sums[get_group_id(1)] = buffer[0];
  }
}

// vim: ts=4 syntax=cpp noexpandtab
