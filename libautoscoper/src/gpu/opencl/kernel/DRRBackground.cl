__kernel
void drr_background_kernel(__global const float* src1,
                           __global float* dest,
                           unsigned width,
                           unsigned height)
{
  const uint x = get_global_id(0);
  const uint y = get_global_id(1);

  if (x > width - 1 || y > height - 1) return;

  const uint xy = y * width + x;

  // src1 maps to orange and src2 to blue
  dest[xy] = (src1[xy] != 0.0f) ? 1.0f : 0.0f;
}

// vim: ts=4 syntax=cpp noexpandtab
