__kernel
void composite_kernel(__global const float* src1,
                      __global const float* src2,
            __global const float* src3,
            __global const float* src4,
                      __global float* dest,
                      unsigned width,
                      unsigned height)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    if (x > width-1 || y > height-1) return;

  const uint xy = y*width + x;

  float multi = (src3[xy] < 0.5f) ? 0.0f : 1.0f;


    // src1 maps to orange and src2 to blue
    dest[3*xy+0] = src1[xy];
  dest[3 * xy + 1] = multi* (src1[xy] / 2.f + src2[xy] / 2.f);
    dest[3*xy+2] = src2[xy];
}

// vim: ts=4 syntax=cpp noexpandtab
