__kernel
void rad_render_kernel(__global float* output,
                       unsigned width, unsigned height,
                       float u0, float v0, float u1, float v1,
                       float u2, float v2, float u3, float v3,
                       __read_only image2d_t image)
{
  const uint x = get_global_id(0);
  const uint y = get_global_id(1);

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_CLAMP |
                            CLK_FILTER_LINEAR;

  if (x > width-1 || y > height-1) {
    return;
  }

  const float u = u2+u3*(x/(float)width);
  const float v = v2+v3*(y/(float)height);

  const float s = (u-u0)/u1+0.5f;
  const float t = 0.5f-(v-v0)/v1;

  if (s < 0.0f || t < 0.0f || s > 1.0f || t > 1.0f) {
    output[width*y+x] = 0.0f;
  }
  else {
    output[width*y+x] = 1.0f-read_imagef(image, sampler, (float2)(s,t)).x;
  }
}

// vim: ts=4 syntax=cpp noexpandtab
