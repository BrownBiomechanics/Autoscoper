__kernel
void contrast_filter_kernel(
    __global const float* input,
    __global float* output,
    int width,
    int height,
    float alpha,
    float beta,
    int size)
{
  short x = get_global_id(0);
  short y = get_global_id(1);

  if (x > width-1 || y > height-1) return;

  float fxy = input[y*width+x];

  // compute average
  float n = 0.0f;
  float sum = 0.0f;
  int minI = max(y-size/2, 0);
  int maxI = min(y+(size+1)/2, height);
  int minJ = max(x-size/2, 0);
  int maxJ = min(x+(size+1)/2, width);
  for (int i = minI; i < maxI; ++i) {
    for (int j = minJ; j < maxJ; ++j) {
      n += 1.0f;
      sum += input[i*width+j];
    }
  }
  float axy = sum/n;

  float gxy = 0.0f;
  if (axy > 0.01f) {
    gxy = pow(axy,alpha-beta)*pow(fxy,beta);
  }

  output[y*width+x] = gxy;
}

// vim: ts=4 syntax=cpp noexpandtab
