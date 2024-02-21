__kernel
void ncc_kernel(
  __global const float* f,
  float meanF,
  __global const float* g,
  float meanG,
  __global const float* mask,
  __global float* nums,
  __global float* den1s,
  __global float* den2s,
  unsigned n)
{
  unsigned i = get_global_id(0);

  if (i < n && mask[i] > 0.5f) {
    float fMinusMean = f[i] - meanF;
    float gMinusMean = g[i] - meanG;

    nums[i] = fMinusMean * gMinusMean;
    den1s[i] = fMinusMean * fMinusMean;
    den2s[i] = gMinusMean * gMinusMean;
  } else {
    nums[i] = 0.0f;
    den1s[i] = 0.0f;
    den2s[i] = 0.0f;
  }
}

// vim: ts=4 syntax=cpp noexpandtab
