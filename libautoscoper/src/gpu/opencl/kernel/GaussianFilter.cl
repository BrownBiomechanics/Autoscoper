__kernel
void gaussian_filter_kernel(
    __global const float* input,
    __global float* output,
    int width,
    int height,
    __constant float* filter,
    int filterSize)
{
  short x = get_global_id(0);
  short y = get_global_id(1);

  if (x > width-1 || y > height-1) return;

  float centerValue=0.0f;
  int filterRadius = (filterSize - 1) / 2;

  for(int i = 0; i < filterSize; ++i){
    for(int j = 0; j < filterSize; ++j){

      int a = x - filterRadius + i;
      int b = y - filterRadius + j;

      if(!(a < 0 || a >=width || b < 0 || b >= height))
         centerValue = centerValue + (filter[i*filterSize + j])*(input[b*width + a]);
    }
  }

  if(centerValue > 1)
    centerValue = 1;
  if(centerValue < 0)
    centerValue = 0;

  output[y*width+x] = centerValue;
}

// vim: ts=4 syntax=cpp noexpandtab
