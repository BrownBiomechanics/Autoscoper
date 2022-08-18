__kernel
void sobel_filter_kernel(
    __global const float* input,
    __global float* output,
    int width,
    int height,
    float scale,
    float blend)
{
  short x1 = get_global_id(0);
  short y1 = get_global_id(1);

  if (x1 > width-1 || y1 > height-1) return;

  short x0 = x1-1; if (x0 < 0) x0 = 0;
  short y0 = y1-1; if (y0 < 0) y0 = 0;

  short x2 = x1+1; if (x2 > width-1) x2 = width-1;
  short y2 = y1+1; if (y2 > height-1) y2 = height-1;

  float pix00 = input[y0*width+x0];
  float pix01 = input[y0*width+x1];
  float pix02 = input[y0*width+x2];
  float pix10 = input[y1*width+x0];
  float pix11 = input[y1*width+x1];
  float pix12 = input[y1*width+x2];
  float pix20 = input[y2*width+x0];
  float pix21 = input[y2*width+x1];
  float pix22 = input[y2*width+x2];

  float horz = pix02+2*pix12+pix22-pix00-2*pix10-pix20;
  float vert = pix00+2*pix01+pix02-pix20-2*pix21-pix22;
  float grad = sqrt(horz*horz+vert*vert);

  float sum;
  if (blend < 0.5f) {
    sum = scale*grad+2.0f*blend*pix11;
  }
  else {
    sum = 2.0f*(1.0f-blend)*scale*grad+pix11;
  }
  if (sum < 0.0f) {
    sum = 0.0f;
  }
  else if (sum > 1.0f) {
    sum = 1.0f;
  }

  output[y1*width+x1] = sum;
}

// vim: ts=4 syntax=cpp noexpandtab
