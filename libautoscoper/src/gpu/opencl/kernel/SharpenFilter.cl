__kernel
void sharpen_filter_kernel(
		__global const float* input,
		__global float* output,
		int width,
		int height, 
		__constant float* filter,
		int filterSize,
		float contrast,
		float threshold)
{
	short x = get_global_id(0);
	short y = get_global_id(1);

	if (x > width-1 || y > height-1) return;
	
	/* perform convolution */
	float blur = 0.0f;
	int filterRadius = (filterSize - 1) / 2;

	for(int i = 0; i < filterSize; ++i){
		for(int j = 0; j < filterSize; ++j){
			
			int a = x - filterRadius + i;
			int b = y - filterRadius + j;			
						
			if(!(a < 0 || a >=width || b < 0 || b >= height))
				blur += filter[i*filterSize + j] * input[b*width + a];
		}
	}

	/* if original pixel and blurred pixel differ by more than threshold,
	 * difference is adjusted by contrast and added to original,
	 * else no change
	 */
	if (fabs(input[y*width+x] - blur) > threshold)
	{
		  output[y*width + x] = input[y*width+x] + contrast*(input[y*width + x] - blur);
		  
		if(output[y*width + x]  > 1)
			output[y*width + x] = 1;
		if(output[y*width + x]  < 0)
			output[y*width + x] = 0;
	}
	else
		output[y*width + x] = input[y*width + x];
	
}

// vim: ts=4 syntax=cpp noexpandtab
