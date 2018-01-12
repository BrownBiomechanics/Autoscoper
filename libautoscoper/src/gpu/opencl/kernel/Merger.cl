__kernel
void merge_kernel(__global const float* src1,
                      __global const float* src2,
                      __global float* dest,
                      unsigned width,
                      unsigned height)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    if (x > width-1 || y > height-1) return;

	const uint xy = y*width + x;	

    // src1 maps to orange and src2 to blue
    dest[xy] = min(src1[xy] + src2[xy],1.0);
}

// vim: ts=4 syntax=cpp noexpandtab
