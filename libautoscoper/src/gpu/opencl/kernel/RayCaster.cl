// Render the volume using ray marching.
__kernel
void volume_render_kernel(__global float* buffer,
                          unsigned width, unsigned height,
                          float step, float intensity, float cutoff,
                          __constant float* viewport,
                          __constant float* imv,
                          __read_only image3d_t image)
{
  const uint x = get_global_id(0);
    const uint y = get_global_id(1);

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_CLAMP |
                            CLK_FILTER_LINEAR;

    if (x > width-1 || y > height-1) return;

    // Calculate the normalized device coordinates using the viewport
    const float u = viewport[0]+viewport[2]*(x/(float)width);
    const float v = viewport[1]+viewport[3]*(y/(float)height);

    // Determine the look ray in camera space.
    const float3 look = step*normalize((float3)(u, v, -2.f));

    // Calculate the ray in world space.
  // Origin is the last column of the invModelView matrix
  const float3 ray_origin = (float3)(imv[3], imv[7], imv[11]);
  // Direction is the invModelView x look
  const float3 ray_direction = (float3)(
              dot((float3)(imv[0], imv[1], imv[2]), look),
              dot((float3)(imv[4], imv[5], imv[6]), look),
              dot((float3)(imv[8], imv[9], imv[10]), look));

    // Find intersection with box.
    const float3 boxMin = (float3)(0.f, 0.f, -1.f);
    const float3 boxMax = (float3)(1.f, 1.f, 0.f);

    // Compute intersection of ray with all six planes.
    const float3 tBot = (boxMin-ray_origin)/ray_direction;
    const float3 tTop = (boxMax-ray_origin)/ray_direction;

    // Re-order intersections to find smallest and largest on each axis.
    const float3 tMin = min(tTop, tBot);
    const float3 tMax = max(tTop, tBot);

    // Find the largest tMin and the smallest tMax.
  float near = max(max(tMin.x, tMin.y), tMin.z);
  float far = min(min(tMax.x, tMax.y), tMax.z);

  if (!(far > near)) {
        buffer[y*width+x] = 0.f;
        return;
    }

    // Clamp to near plane.
  if (near < 0.f) near = 0.f;

    // Preform the ray marching from back to front.
    float t = far;
    float density = 0.f;
    while (t > near) {
        const float3 point = ray_origin+t*ray_direction;
        const float val = read_imagef(image, sampler, (float4)(point.x, 1.f-point.y, -point.z, 0) ).x;
        density += val > cutoff ? step*val : 0.f;
        t -= 1.f;
    }

    buffer[y*width+x] = clamp(density/intensity, 0.f, 1.f);
}

// vim: ts=4 syntax=cpp noexpandtab
