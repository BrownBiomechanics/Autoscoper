//! \file Intersection.hpp
//! \author Andy Loomis (aloomis)

#include <cmath>
#include <iostream>

#include "Vector.hpp"

struct Ray
{
  Vec3f origin;
  Vec3f direction;
};

float ray_point_intersect(const Ray& ray, const Vec3f p, Vec3f* ray_point = 0)
{
  Vec3f u = p - ray.origin;
  Vec3f v = ray.origin + dot(u, ray.direction) * ray.direction;

  if (ray_point != 0) {
    *ray_point = v;
  }

  return len(p - v);
}

float ray_plane_intersect(const Ray& ray,
                          const Vec3f& point,
                          const Vec3f& normal,
                          Vec3f* q)
{
  float num = dot(point - ray.origin, normal);
  float den = dot(ray.direction, normal);

  if (abs(den) < std::numeric_limits<float>::epsilon()) {
    if (q != 0) {
      *q = point;
    }
    return num;
  } else {
    float t = num / den;
    if (q != 0) {
      *q = ray.origin + t * ray.direction;
    }
    return 0.0f;
  }
}

// http://softsurfer.com/Archive/algorithm_0106/algorithm_0106.htm

float ray_line_intersect(const Ray& ray,
                         const Vec3f& p0,
                         const Vec3f& p1,
                         Vec3f* ray_point = 0,
                         Vec3f* line_point = 0)
{
  Vec3f u = ray.direction;
  Vec3f v = p1 - p0;
  Vec3f w = ray.origin - p0;

  float dotuu = dot(u, u);
  float dotuv = dot(u, v);
  float dotuw = dot(u, w);
  float dotvv = dot(v, v);
  float dotvw = dot(v, w);
  float den = dotuu * dotvv - dotuv * dotuv;

  float t[2] = { 0.0f, 0.0f };
  if (den < std::numeric_limits<float>::epsilon()) {
    t[1] = dotuv > dotvv ? dotuw / dotuv : dotvw / dotvv;
  } else {
    t[0] = (dotuv * dotvw - dotvv * dotuw) / den;
    t[1] = (dotuu * dotvw - dotuv * dotuw) / den;
  }

  Vec3f q0 = ray.origin + t[0] * ray.direction;
  Vec3f q1 = p0 + t[1] * (p1 - p0);

  if (ray_point != 0) {
    *ray_point = q0;
  }
  if (line_point != 0) {
    *line_point = q1;
  }

  return len(q0 - q1);
}

float ray_segment_intersect(const Ray& ray,
                            const Vec3f& p0,
                            const Vec3f& p1,
                            Vec3f* ray_point = 0,
                            Vec3f* segment_point = 0)
{
  Vec3f u = ray.direction;
  Vec3f v = p1 - p0;
  Vec3f w = ray.origin - p0;

  float dotuu = dot(u, u);
  float dotuv = dot(u, v);
  float dotuw = dot(u, w);
  float dotvv = dot(v, v);
  float dotvw = dot(v, w);
  float den = dotuu * dotvv - dotuv * dotuv;

  float t[2] = { 0.0f, 0.0f };
  if (den < std::numeric_limits<float>::epsilon()) {
    t[1] = dotuv > dotvv ? dotuw / dotuv : dotvw / dotvv;
  } else {
    t[0] = (dotuv * dotvw - dotvv * dotuw) / den;
    t[1] = (dotuu * dotvw - dotuv * dotuw) / den;
  }

  // Clamp to the ends of the line segment.

  if (t[1] < 0.0f) {
    t[1] = 0.0f;
  }
  if (t[1] > 1.0f) {
    t[1] = 1.0f;
  }

  Vec3f q0 = ray.origin + t[0] * ray.direction;
  Vec3f q1 = p0 + t[1] * (p1 - p0);

  if (ray_point != 0) {
    *ray_point = q0;
  }
  if (segment_point != 0) {
    *segment_point = q1;
  }

  return len(q0 - q1);
}

float ray_circle_intersect(const Ray& ray,
                           const Vec3f& center,
                           float radius,
                           const Vec3f& u,
                           const Vec3f& v,
                           float alpha,
                           float beta,
                           Vec3f* ray_point = 0,
                           Vec3f* circle_point = 0)
{
  const int slices = 64;

  float min_dist = std::numeric_limits<float>::max();

  Vec3f q0, q1;
  for (int i = 0; i < slices; ++i) {
    float theta0 = alpha + i * (beta - alpha) / slices;
    float theta1 = alpha + (i + 1) * (beta - alpha) / slices;

    Vec3f p0 = center + radius * (cos(theta0) * u + sin(theta0) * v);
    Vec3f p1 = center + radius * (cos(theta1) * u + sin(theta1) * v);

    float dist = ray_segment_intersect(ray, p0, p1, &q0, &q1);
    if (dist < min_dist) {
      min_dist = dist;
      if (ray_point != 0) {
        *ray_point = q0;
      }
      if (circle_point != 0) {
        *circle_point = q1;
      }
    }
  }

  return min_dist;
}
