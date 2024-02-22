//! \file Ray.hpp
//! \author Andy Loomis (aloomis)

#ifndef RAY_HPP
#define RAY_HPP

#include "Vector.hpp"

template <class T = float>
class Ray
{
public:

  //! Constructs a ray with the specified origin and direction

  Ray(const Vec3<T>& origin, const Vec3<T>& direction)
    : origin(origin), direction(direction) {}

  //! Returns the point on the ray at the specified distance from the
  //! origin.

  Vec3<T> at(const T& t) const
  {
    return origin + t * direction;
  }

  //! Returns the closest distance between this ray the specified point.
  //! Optionally, returns the point on this ray closes to this point.

  T intersect_point(const Vec3<T>& point, Vec3<T>* ray_point = 0) const
  {
    Vec3d u = origin + dot(point - origin, direction) * direction;

    if (ray_point != 0) {
      *ray_point = u;
    }

    return len(point - u);
  }

  //! Returns the closest distance between this ray and the specified sphere.
  //! Optionally, returns the point on this ray and the sphere that satisfies
  //! this distance.

  T intersect_sphere(const Vec3<T>& center,
                     const T& radius,
                     Vec3<T>* ray_point = 0,
                     Vec3<T>* sphere_point = 0) const
  {
    Vec3d u = origin + dot(center - origin, direction) * direction;
    T distance = len(center - u);

    // The ray misses the sphere

    if (distance >= radius) {
      if (ray_point != 0) {
        *ray_point = u;
      }
      if (sphere_point != 0) {
        *sphere_point = center + radius * unit(u - center);
      }
      return distance - radius;
    }
    // The ray intersects the sphere
    else {

      // Setup the quadratic equation

      T a = lensq(direction);
      // clang-format off
      T b = 2 * (origin.x * direction.x +
                 origin.y * direction.y +
                 origin.z * direction.z);
      // clang-format on
      T c = lensq(origin) - radius * radius;

      // Solve the equation. We don't need to check the determinant
      // because we already know the ray and sphere intersect. Moreover we
      // are only interested in the first intersection with the sphere.

      T t = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);

      u = this->at(t);
      if (ray_point != 0) {
        *ray_point = u;
      }
      if (sphere_point != 0) {
        *sphere_point = u;
      }
      return 0;
    }
  }

  //! Returns the closest distance between this ray and the specified plane.
  //! Optionally returns the point on this ray and the plane that satisfy this
  //! distance.

  T intersect_plane(const Vec3<T>& point,
                    const Vec3<T>& normal,
                    Vec3<T>* ray_point = 0,
                    Vec3<T>* plane_point = 0) const
  {
    T num = dot(point - origin, normal);
    T den = dot(direction, normal);

    // The ray is parallel to the plane, return the origin of the ray and
    // its projection onto the plane.

    if (abs(den) < std::numeric_limits<T>::epsilon()) {
      if (ray_point != 0) {
        *ray_point = origin;
      }
      if (plane_point != 0) {
        *plane_point = origin - num * normal;
      }
      return num;
    }
    // The ray and plane intersect, calculate the intersection point
    else {
      Vec3<T> u = this->at(num / den);
      if (ray_point != 0) {
        *ray_point = u;
      }
      if (plane_point != 0) {
        *plane_point = u;
      }
      return 0;
    }
  }

  //! Returns the closest distance between this ray and the specified line.
  //! Optionally returns the point on this ray and the line that satisfy this
  //! distance.

  // http://softsurfer.com/Archive/algorithm_0106/algorithm_0106.htm

  T intersect_line(const Vec3<T>& point1,
                   const Vec3<T>& point2,
                   Vec3<T>* ray_point = 0,
                   Vec3<T>* line_point = 0) const
  {
    Vec3<T> v = point2 - point1;
    Vec3<T> w = origin - point1;

    T dotuu = dot(direction, direction);
    T dotuv = dot(direction, v);
    T dotuw = dot(direction, w);
    T dotvv = dot(v, v);
    T dotvw = dot(v, w);
    T den = dotuu * dotvv - dotuv * dotuv;

    T t0 = 0;
    T t1 = 0;

    if (den < std::numeric_limits<T>::epsilon()) {
      t1 = dotuv > dotvv ? dotuw / dotuv : dotvw / dotvv;
    } else {
      t0 = (dotuv * dotvw - dotvv * dotuw) / den;
      t1 = (dotuu * dotvw - dotuv * dotuw) / den;
    }

    Vec3<T> p = at(t0);
    Vec3<T> q = point1 + t1 * v;

    if (ray_point != 0) {
      *ray_point = p;
    }
    if (line_point != 0) {
      *line_point = q;
    }

    return len(p - q);
  }

  //! Returns the closest distance between this ray and the specified line
  //! segment. Optionally returns the point on this ray and the line that
  //! satisfy this distance.

  // http://softsurfer.com/Archive/algorithm_0106/algorithm_0106.htm

  T intersect_segment(const Vec3<T>& point1,
                      const Vec3<T>& point2,
                      Vec3<T>* ray_point = 0,
                      Vec3<T>* segment_point = 0) const
  {
    Vec3<T> v = point2 - point1;
    Vec3<T> w = origin - point1;

    T dotuu = dot(direction, direction);
    T dotuv = dot(direction, v);
    T dotuw = dot(direction, w);
    T dotvv = dot(v, v);
    T dotvw = dot(v, w);
    T den = dotuu * dotvv - dotuv * dotuv;

    T t0 = 0;
    T t1 = 0;

    if (den < std::numeric_limits<T>::epsilon()) {
      t1 = dotuv > dotvv ? dotuw / dotuv : dotvw / dotvv;
    } else {
      t0 = (dotuv * dotvw - dotvv * dotuw) / den;
      t1 = (dotuu * dotvw - dotuv * dotuw) / den;
    }

    // Clamp to the segment

    if (t1 < 0.0f) {
      t1 = 0.0f;
    }
    if (t1 > 1.0f) {
      t1 = 1.0f;
    }

    Vec3<T> p = at(t0);
    Vec3<T> q = point1 + t1 * v;

    if (ray_point != 0) {
      *ray_point = p;
    }
    if (segment_point != 0) {
      *segment_point = q;
    }

    return len(p - q);
  }

  //! Returns the closest distance between this ray and the specified circle.
  //! Optionally returns the point on this ray and the circle that satisfy
  //! this distance.

  T intersect_circle(const Vec3<T>& center,
                     const Vec3<T>& normal,
                     const T& radius,
                     Vec3<T>* ray_point = 0,
                     Vec3<T>* circle_point = 0) const
  {
    double min = normal.x; int i = 0;
    if (normal.y < min) { min = normal.y; i = 1; }
    if (normal.z < min) { min = normal.z; i = 2; }
    Vec3d rand(i == 0 ? 1 : 0, i == 1 ? 1 : 0, i == 2 ? 1 : 0);

    Vec3d right = unit(cross(normal, rand));
    Vec3d up = unit(cross(normal, right));

    return intersect_arc(center,
                         right,
                         up,
                         radius,
                         0,
                         2 * M_PI,
                         ray_point,
                         circle_point);
  }

  //! Returns the closest distance between this ray and the specified arc.
  //! Optionally returns the point on this ray and the arc that satisfy this
  //! distance.

  T intersect_arc(const Vec3<T>& center,
                  const Vec3<T>& right,
                  const Vec3<T>& up,
                  const T& radius,
                  const T& alpha,
                  const T& beta,
                  Vec3<T>* ray_point = 0,
                  Vec3<T>* arc_point = 0) const
  {
    const int slices = (int)(abs(alpha - beta) * 32 / M_PI + 1);

    T min_dist = std::numeric_limits<float>::max();
    Vec3<T> min_ray_point;
    Vec3<T> min_arc_point;

    for (int i = 0; i < slices; ++i) {
      T theta1 = alpha + i * (beta - alpha) / slices;
      T theta2 = alpha + (i + 1) * (beta - alpha) / slices;

      Vec3d point1 = center + radius * (cos(theta1) * right + sin(theta1) * up);
      Vec3d point2 = center + radius * (cos(theta2) * right + sin(theta2) * up);

      T dist = intersect_segment(point1, point2, ray_point, arc_point);

      if (dist < min_dist) {
        min_dist = dist;
        if (ray_point != 0) {
          min_ray_point = *ray_point;
        }
        if (arc_point != 0) {
          min_arc_point = *arc_point;
        }
      }
    }

    if (ray_point != 0) {
      *ray_point = min_ray_point;
    }
    if (arc_point != 0) {
      *arc_point = min_arc_point;
    }

    return min_dist;
  }

  //! Data members

  Vec3<T> origin;

  Vec3<T> direction;
};

#endif // RAY_HPP
