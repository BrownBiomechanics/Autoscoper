// ----------------------------------
// Copyright (c) 2011, Brown University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// (1) Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// (2) Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// (3) Neither the name of Brown University nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY BROWN UNIVERSITY “AS IS” WITH NO
// WARRANTIES OR REPRESENTATIONS OF ANY KIND WHATSOEVER EITHER EXPRESS OR
// IMPLIED, INCLUDING WITHOUT LIMITATION ANY WARRANTY OF DESIGN OR
// MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, EACH OF WHICH ARE
// SPECIFICALLY DISCLAIMED, NOR ANY WARRANTY OR REPRESENTATIONS THAT THE
// SOFTWARE IS ERROR FREE OR THAT THE SOFTWARE WILL NOT INFRINGE ANY
// PATENT, COPYRIGHT, TRADEMARK, OR OTHER THIRD PARTY PROPRIETARY RIGHTS.
// IN NO EVENT SHALL BROWN UNIVERSITY BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY OR CAUSE OF ACTION, WHETHER IN CONTRACT,
// STRICT LIABILITY, TORT, NEGLIGENCE OR OTHERWISE, ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE. ANY RECIPIENT OR USER OF THIS SOFTWARE ACKNOWLEDGES THE
// FOREGOING, AND ACCEPTS ALL RISKS AND LIABILITIES THAT MAY ARISE FROM
// THEIR USE OF THE SOFTWARE.
// ---------------------------------

/// \file Manip3D.cpp
/// \author Benjamin Knorlein, Andy Loomis

#ifdef _WIN32
#  include <windows.h>
#  undef max
#  include <algorithm>
#endif

#include <cmath>
#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif
#include <iostream>

#ifdef __APPLE__
#  include <OpenGL/gl.h>
#  include <OpenGL/glext.h>
#  include <OpenGL/glu.h>
#else
#  include <GL/glew.h> // For windows to support glMultTransposeMatrix
#  include <GL/gl.h>
#  include <GL/glu.h>
#endif

#include "Manip3D.hpp"

#include <Ray.hpp>
#include <Matrix.hpp>
#include <Vector.hpp>

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~// Constants //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// One quarter coverage stipple pattern

static const GLubyte pattern[128] = {
  // clang-format off
  0xaa, 0xaa, 0xaa, 0xaa, 0x00, 0x00, 0x00, 0x00, 0xaa, 0xaa, 0xaa, 0xaa,
  0x00, 0x00, 0x00, 0x00, 0xaa, 0xaa, 0xaa, 0xaa, 0x00, 0x00, 0x00, 0x00,
  0xaa, 0xaa, 0xaa, 0xaa, 0x00, 0x00, 0x00, 0x00, 0xaa, 0xaa, 0xaa, 0xaa,
  0x00, 0x00, 0x00, 0x00, 0xaa, 0xaa, 0xaa, 0xaa, 0x00, 0x00, 0x00, 0x00,
  0xaa, 0xaa, 0xaa, 0xaa, 0x00, 0x00, 0x00, 0x00, 0xaa, 0xaa, 0xaa, 0xaa,
  0x00, 0x00, 0x00, 0x00, 0xaa, 0xaa, 0xaa, 0xaa, 0x00, 0x00, 0x00, 0x00,
  0xaa, 0xaa, 0xaa, 0xaa, 0x00, 0x00, 0x00, 0x00, 0xaa, 0xaa, 0xaa, 0xaa,
  0x00, 0x00, 0x00, 0x00, 0xaa, 0xaa, 0xaa, 0xaa, 0x00, 0x00, 0x00, 0x00,
  0xaa, 0xaa, 0xaa, 0xaa, 0x00, 0x00, 0x00, 0x00, 0xaa, 0xaa, 0xaa, 0xaa,
  0x00, 0x00, 0x00, 0x00, 0xaa, 0xaa, 0xaa, 0xaa, 0x00, 0x00, 0x00, 0x00,
  0xaa, 0xaa, 0xaa, 0xaa, 0x00, 0x00, 0x00, 0x00
  // clang-format on
};

// Colors used by the manipulator

static const GLdouble red[4] = { 1.0, 0.0, 0.0, 1.0 };
static const GLdouble green[4] = { 0.0, 1.0, 0.0, 1.0 };
static const GLdouble blue[4] = { 0.0, 0.0, 1.0, 1.0 };
static const GLdouble cyan[4] = { 0.0, 1.0, 1.0, 1.0 };
static const GLdouble yellow[4] = { 1.0, 1.0, 0.0, 1.0 };
static const GLdouble grey[4] = { 0.5, 0.5, 0.5, 1.0 };

// Translation arrow variables

static const double arrow_base = 0.05;
static const double arrow_height = 0.2;
static const int arrow_slices = 20;
static const int arrow_stacks = 20;

// Size of the manipulator

static double tol = 4.0;
static double min_view_angle = 0.2;
static double rads_per_pixel = 0.0;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~// Helper Functions //~~~~~~~~~~~~~~~~~~~~~~~~~~~//

static Ray<double> create_ray(int x, int y, const double* model, const double* proj, const int* view);

static void draw_cone(float base, float height, int slices, int stacks);

static double safe_acos(double x)
{
  static const float min_x = -1.0 + std::numeric_limits<double>::epsilon();
  static const float max_x = 1.0 - std::numeric_limits<double>::epsilon();

  if (x > max_x)
    x = max_x;
  else if (x < min_x)
    x = min_x;

  return acos(x);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~// Class Functions //~~~~~~~~~~~~~~~~~~~~~~~~~~~//

Manip3D::Manip3D()

  : movePivot_(false)
  , pivotSize_(0.25f)
  , is_visible_(true)
  , is_locked_(false)
  , transform1_(Mat4d::eye())
  , transform2_(Mat4d::eye())
  , size_(1.0)
  , prev_size_(1.0)
  , mode_(TRANSLATION)
  , projection_(Mat4d::eye())
  , modelview_(Mat4d::eye())
  , selection_(NONE)
  , point1_(Vec3d::zero())
  , point2_(Vec3d::zero())
  , is_thick_lines_mode_(false)
{
  viewport_[0] = 0;
  viewport_[1] = 0;
  viewport_[2] = 1;
  viewport_[3] = 1;
}

void Manip3D::set_visible(bool is_visible)
{
  is_visible_ = is_visible;

  // Freeze the current transformation

  transform1_ = transform1_ * transform2_;
  transform2_ = Mat4d::eye();

  // Clear the current selection

  selection_ = NONE;
}

void Manip3D::set_locked(bool is_locked)
{
  is_locked_ = is_locked;

  // Freeze the current transformation

  transform1_ = transform1_ * transform2_;
  transform2_ = Mat4d::eye();
}

void Manip3D::set_mode(Mode mode)
{
  mode_ = mode;

  // Freeze the current transformation

  transform1_ = transform1_ * transform2_;
  transform2_ = Mat4d::eye();

  // Clear the current selection

  selection_ = NONE;
}

void Manip3D::set_size(double size)
{
  size_ = size;
}

void Manip3D::set_transform(const Mat4d& transform)
{
  transform1_ = transform;
  transform2_ = Mat4d::eye();

  // Clear the current selection

  selection_ = NONE;
}

void Manip3D::set_view()
{
  // Add the manipulator's transformation to the stack

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glMultTransposeMatrixd(transform1_ * transform2_);

  // Save the viewport, projection, and modelview

  glGetIntegerv(GL_VIEWPORT, viewport_);
  glGetDoublev(GL_PROJECTION_MATRIX, projection_);
  glGetDoublev(GL_MODELVIEW_MATRIX, modelview_);

  glPopMatrix();
}

void Manip3D::draw()
{
  if (!is_visible_) {
    return;
  }

  // The manipulator requires the depth buffer to be active
  // TODO: Disable textures and lighting???

  glPushAttrib(GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);

  // Add the manipulator's transformation to the stack
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glMultTransposeMatrixd(transform1_ * transform2_);

  // Draw either axes or gimbals

  switch (mode_) {
    case TRANSLATION:
      draw_axes();
      break;
    case ROTATION:
      draw_gimbals();
      break;
  }

  // Return to the previous state
  glPopMatrix();
  glPopAttrib();
}

void Manip3D::on_mouse_press(int x, int y, int button)
{
  if (!is_visible_) {
    return;
  }

  selection_ = NONE;

  Ray<double> ray = create_ray(x, y, modelview_, projection_, viewport_);

  Ray<double> ray1 = create_ray(x + 1, y, modelview_, projection_, viewport_);
  Ray<double> ray2 = create_ray(x, y + 1, modelview_, projection_, viewport_);

  rads_per_pixel = std::max(safe_acos(dot(ray.direction, ray1.direction) / len(ray.direction) / len(ray1.direction)),
                            safe_acos(dot(ray.direction, ray2.direction) / len(ray.direction) / len(ray2.direction)));

  switch (mode_) {
    default:
    case TRANSLATION:
      select_axis(ray);
      break;
    case ROTATION:
      select_gimbal(ray);
      break;
  }

  if (selection_ != NONE) {
    prev_size_ = size_;
  }

  point2_ = point1_;
}

void Manip3D::on_mouse_move(int x, int y, int button)
{
  if (!is_visible_ || is_locked_ || selection_ == NONE) {
    return;
  }

  Ray<double> ray = create_ray(x, y, modelview_, projection_, viewport_);

  switch (mode_) {
    default:
    case TRANSLATION:
      move_axis(ray);
      break;
    case ROTATION:
      move_gimbal(ray);
      break;
  }
}

void Manip3D::on_mouse_release(int x, int y)
{
  selection_ = NONE;
  transform1_ = transform1_ * transform2_;
  transform2_ = Mat4d::eye();
}

void Manip3D::setThickLinesMode(bool checked)
{
  is_thick_lines_mode_ = checked;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~// Private Functions //~~~~~~~~~~~~~~~~~~~~~~~~~~//

void Manip3D::draw_axes() const
{
  Mat4d modelview;
  glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
  Mat4d inv_trans_modelview = inv(trans(modelview));

  // Calculate the eye position in object space

  Vec3d eye = col(transform2_ * inv_trans_modelview, 3);

  // Calculate the angle between the eye position and the three axes

  Vec3d view_angle(atan(sqrt(eye.y * eye.y + eye.z * eye.z) / abs(eye.x)),
                   atan(sqrt(eye.x * eye.x + eye.z * eye.z) / abs(eye.y)),
                   atan(sqrt(eye.x * eye.x + eye.y * eye.y) / abs(eye.z)));

  // Draw a square in the view plane

  glPushAttrib(GL_LIGHTING_BIT);
  glDisable(GL_LIGHTING);

  if (selection_ == NONE || selection_ == VIEW_PLANE) {
    if (selection_ == NONE) {
      glColor4dv(cyan);
    } else {
      glColor4dv(yellow);
    }

    // Calculate the axes parallel to the view plane

    Vec3d u = unit(Vec3d(col(inv_trans_modelview, 0)));
    Vec3d v = unit(Vec3d(col(inv_trans_modelview, 1)));

    glBegin(GL_LINE_LOOP);
    glVertex3dv(0.1 * size_ * (u + v));
    glVertex3dv(0.1 * size_ * (-u + v));
    glVertex3dv(0.1 * size_ * (-u - v));
    glVertex3dv(0.1 * size_ * (u - v));
    glEnd();
  }

  glPopAttrib();

  // Draw x-axis if the view angle is large enough

  if (this->is_thick_lines_mode_)
    glLineWidth(5.0); // For thick lines mode

  if (view_angle.x > min_view_angle) {
    if (selection_ == NONE || selection_ == X || selection_ == VIEW_PLANE) {
      if (selection_ == X) {
        glColor4dv(yellow);
      } else {
        glColor4dv(red);
      }

      // Draw the axis

      glPushAttrib(GL_LIGHTING_BIT);
      glDisable(GL_LIGHTING);
      glBegin(GL_LINES);
      glVertex3d(0.0, 0.0, 0.0);
      glVertex3d(size_ - 5, 0.0, 0.0);
      glEnd();
      glPopAttrib();

      // Draw the arrow head

      if (selection_ == NONE || selection_ == X) {

        // Calculate what angle to rotate the cone so that it always
        // appears as if the light source is at the eye position

        double angle = 180.0 * atan2(eye.y, -eye.z) / M_PI;

        glPushMatrix();
        glTranslated(size_ - size_ * arrow_height, 0.0, 0.0);
        glRotated(90.0, 0.0, 1.0, 0.0);
        glRotated(angle, 0.0, 0.0, 1.0);
        draw_cone(size_ * arrow_base, size_ * arrow_height, arrow_slices, arrow_stacks);
        glPopMatrix();
      }
    }
  }

  // Draw y-axis if the view angle is large enough

  if (view_angle.y > min_view_angle) {
    if (selection_ == NONE || selection_ == Y || selection_ == VIEW_PLANE) {
      if (selection_ == Y) {
        glColor4dv(yellow);
      } else {
        glColor4dv(green);
      }

      // Draw the axis

      glPushAttrib(GL_LIGHTING_BIT);
      glDisable(GL_LIGHTING);
      glBegin(GL_LINES);
      glVertex3d(0.0, 0.0, 0.0);
      glVertex3d(0.0, size_ - 5, 0.0);
      glEnd();
      glPopAttrib();

      // Draw the arrow head

      if (selection_ == NONE || selection_ == Y) {

        // Calculate what angle to rotate the cone so that it always
        // appears as if the light source is at the eye position

        double angle = 180.0 * atan2(eye.x, eye.z) / M_PI - 90.0;

        glPushMatrix();
        glTranslated(0.0, size_ - size_ * arrow_height, 0.0);
        glRotated(-90.0, 1.0, 0.0, 0.0);
        glRotated(angle, 0.0, 0.0, 1.0);
        draw_cone(size_ * arrow_base, size_ * arrow_height, arrow_slices, arrow_stacks);
        glPopMatrix();
      }
    }
  }

  // Draw z-axis if the view angle is large enough

  if (view_angle.z > min_view_angle) {
    if (selection_ == NONE || selection_ == Z || selection_ == VIEW_PLANE) {
      if (selection_ == Z) {
        glColor4dv(yellow);
      } else {
        glColor4dv(blue);
      }

      glPushAttrib(GL_LIGHTING_BIT);
      glDisable(GL_LIGHTING);
      glBegin(GL_LINES);
      glVertex3d(0.0, 0.0, 0.0);
      glVertex3d(0.0, 0.0, size_ - 5);
      glEnd();
      glPopAttrib();

      // Draw the arrow head

      if (selection_ == NONE || selection_ == Z) {

        // Calculate what angle to rotate the cone so that it always
        // appears as if the light source is at the eye position

        double angle = 90.0 - 180.0 * atan2(eye.x, eye.y) / M_PI;

        glPushMatrix();
        glTranslated(0.0, 0.0, size_ - size_ * arrow_height);
        glRotated(angle, 0.0, 0.0, 1.0);
        draw_cone(size_ * arrow_base, size_ * arrow_height, arrow_slices, arrow_stacks);
        glPopMatrix();
      }
    }
  }

  // If the manipulator is moving draw its previous position

  glPushAttrib(GL_LIGHTING_BIT);
  glDisable(GL_LIGHTING);

  if (selection_ != NONE) {

    glPointSize(5);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glMultTransposeMatrixd(inv(transform2_));

    glColor4dv(grey);

    if (selection_ == X || (selection_ == VIEW_PLANE && view_angle.x > min_view_angle)) {
      glBegin(GL_LINES);
      glVertex3d(0.0, 0.0, 0.0);
      glVertex3d(prev_size_, 0.0, 0.0);
      glEnd();

      glBegin(GL_POINTS);
      glVertex3d(0.0, 0.0, 0.0);
      glVertex3d(prev_size_, 0.0, 0.0);
      glEnd();
    }
    if (selection_ == Y || (selection_ == VIEW_PLANE && view_angle.y > min_view_angle)) {
      glBegin(GL_LINES);
      glVertex3d(0.0, 0.0, 0.0);
      glVertex3d(0.0, prev_size_, 0.0);
      glEnd();

      glBegin(GL_POINTS);
      glVertex3d(0.0, 0.0, 0.0);
      glVertex3d(0.0, prev_size_, 0.0);
      glEnd();
    }
    if (selection_ == Z || (selection_ == VIEW_PLANE && view_angle.z > min_view_angle)) {
      glBegin(GL_LINES);
      glVertex3d(0.0, 0.0, 0.0);
      glVertex3d(0.0, 0.0, prev_size_);
      glEnd();

      glBegin(GL_POINTS);
      glVertex3d(0.0, 0.0, 0.0);
      glVertex3d(0.0, 0.0, prev_size_);
      glEnd();
    }

    glPopMatrix();
  }
  glPopAttrib();
}

void Manip3D::draw_gimbals() const
{
  glPushAttrib(GL_LIGHTING_BIT);
  glDisable(GL_LIGHTING);

  Mat4d modelview;
  glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
  Mat4d inv_trans_modelview = inv(trans(modelview));

  // Calculate the eye position in object space

  Vec3d eye = col(inv_trans_modelview, 3);

  // Calculate the hemisphere of each gimbal that is visible

  Vec3d alpha = -Vec3d(atan2(eye.y, eye.z), atan2(eye.x, eye.z), atan2(eye.x, eye.y));

  // Draw the view plane gimbal. The view plane gimbal is not actually drawn
  // in the view plane, but rather perpendicular to the vector from the eye
  // point to the center of the gimbal. This is to compensate for perspective
  // distortions caused when the manipulator is not near the center of the
  // view frustum.

  Vec3d v = unit(cross(eye, col(inv_trans_modelview, 0)));
  Vec3d u = unit(cross(v, eye));

  if (selection_ == VIEW_PLANE) {
    glColor4dv(yellow);
  } else {
    glColor4dv(cyan);
  }
  glBegin(GL_LINE_LOOP);
  for (int i = 0; i < 64; ++i) {
    float theta = 2.0f * M_PI * i / (float)64;
    glVertex3dv(1.1 * size_ * (cos(theta) * u + sin(theta) * v));
  }
  glEnd();

  // Draw a silhouette of the gimbal

  glColor4dv(grey);
  glBegin(GL_LINE_LOOP);
  for (int i = 0; i < 64; ++i) {
    float theta = 2.0f * M_PI * i / (float)64;
    glVertex3dv(size_ * (cos(theta) * u + sin(theta) * v));
  }
  glEnd();

  if (this->is_thick_lines_mode_)
    glLineWidth(5.0); // For thick lines mode

  // Draw x-axis gimbal

  if (selection_ == X) {
    glColor4dv(yellow);
  } else {
    glColor4dv(red);
  }
  glBegin(GL_LINE_LOOP);
  for (int i = 0; i < 64; ++i) {
    float theta = alpha.x + M_PI * i / (float)(32 - 1);
    glVertex3d(0.0, size_ * cos(theta), size_ * sin(theta));
  }
  glEnd();

  // Draw y-axis gimbal

  if (selection_ == Y) {
    glColor4dv(yellow);
  } else {
    glColor4dv(green);
  }
  glBegin(GL_LINE_LOOP);
  for (int i = 0; i < 64; ++i) {
    float theta = alpha.y + M_PI * i / (float)(32 - 1);
    glVertex3d(size_ * cos(theta), 0.0, size_ * sin(theta));
  }
  glEnd();

  // Draw z-axis gimbal

  if (selection_ == Z) {
    glColor4dv(yellow);
  } else {
    glColor4dv(blue);
  }
  glBegin(GL_LINE_LOOP);
  for (int i = 0; i < 64; ++i) {
    float theta = alpha.z + M_PI * i / (float)(32 - 1);
    glVertex3d(size_ * cos(theta), size_ * sin(theta), 0.0);
  }
  glEnd();

  // If the gimbals are moving draw the last position

  if (selection_ != NONE && selection_ != FREE_ROTATION) {

    Vec3d point1 = mul_pt(inv(transform2_), point1_);
    Vec3d point2 = mul_pt(inv(transform2_), point2_);

    float theta = safe_acos(dot(point1, point2) / len(point1) / len(point2));
    Vec3d axis = cross(point1, point2);

    glPointSize(5.0);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadMatrixd(modelview);

    glColor4dv(grey);

    glBegin(GL_LINES);
    glVertex3d(0, 0, 0);
    glVertex3dv(point1);
    glVertex3d(0, 0, 0);
    glVertex3dv(point2);
    glEnd();

    glBegin(GL_POINTS);
    glVertex3dv(point1);
    glVertex3dv(point2);
    glVertex3d(0, 0, 0);
    glEnd();

    glPushAttrib(GL_ENABLE_BIT);
    glEnable(GL_POLYGON_STIPPLE);
    glPolygonStipple(pattern);

    glBegin(GL_TRIANGLE_FAN);
    glVertex3d(0, 0, 0);
    for (int i = 0; i < 32; i++) {
      double angle = i * theta / 31;
      Vec3d p = Mat3d::rot(angle, axis) * point1;
      glVertex3dv(p);
    }
    glEnd();

    glPopAttrib();

    glPopMatrix();
  }
  glPopAttrib();
}

void Manip3D::select_axis(const Ray<double>& ray)
{
  // Test for selection of each of the axes that are visible

  Vec3d eye = col(inv(trans(modelview_)), 3);

  // Calculate the angle between the eye position and the three axes

  Vec3d view_angle(atan(sqrt(eye.y * eye.y + eye.z * eye.z) / abs(eye.x)),
                   atan(sqrt(eye.x * eye.x + eye.z * eye.z) / abs(eye.y)),
                   atan(sqrt(eye.x * eye.x + eye.y * eye.y) / abs(eye.z)));

  // Test for selection of the in plane translation handle.

  Vec3d p, q;
  double dist = ray.intersect_point(Vec3d(0, 0, 0), &p);
  if (dist < 0.1f * size_) {
    point1_ = p;
    selection_ = VIEW_PLANE;
    return;
  }

  // We calculate the distance between the ray and each of the axis in pixel
  // units. The 'tol' variable specifies the pixel tolerance criteria for a
  // selection. Because multiple axes can be selected at once, we select the
  // one closest to the origin of the ray.

  double min_dist = std::numeric_limits<double>::max();

  if (view_angle.x > min_view_angle) {

    ray.intersect_segment(Vec3d::zero(), size_ * Vec3d::unit_x(), &p, &q);
    double theta = safe_acos(dot(unit(p - eye), unit(q - eye)));
    double dist = len(p - ray.origin);

    if (theta / rads_per_pixel < tol && dist < min_dist) {
      min_dist = dist;
      point1_ = q;
      selection_ = X;
    }
  }

  if (view_angle.y > min_view_angle) {

    ray.intersect_segment(Vec3d::zero(), size_ * Vec3d::unit_y(), &p, &q);
    double theta = safe_acos(dot(unit(p - eye), unit(q - eye)));
    double dist = len(p - ray.origin);

    if (theta / rads_per_pixel < tol && dist < min_dist) {
      min_dist = dist;
      point1_ = q;
      selection_ = Y;
    }
  }

  if (view_angle.z > min_view_angle) {

    ray.intersect_segment(Vec3d::zero(), size_ * Vec3d::unit_z(), &p, &q);
    double theta = safe_acos(dot(unit(p - eye), unit(q - eye)));
    double dist = len(p - ray.origin);

    if (theta / rads_per_pixel < tol && dist < min_dist) {
      min_dist = dist;
      point1_ = q;
      selection_ = Z;
    }
  }
}

void Manip3D::select_gimbal(const Ray<double>& ray)
{
  // Calculate the eye position

  Vec3d eye = col(inv(trans(modelview_)), 3);

  // Determine if the ray selects the view plane gimbal

  Vec3d p, q;
  ray.intersect_circle(Vec3d::zero(), unit(eye), 1.1 * size_, &p, &q);
  double theta = safe_acos(dot(unit(p - eye), unit(q - eye)));

  // If it is selected then we are done

  if (theta / rads_per_pixel < tol) {
    point1_ = q;
    selection_ = VIEW_PLANE;
    return;
  }

  // Determine if the ray selects the gimbal sphere

  ray.intersect_sphere(Vec3d::zero(), size_, &p, &q);
  theta = safe_acos(dot(unit(p - eye), unit(q - eye)));

  // If it does, we must test for intersection with each gimbal

  if (theta / rads_per_pixel < tol) {
    point1_ = q;
    selection_ = FREE_ROTATION;

    // Calculate the hemisphere of each gimbal that is visible

    Vec3d alpha = -Vec3d(atan2(eye.y, eye.z), atan2(eye.x, eye.z), atan2(eye.x, eye.y));

    double min_dist = std::numeric_limits<double>::max();

    // Determine if there is a selection of the x-axis gimbal

    ray.intersect_arc(Vec3d::zero(), Vec3d::unit_y(), Vec3d::unit_z(), size_, alpha.x, alpha.x + M_PI, &p, &q);

    theta = safe_acos(dot(unit(p - eye), unit(q - eye)));
    double dist = len(p - ray.origin);

    if (theta / rads_per_pixel < tol && dist < min_dist) {
      min_dist = dist;
      point1_ = q;
      selection_ = X;
    }

    // Determine if there is a selection of the y-axis gimbal

    ray.intersect_arc(Vec3d::zero(), Vec3d::unit_x(), Vec3d::unit_z(), size_, alpha.y, alpha.y + M_PI, &p, &q);

    theta = safe_acos(dot(unit(p - eye), unit(q - eye)));
    dist = len(p - ray.origin);

    if (theta / rads_per_pixel < tol && dist < min_dist) {
      min_dist = dist;
      point1_ = q;
      selection_ = Y;
    }

    // Determine if there is a selection of the z-axis gimbal

    ray.intersect_arc(Vec3d::zero(), Vec3d::unit_x(), Vec3d::unit_y(), size_, alpha.z, alpha.z + M_PI, &p, &q);

    theta = safe_acos(dot(unit(p - eye), unit(q - eye)));
    dist = len(p - ray.origin);

    if (theta / rads_per_pixel < tol && dist < min_dist) {
      min_dist = dist;
      point1_ = q;
      selection_ = Z;
    }
  }
}

void Manip3D::move_axis(const Ray<double>& ray)
{
  switch (selection_) {
    default:
    case NONE:
      break;
    case X:
      ray.intersect_line(Vec3d::zero(), Vec3d::unit_x(), 0, &point2_);
      break;
    case Y:
      ray.intersect_line(Vec3d::zero(), Vec3d::unit_y(), 0, &point2_);
      break;
    case Z:
      ray.intersect_line(Vec3d::zero(), Vec3d::unit_z(), 0, &point2_);
      break;
    case VIEW_PLANE: {
      Vec3d normal = unit(Vec3d(col(transform2_ * inv(trans(modelview_)), 2)));
      ray.intersect_plane(Vec3d::zero(), normal, &point2_);
    } break;
  }

  transform2_ = Mat4d(Mat3d::eye(), point2_ - point1_);
}

void Manip3D::move_gimbal(const Ray<double>& ray)
{
  Vec3d eye = col(inv(trans(modelview_)), 3);

  // Calculate the hemisphere of each gimbal that is visible

  Vec3d alpha = -Vec3d(atan2(eye.y, eye.z), atan2(eye.x, eye.z), atan2(eye.x, eye.y));

  switch (selection_) {
    default:
    case NONE:
      break;
    case X:
      ray.intersect_circle(Vec3d::zero(), Vec3d::unit_x(), size_, 0, &point2_);
      break;
    case Y:
      ray.intersect_circle(Vec3d::zero(), Vec3d::unit_y(), size_, 0, &point2_);
      break;
    case Z:
      ray.intersect_circle(Vec3d::zero(), Vec3d::unit_z(), size_, 0, &point2_);
      break;
    case VIEW_PLANE:
      ray.intersect_circle(Vec3d::zero(), unit(eye), 1.1 * size_, 0, &point2_);
      break;
    case FREE_ROTATION:
      ray.intersect_sphere(Vec3d::zero(), size_, 0, &point2_);
      break;
  }

  // Calculate the angle and axis of rotation

  double theta = safe_acos(dot(point1_, point2_) / len(point1_) / len(point2_));

  Vec3d axis = theta < 1e-6 ? Vec3d(1.0, 0.0, 0.0) : unit(cross(point1_, point2_));

  transform2_ = Mat4d(Mat3d::rot(theta, axis));
}

// ~~~~~~~~~~~~~~~~~~~~~~// Helper Function Definitions //~~~~~~~~~~~~~~~~~~~~~~//

Ray<double> create_ray(int x, int y, const GLdouble* model, const GLdouble* proj, const GLint* view)
{
  // y = view[3]-y; XXX: Fails if the viewport and screen are different sizes

  Vec3d pnear, pfar;
  gluUnProject(x, y, 0, model, proj, view, &pnear.x, &pnear.y, &pnear.z);
  gluUnProject(x, y, 1, model, proj, view, &pfar.x, &pfar.y, &pfar.z);

  return Ray<double>(col(inv(trans(Mat4d(model))), 3), unit(pfar - pnear));
}

void draw_cone(float base, float height, int slices, int stacks)
{
  for (int i = 0; i < slices; i++) {
    float alpha = 2.0 * M_PI * i / slices;
    float beta = 2.0 * M_PI * (i + 1) / slices;

    float cos_alpha = cos(alpha);
    float sin_alpha = sin(alpha);
    float cos_beta = cos(beta);
    float sin_beta = sin(beta);

    // Draw the base

    glBegin(GL_TRIANGLES);
    glNormal3d(0.0, 0.0, -1.0f);
    glVertex3d(base * cos_alpha, base * sin_alpha, 0.0);
    glVertex3d(0.0, 0.0, 0.0);
    glVertex3d(base * cos_beta, base * sin_beta, 0.0);
    glEnd();

    // Draw the side

    float mag = sqrt(1.0f + base * base / height / height);

    glBegin(GL_QUADS);
    for (int j = 0; j < stacks; j++) {
      float bot = height * j / stacks;
      float top = height * (j + 1) / stacks;
      float bot_base = base * (stacks - j) / stacks;
      float top_base = base * (stacks - j - 1) / stacks;

      glNormal3d(cos_alpha / mag, sin_alpha / mag, base / height / mag);
      glVertex3d(bot_base * cos_alpha, bot_base * sin_alpha, bot);

      glNormal3d(cos_beta / mag, sin_beta / mag, base / height / mag);
      glVertex3d(bot_base * cos_beta, bot_base * sin_beta, bot);

      glNormal3d(cos_beta / mag, sin_beta / mag, base / height / mag);
      glVertex3d(top_base * cos_beta, top_base * sin_beta, top);

      glNormal3d(cos_alpha / mag, sin_alpha / mag, base / height / mag);
      glVertex3d(top_base * cos_alpha, top_base * sin_alpha, top);
    }
    glEnd();
  }
}
