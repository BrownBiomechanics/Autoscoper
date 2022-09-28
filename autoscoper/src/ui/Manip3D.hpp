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

/// \file Manip3D.h
/// \author Benjamin Knorlein, Andy Loomis



#ifndef MANIP_3D_HPP
#define MANIP_3D_HPP

#include <Matrix.hpp>

using namespace std;

template <class T> class Ray;

//! This class is a provides a 3-dimensional manipulator for translating and
//! rotating objects in an OpenGL scene.

class Manip3D
{
public:

    // Enums specifying the two modes of the manipulator

    enum Mode { TRANSLATION, ROTATION };

    enum Selection { NONE, X, Y, Z, VIEW_PLANE, FREE_ROTATION };


    //! Constructs a default manipulator.

    Manip3D();

    //! Destructor.

    virtual ~Manip3D() {}

  void set_movePivot(bool movePivot){ movePivot_ = movePivot;}

  bool get_movePivot() const { return movePivot_; }

  void set_pivotSize(double pivotSize){ pivotSize_ = pivotSize;}

  double get_pivotSize() const { return pivotSize_; }

    //! Sets the manipulator to be visible or not. If the manipulator is not
    //! visible, then it will not respond to mouse events.

    void set_visible(bool is_visible);

    //! Returns true if the manipulator is visible and false otherwise.

    bool is_visible() const { return is_visible_; }

    //! Sets the manipulator to be locked or not. A locked manipulator is still
    //! visible, but will not respons to any user interaction.

    void set_locked(bool is_locked);

    //! Returns true if the manipulator is locked and false otherwise.

    bool is_locked() const { return is_locked_; }

    void set_size(double size);

    double size() const { return size_; }

    //! Sets the mode of the manipulator. Valid modes are either
    //! Manip3D::TRANSLATION or Manip3D::ROTATION

    void set_mode(Mode mode);

    //! Returns the Mode of the manipulator.

    int mode() const { return mode_; }

    //! Returns true if the manipulator has been selected

    bool is_selected() const { return selection_ != NONE; }

    Selection selection() const { return selection_; }

    //! XXX This is an ugly hack, don't use...

    void set_selection(Selection s) { selection_ = s; }

    //! Sets the modeling transformation for the manipulator. This
    //! transformation determines the location and orientation of the
    //! manipulator in three dimensional space. This method will override any
    //! translations or rotations that have been preformed on the manipulator
    //! through user interaction.

    void set_transform(const Mat4d& transform);

    //! Returns the modeling transformation of the manipulator.

    Mat4d transform() const { return transform1_*transform2_; }

    //! Updates the viewport, projection, and modelview matrices

    void set_view();

    //! Draws the manipulator to the current OpenGL window.

    void draw();

    //! In order for the manipulator to respond to any user interaction, this
    //! function must be called whenever a mouse button is pressed. The
    //! specified window coordinates are assumed to be relative to the window
    //! where the manipulator was last drawn.

    void on_mouse_press(int x, int y, int button=0);

    //! In order for the manipulator to respond to any user interaction, this
    //! function must be called whenever a mouse is dragged across the screen.
    //! In practice this function will be called multiple times after a single
    //! call to on_mouse_press. It will translate or rotate the manipulator
    //! based on the movement of the mouse. The specified window coordinates are
    //! assumed to be relative to the window where the manipulator was last
    //! drawn.

    void on_mouse_move(int x, int y, int button=0);

    //! In order for the manipulator to respond to any user interaction, this
    //! function must be called whenever a mouse button is released. The
    //! specified window coordinates are assumed to be relative to the window
    //! where the manipulator was last drawn.

    void on_mouse_release(int x, int y);

private:

    void draw_axes() const;

    void draw_gimbals() const;

    void select_axis(const Ray<double>& ray);

    void select_gimbal(const Ray<double>& ray);

    void move_axis(const Ray<double>& ray);

    void move_gimbal(const Ray<double>& ray);


    bool is_visible_;

    bool is_locked_;

    Mat4d transform1_;

    Mat4d transform2_;

    double size_;

    double prev_size_;

    Mode mode_;

    int viewport_[4];

    Mat4d projection_;

    Mat4d modelview_;

    Selection selection_;

    Vec3d point1_;

    Vec3d point2_;

  bool movePivot_;

  double pivotSize_;
};

#endif // MANIP_3D_HPP
