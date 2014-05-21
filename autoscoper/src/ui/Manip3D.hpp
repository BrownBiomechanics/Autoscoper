//! \file Manip3D.hpp
//! \author Andy Loomis (aloomis)

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
