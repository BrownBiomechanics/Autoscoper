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

//! \file KeyCurve.hpp
//! \author Andy Loomis

#ifndef KEY_CURVE_HPP
#define KEY_CURVE_HPP

#include <map>

// This class represents a two dimensional curve that smoothly transitions from
// one keyed value to another. Interpolation between keyed values is done using
// biezer curves. The value at each keypoint and the derivative can be setup as
// desired.

class KeyCurve
{
private:

    class Key;

    typedef std::map<int,Key> key_map;

public:

    enum Tangent_type { SMOOTH };

	enum Curve_type { X_CURVE, Y_CURVE, Z_CURVE, YAW_CURVE, PITCH_CURVE, ROLL_CURVE };

    // Typedefs

    typedef key_map::iterator iterator;

    typedef key_map::const_iterator const_iterator;

    // COnstructors and Destructor

    KeyCurve() {}

    ~KeyCurve() {}

	KeyCurve(Curve_type _type) { type = _type; }

	Curve_type type;

    // Removes all keyframes from the curve

    void clear() { keys.clear(); }

    // Returns true if there are no keyframes

    bool empty() const { return keys.empty(); }

    // Returns the number of keyframes

    int size() const { return keys.size(); }

    // Insertion and deletion

    void insert(int time);

    void insert(int time, float value);

    void erase(iterator position);

    // Iterators

    iterator begin() { return keys.begin(); }

    const_iterator begin() const { return keys.begin(); }

    iterator end() { return keys.end(); }

    const_iterator end() const { return keys.end(); }

    // Search

    iterator find(int time) { return keys.find(time); }

    const_iterator find(int time) const { return keys.find(time); }

    // Accessors and mutators

    int time(const_iterator position) const { return position->first; }

    iterator set_time(iterator position, int time);

    float value(const_iterator position) const
    {
        return position->second.value;
    }

    void set_value(iterator position, float value)
    {
        position->second.value = value;
        key_changed(position);
    }

    Tangent_type in_tangent_type(const_iterator position) const
    {
        return position->second.in_tangent_type;
    }

    void set_in_tangent_type(iterator position, Tangent_type type)
    {
        position->second.in_tangent_type = type;
        position->second.in_tangent_lock = false;
        key_changed(position);
    }

    Tangent_type out_tangent_type(const_iterator position) const
    {
        return position->second.out_tangent_type;
    }

    void set_out_tangent_type(iterator position, Tangent_type type)
    {
        position->second.out_tangent_type = type;
        position->second.out_tangent_lock = false;
        key_changed(position);
    }

    bool bind_tangents(iterator position) const
    {
        return position->second.bind_tangents;
    }

    void set_bind_tangents(iterator position, bool bind)
    {
        position->second.bind_tangents = bind;
    }

    bool in_tangent_lock(const_iterator position) const
    {
        return position->second.in_tangent_lock;
    }

    void set_in_tangent_lock(iterator position, bool lock)
    {
        position->second.in_tangent_lock = lock;
    }

    bool out_tangent_lock(const_iterator position) const
    {
        return position->second.out_tangent_lock;
    }

    void set_out_tangent_lock(iterator position, bool lock)
    {
        position->second.out_tangent_lock = lock;
    }

    float in_tangent(const_iterator position) const
    {
        return position->second.in_tangent;
    }

    void set_in_tangent(iterator position, float tangent)
    {
        position->second.in_tangent = tangent;
        set_in_tangent_lock(position,true);
        if (position->second.bind_tangents) {
            position->second.out_tangent = tangent;
            set_out_tangent_lock(position,true);
        }
        key_changed(position);
    }

    float out_tangent(const_iterator position) const
    {
        return position->second.out_tangent;
    }

    void set_out_tangent(iterator position, float tangent)
    {
        position->second.out_tangent = tangent;
        set_out_tangent_lock(position,true);
        if (position->second.bind_tangents) {
            position->second.in_tangent = tangent;
            set_in_tangent_lock(position,true);
        }
        key_changed(position);
    }

    // Interpolation

    float evaluate(float time) const { return (*this)(time); }

    float operator()(float time) const;

private:

    // Internal structure to store keyframes and Bezier curves

    class Key
    {
        friend class KeyCurve;

        float value;
        Tangent_type in_tangent_type;
        Tangent_type out_tangent_type;
        bool bind_tangents;
        bool in_tangent_lock;
        bool out_tangent_lock;
        float in_tangent;
        float out_tangent;
        float a,b,c,d;
    };

    // Called whenever a keyframe is modified

    void key_changed(iterator position);

    // Updates the Bezier curve

    void update_curve(iterator position);

    key_map keys;
};

#endif // KEY_CURVE_HPP
