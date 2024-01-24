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
#include "Quaternion.hpp"
#include <iostream>

// This class represents a two dimensional curve that smoothly transitions from
// one keyed value to another. Interpolation between keyed values is done using
// biezer curves. The value at each keypoint and the derivative can be setup as
// desired.

class IKeyCurve // Interface to simplify the use of KeyCurve in the frontend
{
public:
  enum Tangent_type
  {
    SMOOTH
  };

protected:
  class IKey
  {
  public:
    virtual ~IKey() {}
    Tangent_type in_tangent_type;
    Tangent_type out_tangent_type;
    bool bind_tangents;
    bool in_tangent_lock;
    bool out_tangent_lock;
    float in_tangent;
    float out_tangent;
    float a, b, c, d;
  };
  typedef std::map<int, std::shared_ptr<IKey>> key_map;

public:
  enum Curve_type
  {
    X_CURVE,
    Y_CURVE,
    Z_CURVE,
    QUAT_CURVE
  };
  Curve_type type;

  // Typedefs
  typedef typename key_map::iterator iterator;
  typedef typename key_map::const_iterator const_iterator;

  virtual ~IKeyCurve() {}
  virtual void clear() = 0;
  virtual void erase(iterator position) = 0;
  virtual int time(const_iterator position) const = 0;
  virtual void set_bind_tangents(iterator position, bool bind) = 0;
  virtual void set_in_tangent(iterator position, float tangent) = 0;
  virtual float in_tangent(const_iterator position) const = 0;
  virtual float out_tangent(const_iterator position) const = 0;
  virtual void set_out_tangent(iterator position, float tangent) = 0;
};

template <typename T>
class KeyCurve : public IKeyCurve
{
  friend class IKeyCurve;

public:
  // COnstructors and Destructor

  KeyCurve() {}

  ~KeyCurve() {}

  KeyCurve(Curve_type _type) { type = _type; }

  // Removes all keyframes from the curve

  void clear() { keys.clear(); }

  // Returns true if there are no keyframes

  bool empty() const { return keys.empty(); }

  // Returns the number of keyframes

  int size() const { return keys.size(); }

  // Insertion and deletion

  void insert(int time);

  void insert(int time, T value);

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

  T value(const_iterator position) const
  {
    return std::dynamic_pointer_cast<const KeyCurve<T>::Key>(position->second)->value;
  }

  void set_value(iterator position, T value)
  {
    std::dynamic_pointer_cast<KeyCurve<T>::Key>(position->second)->value = value;
    key_changed(position);
  }

  Tangent_type in_tangent_type(const_iterator position) const { return position->second->in_tangent_type; }

  void set_in_tangent_type(iterator position, Tangent_type type)
  {
    position->second->in_tangent_type = type;
    position->second->in_tangent_lock = false;
    key_changed(position);
  }

  Tangent_type out_tangent_type(const_iterator position) const { return position->second->out_tangent_type; }

  void set_out_tangent_type(iterator position, Tangent_type type)
  {
    position->second->out_tangent_type = type;
    position->second->out_tangent_lock = false;
    key_changed(position);
  }

  bool bind_tangents(iterator position) const { return position->second->bind_tangents; }

  void set_bind_tangents(iterator position, bool bind) { position->second->bind_tangents = bind; }

  bool in_tangent_lock(const_iterator position) const { return position->second->in_tangent_lock; }

  void set_in_tangent_lock(iterator position, bool lock) { position->second->in_tangent_lock = lock; }

  bool out_tangent_lock(const_iterator position) const { return position->second->out_tangent_lock; }

  void set_out_tangent_lock(iterator position, bool lock) { position->second->out_tangent_lock = lock; }

  float in_tangent(const_iterator position) const { return position->second->in_tangent; }

  void set_in_tangent(iterator position, float tangent)
  {
    position->second->in_tangent = tangent;
    set_in_tangent_lock(position, true);
    if (position->second->bind_tangents) {
      position->second->out_tangent = tangent;
      set_out_tangent_lock(position, true);
    }
    key_changed(position);
  }

  float out_tangent(const_iterator position) const { return position->second->out_tangent; }

  void set_out_tangent(iterator position, float tangent)
  {
    position->second->out_tangent = tangent;
    set_out_tangent_lock(position, true);
    if (position->second->bind_tangents) {
      position->second->in_tangent = tangent;
      set_in_tangent_lock(position, true);
    }
    key_changed(position);
  }

  // Interpolation

  T evaluate(float time) const { return (*this)(time); }

  T operator()(float time) const;

private:
  // Internal structure to store keyframes and Bezier curves

  class Key : public IKey
  {
    friend class KeyCurve;
    friend class IKey;

    T value;
  };

  // Called whenever a keyframe is modified

  void key_changed(iterator position);

  // Updates the Bezier curve

  void update_curve(iterator position);

  key_map keys;
};

// Need to have the implementation here in the header because of the template

template <typename T>
void KeyCurve<T>::insert(int time)
{
  insert(time, (*this)(time));
}

template <typename T>
void KeyCurve<T>::insert(int time, T value)
{
  std::shared_ptr<Key> key(new Key);
  key->value = T(value); // Copy the value
  key->in_tangent_type = SMOOTH;
  key->out_tangent_type = SMOOTH;
  key->bind_tangents = true;
  key->in_tangent_lock = false;
  key->out_tangent_lock = false;

  iterator position = keys.find(time);
  if (position == keys.end()) {
    std::pair<iterator, bool> temp = keys.insert(std::make_pair(time, key));
    position = temp.first;
  } else {
    key->in_tangent_type = position->second->in_tangent_type;
    key->out_tangent_type = position->second->out_tangent_type;
    key->bind_tangents = position->second->bind_tangents;
    key->in_tangent_lock = position->second->in_tangent_lock;
    key->out_tangent_lock = position->second->out_tangent_lock;
    key->in_tangent = position->second->in_tangent;
    key->out_tangent = position->second->out_tangent;

    if (position == keys.begin()) {
      erase(position);
      position = keys.insert(keys.begin(), std::make_pair(time, key));
    } else {
      erase(position--);
      position = keys.insert(position, std::make_pair(time, key));
    }
  }
  key_changed(position);
}
template <typename T>
void KeyCurve<T>::erase(iterator position)
{
  if (position == keys.begin()) {
    keys.erase(position);
    if (!keys.empty()) {
      key_changed(keys.begin());
    }
  } else {
    keys.erase(position--);
    key_changed(position);
  }
}
template <typename T>
typename KeyCurve<T>::iterator KeyCurve<T>::set_time(iterator position, int time)
{
  iterator temp = keys.find(time);
  if (temp == position || temp != keys.end()) {
    return position;
  }

  Key key = position->second;
  if (position == keys.begin()) {
    erase(position);
    position = keys.insert(keys.begin(), std::make_pair(time, key));
  } else {
    erase(position--);
    position = keys.insert(position, std::make_pair(time, key));
  }
  key_changed(position);

  return position;
}
template <typename T>
T KeyCurve<T>::operator()(float time) const
{
  if (keys.empty()) {
    return T();
  }

  const_iterator p1 = keys.upper_bound(time);
  if (p1 == keys.begin()) {
    return std::dynamic_pointer_cast<KeyCurve<T>::Key>(p1->second)->value;
  }

  const_iterator p0 = p1;
  p0--;
  if (p1 == keys.end()) {
    return std::dynamic_pointer_cast<KeyCurve<T>::Key>(p0->second)->value;
  }

  float t = (time - p0->first) / (float)(p1->first - p0->first); // 0 <= t <= 1
  if constexpr (std::is_same<T, Quatf>::value) {
    return slerp(std::dynamic_pointer_cast<KeyCurve<T>::Key>(p0->second)->value,
                 std::dynamic_pointer_cast<KeyCurve<T>::Key>(p1->second)->value,
                 t);
  } else {
    return p0->second->a + t * p0->second->b + t * t * p0->second->c + t * t * t * p0->second->d;
  }
}

template <typename T>
void KeyCurve<T>::key_changed(iterator position)
{
  if (std::is_same<T, Quatf>::value)
    return;

  iterator next = position;
  if (next != --keys.end() && next != keys.end()) {
    ++next;
  }

  iterator prev = position;
  if (prev != keys.begin()) {
    --prev;
  }

  iterator prev_prev = prev;
  if (prev_prev != keys.begin()) {
    --prev_prev;
  }

  if (next != position) {
    update_curve(next);
  }

  update_curve(position);

  if (prev != position) {
    update_curve(prev);
  }

  if (prev_prev != prev) {
    update_curve(prev_prev);
  }
}

template <typename T>
void KeyCurve<T>::update_curve(iterator position)
{
  if constexpr (!std::is_same<T, Quatf>::value) {

    float time = (float)position->first;
    std::shared_ptr<Key> key = std::dynamic_pointer_cast<KeyCurve<T>::Key>(position->second);

    // First update the tangents

    iterator prev = position;
    if (prev != keys.begin()) {
      --prev;
    }
    iterator next = position;
    if (next != --keys.end() && next != keys.end()) {
      ++next;
    }
    float next_val = std::dynamic_pointer_cast<KeyCurve<T>::Key>(next->second)->value;
    float prev_val = std::dynamic_pointer_cast<KeyCurve<T>::Key>(prev->second)->value;

    if (!key->in_tangent_lock) {
      switch (key->in_tangent_type) {
        default:
        case SMOOTH:
          key->in_tangent = (next_val - prev_val) / (float)(next->first - prev->first);
          break;
      }
    }
    if (!key->out_tangent_lock) {
      switch (key->out_tangent_type) {
        default:
        case SMOOTH:
          key->out_tangent = (next_val - prev_val) / (float)(next->first - prev->first);
          break;
      }
    }

    // Update the cubic spline

    if (next != position) {
      float D1 = key->out_tangent * (next->first - time);
      float D2 = next->second->in_tangent * (next->first - time);

      key->a = key->value;
      key->b = D1;
      key->c = 3.0f * (next_val - key->value) - 2.0f * D1 - D2;
      key->d = 2.0f * (key->value - next_val) + D1 + D2;
    }
  }
}

#endif // KEY_CURVE_HPP
