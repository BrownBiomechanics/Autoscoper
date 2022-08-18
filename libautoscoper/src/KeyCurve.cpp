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

//! \file KeyCurve.cpp
//! \author Andy Loomis

#include <iostream>

#include "KeyCurve.hpp"

using namespace std;

void KeyCurve::insert(int time)
{
    insert(time,(*this)(time));
}

void KeyCurve::insert(int time, float value)
{
    Key key;
    key.value = value;
    key.in_tangent_type = SMOOTH;
    key.out_tangent_type = SMOOTH;
    key.bind_tangents = true;
    key.in_tangent_lock = false;
    key.out_tangent_lock = false;

    iterator position = keys.find(time);
    if (position == keys.end()) {
        pair<iterator,bool> temp = keys.insert(make_pair(time,key));
        position = temp.first;
    }
    else {
        key.in_tangent_type = position->second.in_tangent_type;
        key.out_tangent_type = position->second.out_tangent_type;
        key.bind_tangents = position->second.bind_tangents;
        key.in_tangent_lock = position->second.in_tangent_lock;
        key.out_tangent_lock = position->second.out_tangent_lock;
        key.in_tangent = position->second.in_tangent;
        key.out_tangent = position->second.out_tangent;

        if (position == keys.begin()) {
            erase(position);
            position = keys.insert(keys.begin(),make_pair(time,key));
        }
        else {
            erase(position--);
            position = keys.insert(position,make_pair(time,key));
        }
    }
    key_changed(position);
}

void KeyCurve::erase(iterator position)
{
    if (position == keys.begin()) {
        keys.erase(position);
        if (!keys.empty()) {
            key_changed(keys.begin());
        }
    }
    else {
        keys.erase(position--);
        key_changed(position);
    }
}

KeyCurve::iterator KeyCurve::set_time(iterator position, int time)
{
    iterator temp = keys.find(time);
    if (temp == position || temp != keys.end()) {
        return position;
    }

    Key key = position->second;
    if (position == keys.begin()) {
        erase(position);
        position = keys.insert(keys.begin(),make_pair(time,key));
    }
    else {
        erase(position--);
        position = keys.insert(position,make_pair(time,key));
    }
    key_changed(position);

    return position;
}

float KeyCurve::operator()(float time) const
{
    if (keys.empty()) {
        return 0.0f;
    }

    const_iterator p1 = keys.upper_bound(time);
    if (p1 == keys.begin()) {
        return p1->second.value;
    }

    const_iterator p0 = p1; p0--;
    if (p1 == keys.end()) {
        return p0->second.value;
    }

    float t = (time-p0->first)/(float)(p1->first-p0->first);
    return p0->second.a+t*p0->second.b+t*t*p0->second.c+t*t*t*p0->second.d;
}

void KeyCurve::key_changed(iterator position)
{
    iterator next = position;
    if (next != --keys.end() && next != keys.end()) {
        ++next;
    }

    iterator prev = position;
    if (prev != keys.begin()){
        --prev;
    }

    iterator prev_prev = prev;
    if (prev_prev != keys.begin()){
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

void KeyCurve::update_curve(iterator position)
{
    float time = (float)position->first;
    Key& key = position->second;

    // First update the tangents

    iterator prev = position;
    if (prev != keys.begin()) {
        --prev;
    }
    iterator next = position;
    if (next != --keys.end() && next != keys.end()) {
        ++next;
    }

    if (!key.in_tangent_lock) {
        switch (key.in_tangent_type) {
            default:
            case SMOOTH:
                key.in_tangent = (next->second.value-prev->second.value)/
                                 (float)(next->first-prev->first);
                break;
        }
    }
    if (!key.out_tangent_lock) {
        switch (key.out_tangent_type) {
            default:
            case SMOOTH:
                key.out_tangent = (next->second.value-prev->second.value)/
                                  (float)(next->first-prev->first);
                break;
        }
    }

    // Update the cubic spline

    if (next != position) {
        float D1 = key.out_tangent*(next->first-time);
        float D2 = next->second.in_tangent*(next->first-time);

        key.a = key.value;
        key.b = D1;
        key.c = 3.0f*(next->second.value-key.value)-2.0f*D1-D2;
        key.d = 2.0f*(key.value-next->second.value)+D1+D2;
    }
}
