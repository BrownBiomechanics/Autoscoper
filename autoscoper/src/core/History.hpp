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

//! \file History.hpp
//! \author Andy Loomis, Benjamin Knorlein

#ifndef HISTORY_HPP
#define HISTORY_HPP

#include <list>
#include <stdexcept>

#include "KeyCurve.hpp"

struct State
{
	std::vector <KeyCurve> x_curve;
	std::vector <KeyCurve> y_curve;
	std::vector <KeyCurve> z_curve;
	std::vector <KeyCurve> x_rot_curve;
	std::vector <KeyCurve> y_rot_curve;
	std::vector <KeyCurve> z_rot_curve;
};

class History
{
public:

    History(unsigned size) : size(size)
    {
        it = states.begin();
    }

    void clear()
    {
        states.clear();
        it = states.end();
    }

    void push(const State& state)
    {
        states.erase(it,states.end());
        states.push_back(state);
        if (states.size() > size) {
            states.pop_front();
        }
        it = states.end();
    }
    
    bool can_undo() const
    {
        return it != states.begin();
    }

    State undo()
    {
        if (!can_undo()) {
            throw std::runtime_error("There is nothing to undo");
        }
        return *(--it);
    }

    bool can_redo() const
    {
        std::list<State>::iterator test = it;
        return test != states.end() && (++test) != states.end();
    }

    State redo()
    {
        if (!can_redo()) {
            throw std::runtime_error("There is nothing to redo");
        }
        return *(++it);
    }

private:

    unsigned size;

    std::list<State> states;
    
    std::list<State>::iterator it;
};

#endif // HISTORY_HPP
