// ----------------------------------
// Copyright (c) 2018, Brown University
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

/// \file SimulatedAnnealing.cpp
/// \author Bardiya Akhbari


#include "SimulatedAnnealing.hpp"

#include <stdio.h>
#include <cmath>
#include <iostream>

void SA_BA(MAT P0, double *Y, int *ITER, double MAX_TEMP, double MAX_ITER) {
	// P0: This is a matrix of offsets. This is a manipulator on the model,
	// so change in this will be multiply by the orientation and translation
	// of model position. We run the optimization on this. Every change in the
	// first three P0[1] to P0[3] are translation change (in mm)
	// The last three P0[4] to P0[6] are rotation change (in degree)
	// Y: This is the vector of minimized values from the Cost Function

	double Pi;
	double Pcur[6];
	//double Pbest[];

	printf("In Annealing Function\n");
	std::cout << sizeof(P0[1]);

	//std::cout << "R (default) = " << std::endl;
	//	for (int j = 0; j < 6; j++)
	//	{			
	//		//		Y[i + 1] = FUNC(P0[i + 1]);
	//		std::cout << P0[1][j] << " ";

	//		
	//	}
	//	std::cout <<  std::endl;

	Pcur[0] = P0[1][0];
	Pcur[1] = P0[1][1];
	Pcur[2] = P0[1][2];
	Pcur[3] = P0[1][3];
	Pcur[4] = P0[1][4];
	Pcur[5] = P0[1][5];

	for (int i = 0; i < sizeof(Pcur[1]); i++)
	{
		std::cout << Pcur[i] << " ";
		Pi = Pcur[1] + 1;
	}
	//Pbest = Pcur;

	for (int i = 0; i < MAX_ITER; i++)
	{
		Pi = Pcur[1] + 1;
	}
	//double YPR = FUNC(&Pi);

	//for (int j = 0; j < 6; j++)
	//{			
	//	std::cout << Pcur[j] << " ";
	//}
}
