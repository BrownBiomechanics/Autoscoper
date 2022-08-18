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

/// \file Backtrace.cpp
/// \author Mark Howison

#include <stdlib.h>
#include <stdio.h>
//#include <execinfo.h>

#include "Backtrace.hpp"

namespace xromm {

void bt()
{
  //void* trace[16];
  //char** messages = (char**)NULL;
  //int trace_size = 0;

  //trace_size = backtrace(trace, 16);

  ///* overwrite sigaction with caller's address */
  ////if (addr) trace[1] = addr;
  //messages = backtrace_symbols(trace, trace_size);

  ///* skip first stack frame (points here) */
  //fprintf(stderr, "[bt] Execution path:\n");
  //for (int i=1; i<trace_size; ++i)
  //{
  //  fprintf(stderr, "[bt] #%d %s\n", i, messages[i]);
  //}
}

//void bt_sighandler(int sig, siginfo_t *info, void *secret)
//{
//  if (sig == SIGSEGV)
//    printf("Got signal %d, faulty address is %p\n", sig, info->si_addr);
//  else
//    printf("Got signal %d\n", sig);
//
//  bt();
//
//  exit(EXIT_FAILURE);
//}

void register_bt_sighandler()
{
  ///* Install our signal handler */
  //struct sigaction sa;

  //sa.sa_sigaction = bt_sighandler;
  //sigemptyset(&sa.sa_mask);
  //sa.sa_flags = SA_RESTART;

  //sigaction(SIGSEGV, &sa, NULL);
  //sigaction(SIGUSR1, &sa, NULL);
}

} // xromm
