/* Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
   file Copyright.txt or https://cmake.org/licensing#kwsys for details.  */
#ifndef asys_SystemTools_hxx
#define asys_SystemTools_hxx

// Adapted from https://gitlab.kitware.com/utils/kwsys
// "asys" is used as namespace and is short for "Autoscoper SystemTools"

#include <iosfwd>
#include <string>

namespace asys {

/** \class SystemTools
 * \brief A collection of useful platform-independent system functions.
 */
class SystemTools
{
public:
  /** -----------------------------------------------------------------
   *               Filename Manipulation Routines
   *  -----------------------------------------------------------------
   */

  /**
   * Read line from file. Make sure to read a full line and truncates it if
   * requested via sizeLimit. Returns true if any data were read before the
   * end-of-file was reached. If the has_newline argument is specified, it will
   * be true when the line read had a newline character.
   */
  static bool GetLineFromStream(std::istream& istr,
                                std::string& line,
                                bool* has_newline = nullptr,
                                std::string::size_type sizeLimit = std::string::npos);
};

} // namespace asys

#endif
