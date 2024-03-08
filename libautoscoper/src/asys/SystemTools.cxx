

#include "asys/SystemTools.hxx"

#include <fstream>

namespace asys {

// Convenience function around std::getline which removes a trailing carriage
// return and can truncate the buffer as needed.  Returns true
// if any data were read before the end-of-file was reached.
bool SystemTools::GetLineFromStream(std::istream& is,
                                    std::string& line,
                                    bool* has_newline /* = 0 */,
                                    std::string::size_type sizeLimit /* = std::string::npos */)
{
  // Start with an empty line.
  line = "";

  // Early short circuit return if stream is no good. Just return
  // false and the empty line. (Probably means caller tried to
  // create a file stream with a non-existent file name...)
  //
  if (!is) {
    if (has_newline) {
      *has_newline = false;
    }
    return false;
  }

  std::getline(is, line);
  bool haveData = !line.empty() || !is.eof();
  if (!line.empty()) {
    // Avoid storing a carriage return character.
    if (line.back() == '\r') {
      line.resize(line.size() - 1);
    }

    // if we read too much then truncate the buffer
    if (sizeLimit != std::string::npos && line.size() > sizeLimit) {
      line.resize(sizeLimit);
    }
  }

  // Return the results.
  if (has_newline) {
    *has_newline = !is.eof();
  }
  return haveData;
}

} // namespace asys
