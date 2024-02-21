#include "filesystem_compat.hpp"

namespace xromm {
namespace filesystem {
// Experimental filesystem compliant version where there is no fs::relative
std::filesystem::path relative(
  const std::filesystem::path& path,
  const std::filesystem::path& basePath)
{
  // find the first mismatched element and the shared root
  auto mismatched = std::mismatch(path.begin(), path.end(), basePath.begin(), basePath.end());

  // If the paths are the same, return "."
  if (mismatched.first == path.end() && mismatched.second == basePath.end()) {
    return ".";
  }

  std::filesystem::path relativePath;
  auto pathIt = mismatched.first;
  auto baseIt = mismatched.second;

  // Move the base to the shared root
  for (; baseIt != basePath.end(); ++baseIt) {
    relativePath /= "..";
  }

  // Add the remaining path
  for (; pathIt != path.end(); ++pathIt) {
    relativePath /= *pathIt;
  }

  return relativePath.string();
}
}
}

#if Autoscoper_HAS_EXPERIMENTAL_FILESYSTEM
namespace std {
namespace experimental::filesystem {
std::filesystem::path relative(
  const std::filesystem::path& path,
  const std::filesystem::path& basePath)
{
  return xromm::filesystem::relative(path, basePath);
}
}
}
#endif
