
#include <filesystem_compat.hpp>
#include <vtkAddonTestingMacros.h>

#include <cstdlib>

int testPathRelative();

int TestFileSystemCompat(int argc, char* argv[])
{
  CHECK_EXIT_SUCCESS(testPathRelative());

  return EXIT_SUCCESS;
}

int testPathRelative()
{
#if defined(WIN32)
  CHECK_STD_STRING(xromm::filesystem::relative("C:\\path\\to\\a\\file", "C:\\path\to").string(), "a\\file");
#else
  CHECK_STD_STRING(xromm::filesystem::relative("/path/to/a/file", "/path/to").string(), "a/file");
#endif
  return EXIT_SUCCESS;
}
