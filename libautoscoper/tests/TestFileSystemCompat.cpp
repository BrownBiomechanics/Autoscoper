
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
  return EXIT_SUCCESS;
}
