# - Try to find OpenCL
# This module tries to find an OpenCL implementation on your system. It supports
# AMD / ATI, Apple and NVIDIA implementations, but shoudl work, too.
#
# Once done this will define
#  TIFFIO_FOUND        - system has OpenCL
#  TIFFIO_INCLUDE_DIRS  - the OpenCL include directory
#  TIFFIO_LIBRARIES    - link these to use OpenCL
#
# WIN32 should work, but is untested

FIND_PACKAGE( PackageHandleStandardArgs )

IF (APPLE)

  FIND_LIBRARY(TIFFIO_LIBRARIES tiffio DOC "TiffIO lib for OSX")
  FIND_LIBRARY(LZMA_LIBRARIES lzma DOC "LZMA lib for OSX")
  FIND_PATH(TIFFIO_INCLUDE_DIRS tiffio.h DOC "Include for TiffIO on OSX")


ELSE (APPLE)

	IF (WIN32)
	
	    FIND_PATH(TIFFIO_INCLUDE_DIRS tiffio.h)
	    FIND_LIBRARY(TIFFIO_LIBRARIES tiffio.lib )
	
	ELSE (WIN32)

            # Unix style platforms
            FIND_LIBRARY(TIFFIO_LIBRARIES tiffio
              ENV LD_LIBRARY_PATH
            )

            FIND_PATH(TIFFIO_INCLUDE_DIRS tiffio.h PATHS)

	ENDIF (WIN32)
ENDIF (APPLE)


