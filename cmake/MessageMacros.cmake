# This file was taken from the MinVR cmake build system. https://github.com/MinVR/MinVR
# See the main MinVR/CMakeLists.txt file for authors, copyright, and license info.

macro(h1 TITLE)
  string(TOUPPER ${TITLE} TITLE)
  message(STATUS "\n\n==== ${TITLE} ====")
endmacro()

macro(h2 TITLE)
  message(STATUS "\n* ${TITLE}")
endmacro()

macro(h3 TITLE)
  message(STATUS "- ${TITLE}")
endmacro()

