# This file was taken from the MinVR cmake build system. https://github.com/MinVR/MinVR
# See the main MinVR/CMakeLists.txt file for authors, copyright, and license info.
#
# Original Author(s) of this File:
#   Daniel Keefe, 2017, University of Minnesota
#
# Author(s) of Significant Updates/Modifications to the File:
#   ...


# Usage:
# ExternalProject_Download(
#     # This first argument is the name of the project to download.  It is required:
#     glm
#
#     # Additional arguments specify how to download the project using GIT, SVN, CVS, or URL.
#     # These can be any of the arguments used for the downloading step of the cmake builtin
#     # ExternalProject_Add command.
#     GIT_REPOSITORY "https://github.com/g-truc/glm.git"
#     GIT_TAG master
#     etc..
# )
function(ExternalProject_Download DLHELPER_PROJECT_NAME)

    include(MessageMacros)
    h1("BEGIN EXTERNAL PROJECT DOWNLOAD (${DLHELPER_PROJECT_NAME}).")

    h2("Creating a download helper project for ${DLHELPER_PROJECT_NAME}.")

    set(DLHELPER_DOWNLOAD_OPTIONS ${ARGN})
    string (REGEX REPLACE "(^|[^\\\\]);" "\\1 " DLHELPER_DOWNLOAD_OPTIONS "${DLHELPER_DOWNLOAD_OPTIONS}")

    configure_file(
        ${PROJECT_ROOT}/cmake/CMakeLists-DownloadHelper.txt.in
        ${EXTERNAL_DIR}/${DLHELPER_PROJECT_NAME}/download-helper/CMakeLists.txt
    )

    h2("Generating build files for the ${DLHELPER_PROJECT_NAME} download helper project.")
    execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" . WORKING_DIRECTORY "${EXTERNAL_DIR}/${DLHELPER_PROJECT_NAME}/download-helper")

    h2("Building the ${DLHELPER_PROJECT_NAME} download helper project.  (This actually performs the download and may take some time...)")
    execute_process(COMMAND "${CMAKE_COMMAND}" --build . WORKING_DIRECTORY "${EXTERNAL_DIR}/${DLHELPER_PROJECT_NAME}/download-helper")

    h2("Completed download of external project ${DLHELPER_PROJECT_NAME}.")

endfunction()


# Usage:
# ExternalProject_BuildAndInstallNow(
#     # This first argument is the name of the external project to download.  It is required:
#     VRPN
#     # This second argument is the relative path from ${EXTERNAL_DIR_NAME}/projectname/ to the project's
#     # main CMakeLists.txt file:
#     src
#
#     # Additional arguments are passed on as options to the cmake build file generator
#     -DVRPN_BUILD_DIRECTSHOW_VIDEO_SERVER=OFF
#     -DVRPN_BUILD_HID_GUI=OFF
#     etc..
# )
function(ExternalProject_BuildAndInstallNow EXT_PROJECT_NAME RELPATH_TO_CMAKELISTS)

    include(MessageMacros)
    h1("BEGIN EXTERNAL PROJECT BUILD AND install (${EXT_PROJECT_NAME}).")

    # any extra args to the function are interpreted as arguments for the cmake config process
    set(CMAKE_CONFIG_OPTIONS ${ARGN})

    # always set the install prefix to be the same
    list(APPEND CMAKE_CONFIG_OPTIONS -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/external)

    #string (REGEX REPLACE "(^|[^\\\\]);" "\\1 " CMAKE_CONFIG_OPTIONS "${CMAKE_CONFIG_OPTIONS}")


    set(SRC_DIR "${EXTERNAL_DIR}/${EXT_PROJECT_NAME}/${RELPATH_TO_CMAKELISTS}")
    set(BUILD_DIR "${CMAKE_BINARY_DIR}/${EXTERNAL_DIR_NAME}/${EXT_PROJECT_NAME}")

    file(MAKE_DIRECTORY ${BUILD_DIR})

    h2("Generating build files for external project ${EXT_PROJECT_NAME}.")
    message(STATUS "Using source dir: ${SRC_DIR}")
    message(STATUS "Using build dir: ${BUILD_DIR}")
    message(STATUS "Config options: ${CMAKE_CONFIG_OPTIONS}")

    execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" ${SRC_DIR} ${CMAKE_CONFIG_OPTIONS} WORKING_DIRECTORY ${BUILD_DIR})

    h2("Building external project ${EXT_PROJECT_NAME}.  (This may take some time...)")
  execute_process(COMMAND "${CMAKE_COMMAND}" --build ${BUILD_DIR} --target install --config Debug)
    execute_process(COMMAND "${CMAKE_COMMAND}" --build ${BUILD_DIR} --target install --config Release)

    h2("Completed external build of ${EXT_PROJECT_NAME}.")

endfunction()

