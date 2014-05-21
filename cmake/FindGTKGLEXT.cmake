if(WIN32)
    
    find_path(GTKGLEXT_INCLUDE_DIRS gtk/gtkgl.h)

    mark_as_advanced(
        GTKGLEXT_INCLUDE_DIRS
    )

    find_library(GTKGLEXT_gdkglext_LIBRARY   NAMES gdkglext)
    find_library(GTKGLEXT_gdkglext_LIBRARY_D NAMES gdkglextd)
    find_library(GTKGLEXT_gtkglext_LIBRARY   NAMES gtkglext)
    find_library(GTKGLEXT_gtkglext_LIBRARY_D NAMES gtkglextd)

    mark_as_advanced(
        GTKGLEXT_gdkglext_LIBRARY GTKGLEXT_gdkglext_LIBRARY_D
        GTKGLEXT_gtkglext_LIBRARY GTKGLEXT_gtkglext_LIBRARY_D
    )

    set(GTKGLEXT_LIBRARIES   ${GTKGLEXT_gdkglext_LIBRARY}   ${GTKGLEXT_gtkglext_LIBRARY})
    set(GTKGLEXT_LIBRARIES_D ${GTKGLEXT_gdkglext_LIBRARY_D} ${GTKGLEXT_gtkglext_LIBRARY_D})

    mark_as_advanced(
        GTKGLEXT_LIBRARIES GTKGLEXT_LIBRARIES_D
    )
else(WIN32)
    include(FindPkgConfig)
    pkg_search_module(GTKGLEXT gtkglext-1.0)
endif(WIN32)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    GTKGLEXT DEFAULT_MSG GTKGLEXT_LIBRARIES
)