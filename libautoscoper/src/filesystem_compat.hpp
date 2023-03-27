#if !defined(Autoscoper_HAS_FILESYSTEM) && !defined(Autoscoper_HAS_EXPERIMENTAL_FILESYSTEM)
#   if defined(__cpp_lib_filesystem)
#       define Autoscoper_HAS_FILESYSTEM 1
#   elif defined(__cpp_lib_experimental_filesystem)
#       define Autoscoper_HAS_EXPERIMENTAL_FILESYSTEM 1
#   elif !defined(__has_include)
#       define Autoscoper_HAS_EXPERIMENTAL_FILESYSTEM 1
#   elif __has_include(<filesystem>)
#       define Autoscoper_HAS_FILESYSTEM 1
#   elif __has_include(<experimental/filesystem>)
#       define Autoscoper_HAS_EXPERIMENTAL_FILESYSTEM 1
#   endif
#endif

#ifndef Autoscoper_HAS_EXPERIMENTAL_FILESYSTEM
#    define Autoscoper_HAS_EXPERIMENTAL_FILESYSTEM 0
#endif

#ifndef Autoscoper_HAS_FILESYSTEM
#    define Autoscoper_HAS_FILESYSTEM 0
#endif

#if Autoscoper_HAS_EXPERIMENTAL_FILESYSTEM
#    include <experimental/filesystem>
namespace std {
    namespace filesystem = experimental::filesystem;
}
#elif Autoscoper_HAS_FILESYSTEM
#    include <filesystem>
#else
#    error Could not find system header "<filesystem>" or "<experimental/filesystem>"
#endif
