set(KIT libautoscoper)

set(KITTests_SOURCES
  TestFileSystemCompat.cpp
)

create_test_sourcelist(Tests ${KIT}CxxTests.cxx
  ${KITTests_SOURCES}
)

add_executable(${KIT}CppTests ${Tests})
target_link_libraries(${KIT}CppTests ${KIT})

add_test(NAME TestFileSystemCompat COMMAND $<TARGET_FILE:${KIT}CppTests> TestFileSystemCompat)

