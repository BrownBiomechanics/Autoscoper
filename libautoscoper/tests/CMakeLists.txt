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
if(WIN32)
  # The xromm::filesystem::relative compatibility function is not expected to work on Windows
  set_tests_properties(TestFileSystemCompat PROPERTIES WILL_FAIL TRUE)
endif()
