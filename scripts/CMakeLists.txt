# find_package(fork_union REQUIRED)
add_executable(fork_union_test test.cpp)
target_link_libraries(fork_union_test PRIVATE fork_union)
add_test(NAME fork_union_tests COMMAND fork_union_test)

# Enable Thread Sanitizer if using GCC in a Debug (or not-Release) build:
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Enabling ThreadSanitizer (TSAN) for fork_union_test.")
    target_compile_options(fork_union_test PRIVATE -fsanitize=thread -g -O1)
    target_link_options(fork_union_test PRIVATE -fsanitize=thread)
endif ()
