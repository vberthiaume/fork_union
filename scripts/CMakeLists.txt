# Test targets for C++11, C++14, C++17, and C++20
set(TEST_SOURCES test.cpp)
set(CXX_STANDARDS 11 14 17 20)

foreach (STD IN LISTS CXX_STANDARDS)
    # Derive a unique target name
    set(TGT fork_union_test_cxx${STD})

    # Create the executable
    add_executable(${TGT} ${TEST_SOURCES})
    target_link_libraries(${TGT} PRIVATE fork_union)

    # Enforce the language standard (requires CMake ≥3.8; otherwise use target_compile_options)
    target_compile_features(${TGT} PRIVATE cxx_std_${STD})

    # register it as a CTest test
    add_test(NAME ${TGT} COMMAND ${TGT})

    # if we’re on GCC and not in Release, turn on TSAN here too
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT CMAKE_BUILD_TYPE STREQUAL "Release")
        message(STATUS "Enabling ThreadSanitizer for ${TGT}")
        target_compile_options(${TGT} PRIVATE -fsanitize=thread -g -O1)
        target_link_options(${TGT} PRIVATE -fsanitize=thread)
    endif ()
endforeach ()
