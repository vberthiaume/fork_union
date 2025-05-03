# find_package(fork_union REQUIRED)
add_executable(fork_union_test test.cpp)
target_link_libraries(fork_union_test PRIVATE fork_union)
add_test(NAME fork_union_tests COMMAND fork_union_test)
