cmake_minimum_required(VERSION 3.8)
project (hyperfine_moments C CXX )

unset(COMPILER_SET CACHE)
unset(LINALG_INCLUDE_DIRS CACHE)
unset(LINALG_LIBS CACHE)
unset(LIBS CACHE)

add_subdirectory(linalg)
add_subdirectory(utils)
add_subdirectory(src)
