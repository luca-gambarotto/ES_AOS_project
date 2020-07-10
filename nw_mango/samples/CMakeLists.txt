cmake_minimum_required(VERSION 3.0)

include(../global.cmake)

set(MANGO_ROOT "$ENV{MANGO_ROOT}" CACHE STRING "MANGO installation root directory")
set(BOSP_PATH "$ENV{BOSP_PATH}" CACHE STRING "BOSP installation directory")

if(CONFIG_LIBMANGO_GN)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DGNEMU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGNEMU")
endif(CONFIG_LIBMANGO_GN)

add_definitions(-DBOOST_NO_CXX98_BINDERS -DCL_TARGET_OPENCL_VERSION=120)

set (CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")
if (BUILD_TYPE STREQUAL "Debug")
  set (CMAKE_BUILD_TYPE Debug)
  set (DEBUG ON)
else (BUILD_TYPE STREQUAL "Debug")
  set (CMAKE_BUILD_TYPE Release)
  set (CMAKE_CXX_FLAGS "-O3")
endif (BUILD_TYPE STREQUAL "Debug")

set (CMAKE_INSTALL_RPATH "${MANGO_ROOT}/lib:${BOSP_PATH}/lib/:${BOSP_PATH}/lib/bbque")
set (CMAKE_INSTALL_RPATH_USE_LINK TRUE)

if (CONFIG_SAMPLES_MATRIX_MULTIPLICATION)
add_subdirectory(matrix_multiplication)
endif (CONFIG_SAMPLES_MATRIX_MULTIPLICATION)

if (CONFIG_SAMPLES_MATRIX_MULTIPLICATION_CPP)
add_subdirectory(matrix_multiplication_cpp)
endif (CONFIG_SAMPLES_MATRIX_MULTIPLICATION_CPP)

if (CONFIG_SAMPLES_MATRIX_MULTIPLICATION_OCL)
add_subdirectory(matrix_multiplication_opencl)
endif (CONFIG_SAMPLES_MATRIX_MULTIPLICATION_OCL)

if (CONFIG_SAMPLES_MATRIX_MULTIPLICATION_NUP)
add_subdirectory(nuplus_matrix_multiplication)
endif (CONFIG_SAMPLES_MATRIX_MULTIPLICATION_NUP)

if (CONFIG_SAMPLES_MATRIX_MULTIPLICATION_MT)
add_subdirectory(matrix_multiplication_multitarget)
endif (CONFIG_SAMPLES_MATRIX_MULTIPLICATION_MT)

if (CONFIG_SAMPLES_GIF_ANIMATION)
add_subdirectory(gif_animation)
endif (CONFIG_SAMPLES_GIF_ANIMATION)

if (CONFIG_SAMPLES_GIF_FIFO)
add_subdirectory(gif_fifo)
endif (CONFIG_SAMPLES_GIF_FIFO)

if (CONFIG_SAMPLES_SYNC)
add_subdirectory(sync)
endif (CONFIG_SAMPLES_SYNC)

if (CONFIG_SAMPLES_PATHFINDER)
add_subdirectory(pathfinder)
endif (CONFIG_SAMPLES_PATHFINDER)

#if (CONFIG_SAMPLES_EVENT_DEMO)
#add_subdirectory(event_demo)
#endif (CONFIG_SAMPLES_EVENT_DEMO)

if (CONFIG_SAMPLES_NW_OPENCL)
add_subdirectory(nw_opencl)
endif (CONFIG_SAMPLES_NW_OPENCL)
