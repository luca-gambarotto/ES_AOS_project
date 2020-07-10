cmake_minimum_required(VERSION 3.5)

if (NOT CONFIG_SAMPLES_NW_OPENCL)
	return()
endif(NOT CONFIG_SAMPLES_NW_OPENCL)

set(MANGO_ROOT "$ENV{MANGO_ROOT}" CACHE STRING "MANGO installation root directory")
set(BOSP_PATH "$ENV{BOSP_PATH}" CACHE STRING "BOSP installation directory")

if (NOT BOSP_PATH) 
	message(FATAL_ERROR "You have to specify -DBOSP_PATH=\"/path/to/bbque\"!")
endif (NOT BOSP_PATH)

if (NOT MANGO_ROOT) 
	message(FATAL_ERROR "You have to specify -DMANGO_ROOT=\"/path/to/mango\"!")
endif (NOT MANGO_ROOT)


set(TARGET_NAME nw_opencl)

set(NW_OCL_SOURCES nw_host)
set_source_files_properties(${NW_OCL_SOURCES} PROPERTIES LANGUAGE CXX)

add_executable (${TARGET_NAME} ${NW_OCL_SOURCES})

target_include_directories(${TARGET_NAME} PUBLIC ${MANGO_ROOT}/include/)
target_include_directories(${TARGET_NAME} PUBLIC ${MANGO_ROOT}/include/libmango/)
target_include_directories(${TARGET_NAME} PUBLIC ${MANGO_ROOT}/include/libmango/host)
target_include_directories(${TARGET_NAME} PUBLIC ${MANGO_ROOT}/usr/include/)
target_include_directories(${TARGET_NAME} PUBLIC ${MANGO_ROOT}/usr/local/include/)
target_include_directories(${TARGET_NAME} PUBLIC ${BOSP_PATH}/include/)
target_include_directories(${TARGET_NAME} PUBLIC ${BOSP_PATH}/include/bbque)

target_compile_options(${TARGET_NAME} PRIVATE -std=c++14)

add_definitions(-DCL_TARGET_OPENCL_VERSION=120)

find_library(LIBMANGO mango HINTS ${MANGO_ROOT}/lib)

target_link_libraries(${TARGET_NAME} PUBLIC
	${MANGO_ROOT}/usr/local/lib/libOpenCL.so
)

install(TARGETS ${TARGET_NAME} DESTINATION ${MANGO_ROOT}/usr/bin)

add_subdirectory(kernel)