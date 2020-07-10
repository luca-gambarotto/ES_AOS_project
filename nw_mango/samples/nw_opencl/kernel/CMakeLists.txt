set(Main_File main.c)

if(CONFIG_LIBMANGO_GN)
	set(ParserArgs GN)
else(CONFIG_LIBMANGO_GN)
	set(ParserArgs PEAK)
endif(CONFIG_LIBMANGO_GN)

set_source_files_properties(${Main_File}
                           PROPERTIES GENERATED TRUE)
add_custom_target (nw_opencl_main_gen
        COMMAND ${MANGO_ROOT}/bin/mango_gen_kernel_entry.py ${ParserArgs} ${CMAKE_CURRENT_SOURCE_DIR}/nw_opencl_dev.c
        COMMENT "Generating main.c..."
        WORKING_DIRECTORY "${CMAKE_CURRENT_BUILD_DIR}"
)


if(CONFIG_LIBMANGO_GN)

        add_executable(nw_opencl_dev nw_opencl_dev.c ${Main_File})
	find_library(MANGODEV mango-dev-gn PATHS ${MANGO_ROOT}/lib)
	find_package(Threads)
        target_link_libraries(nw_opencl_dev ${MANGODEV} ${CMAKE_THREAD_LIBS_INIT})
	target_include_directories(nw_opencl_dev PUBLIC ${MANGO_ROOT}/include/libmango/)

	install(FILES ${CMAKE_CURRENT_BINARY_DIR}/nw_opencl_dev
        DESTINATION ${MANGO_ROOT}/usr/local/share/nw_kernel/
	PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

        add_dependencies (nw_opencl nw_opencl_dev)
        add_dependencies (nw_opencl_dev nw_opencl_main_gen)

else(CONFIG_LIBMANGO_GN)

find_program (PEAK_COMPILER "mipsel-unknown-gappeak-gcc" PATHS "/usr/local/bin"
		"${MANGO_ROOT}/usr/local/mipsel-unknown-gappeak/bin/")
find_program (PEAK_READELF "mipsel-unknown-gappeak-readelf" PATHS "/usr/local/bin"
		"${MANGO_ROOT}/usr/local/mipsel-unknown-gappeak/bin/")
find_program (PEAK_OBJDUMP "mipsel-unknown-gappeak-objdump" PATHS "/usr/local/bin"
		"${MANGO_ROOT}/usr/local/mipsel-unknown-gappeak/bin/")

if (${PEAK_COMPILER} STREQUAL "PEAK_COMPILER-NOTFOUND") 
	message(FATAL_ERROR "I'm not able to find the PEAK compiler.")
endif (${PEAK_COMPILER} STREQUAL "PEAK_COMPILER-NOTFOUND") 

if (${PEAK_READELF} STREQUAL "PEAK_READELF-NOTFOUND") 
	message(FATAL_ERROR "I'm not able to find the PEAK readelf.")
endif (${PEAK_READELF} STREQUAL "PEAK_READELF-NOTFOUND") 

if (${PEAK_OBJDUMP} STREQUAL "PEAK_OBJDUMP-NOTFOUND") 
	message(FATAL_ERROR "I'm not able to find the PEAK objdump.")
endif (${PEAK_OBJDUMP} STREQUAL "PEAK_OBJDUMP-NOTFOUND") 


add_custom_target (nw_opencl_dev
	COMMAND ${CMAKE_MAKE_PROGRAM} -f ${CMAKE_CURRENT_SOURCE_DIR}/Makefile.peak MANGO_ROOT=${MANGO_ROOT}
	COMMENT "Compiling kernel..."
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_target(nw_opencl_dev_cp
	COMMAND ${CMAKE_COMMAND} -E copy_directory
	${CMAKE_CURRENT_SOURCE_DIR}/output ${CMAKE_CURRENT_BINARY_DIR}
	COMMENT "Copying kernel directory..."
)

add_custom_target(nw_opencl_dev_cp_main
	COMMAND ${CMAKE_COMMAND} -E copy
	${CMAKE_CURRENT_BINARY_DIR}/main.c ${CMAKE_CURRENT_SOURCE_DIR}
	COMMENT "Copying main.c..."
)

add_dependencies (nw_opencl nw_opencl_dev)
add_dependencies (nw_opencl_dev nw_opencl_dev_cp)
add_dependencies (nw_opencl_dev_cp nw_opencl_dev_cp_main)
add_dependencies (nw_opencl_dev_cp_main nw_opencl_main_gen)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/output/memory.data.fpga.datafile
	DESTINATION ${MANGO_ROOT}/usr/local/share/nw_opencl/)

endif(CONFIG_LIBMANGO_GN)


