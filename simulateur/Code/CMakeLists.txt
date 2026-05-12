cmake_minimum_required(VERSION 3.10.2)
# *******************************************
# ************* CMake Content ***************
# *******************************************
# This CMake create a workspace containing the following projects


#
# Program
#  - degridder
# Add definition for relative path into project
add_definitions( -DPROJECT_ROOT_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
project(degridder C CXX)
# Set to 1 to activate debug compilation (0 for release)
if(NOT ${CMAKE_GENERATOR} MATCHES "Visual Studio.*")
    if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
        MESSAGE("Generate Debug project")
        #           set(CMAKE_BUILD_TYPE Debug)
        #           set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Debug)
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -pg -Wall ${ADDITIONAL_C_FLAGS}")
    else()
        MESSAGE("Generate Release project")
        #           set(CMAKE_BUILD_TYPE Release)
        #           set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Release)
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -g -pg -Wall ${ADDITIONAL_C_FLAGS}")
    endif()
    #add libmath during non visual studio builds
    set(CMAKE_EXTRA_LIB m)
else()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endif()
set(LIBS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/lib/cmake_modules/")

# *******************************************
# ************ Pthread LIBRARY **************
# *******************************************
if (WIN32)
    set(THREADS_USE_PTHREADS_WIN32 true)
    # pthread included AFTER Sdl to avoid unnecessary warnings
    file(GLOB PTHREADDIR "${LIBS_DIR}/pthread-[\\.|0-9]*")
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${PTHREADDIR})
    Find_Package ( THREADS REQUIRED )
else()
    Find_Package ( Threads )
endif()
if(NOT THREADS_FOUND)
    MESSAGE(FATAL_ERROR "Pthread not found !")
endif()
if (WIN32)
    file(GLOB
            PTHREADS_DLL
            ${CMAKE_PREFIX_PATH}/lib/*.dll
            )
    MESSAGE("Copy Pthreads DLLs into ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    if(NOT ${CMAKE_GENERATOR} MATCHES "Visual Studio.*")
        file(COPY ${PTHREADS_DLL} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    else()
        file(COPY ${PTHREADS_DLL} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug)
        file(COPY ${PTHREADS_DLL} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release)
    endif()
endif()

# *******************************************
# ************ OpenMP LIBRARY ***************
# *******************************************
OPTION (USE_OpenMP "Use OpenMP" ON)
IF(USE_OpenMP)
    FIND_PACKAGE(OpenMP)
    IF(OPENMP_FOUND)
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    ENDIF()
ENDIF()

# *******************************************
# ************ casacore LIBRARY **************
# *******************************************
add_library(casa_wrapper STATIC
        src/casacore_wrapper.cpp
        include/casacore_wrapper.h
)
target_link_libraries(casa_wrapper fftw3 fftw3_threads)

# *******************************************
# ***** CPU Version  ***********
# *******************************************
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/${FOLDER} ${THREADS_PTHREADS_INCLUDE_DIR} ${FFTW_INCLUDE_DIRS})
file(
        GLOB
        source_files_CPU
        ./src/CPU/*.c*
        ./${FOLDER}/*.c*
        ./src/*.c*
)
file(
        GLOB
        header_files
        ./include/*.h
        ./${FOLDER}/*.h
)

add_executable( degridder_pipeline ${source_files_CPU} ${header_files})
target_link_libraries(degridder_pipeline ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_EXTRA_LIB} fftw3 fftw3f m casa_wrapper)