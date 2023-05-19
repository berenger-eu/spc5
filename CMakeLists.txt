project(spc5 CXX)
cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)

# Set a default build type if none was specified
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")

SET(SPC5_EXTRA_FLAG "-march=native -Wextra -Wnon-virtual-dtor -Wshadow -Wpointer-arith -Wcast-qual -Wconversion  -Wall -Wno-sign-conversion -pedantic -Woverloaded-virtual -Wpointer-arith -Wcast-qual -Wconversion -Wno-error" CACHE STRING "Additional flags"  )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SPC5_EXTRA_FLAG}")

#################################################

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64|arm64|ARM64")
    message(STATUS "ARMv8 architecture detected")
    SET(SPC5_ARM_SVE_FLAG "-march=armv8.2-a+sve" CACHE STRING "Additional ARM SVE flags"  )    
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SPC5_ARM_SVE_FLAG}")
else()
    message(STATUS "Non-ARMv8 architecture")
    # Add configuration or build options for other architectures here
    add_definitions(-DUSE_FARM)
    add_definitions(-DFARM_NB_BITS_IN_VEC=512)
endif()

#################################################

option(USEDENSE "Use dense or no" FALSE)
message(STATUS "USEDENSE = ${USEDENSE}")
if(USEDENSE)
    add_definitions(-DUSEDENSE)
else()

endif()

#################################################

option(USEFLOAT "Use float" FALSE)
message(STATUS "USEFLOAT = ${USEFLOAT}")
if(USEFLOAT)
    add_definitions(-DUSEFLOAT)
else()

endif()

#################################################

option(SPLIT_NUMA "Set to true to split the arrays between threads" TRUE)
message(STATUS "SPLIT_NUMA = ${SPLIT_NUMA}")
if(SPLIT_NUMA)
    add_definitions("-DSPLIT_NUMA")
endif()

#################################################

include(FindOpenMP)
message(STATUS "OPENMP_FOUND = ${OPENMP_FOUND}")
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


#################################################

message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")

file(GLOB_RECURSE cpp_for_lib ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE hpp_for_lib ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)
add_library(spc5 STATIC ${cpp_for_lib} ${hpp_for_lib})

#################################################

if(DEFINED ENV{ARMPLROOT})
    set(ARMPL_FOUND TRUE)
else()
    set(ARMPL_FOUND FALSE)
endif()

option(USE_ARMPL "Compile examples with ARMPL" ${ARMPL_FOUND})
message(STATUS "USE_ARMPL  = ${USE_ARMPL}")
if(DEFINED ENV{ARMPLROOT} AND USE_ARMPL)
    include_directories("$ENV{ARMPLROOT}/include")
    add_definitions("-DUSE_ARMPL")
endif()

#################################################


file(GLOB_RECURSE cpp_for_examples ${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp)
foreach(an_example ${cpp_for_examples})
    get_filename_component(execname ${an_example} NAME_WE)
    
    message(STATUS "Add example ${execname} for ${an_example}")

    add_executable(${execname} ${an_example})
    target_link_libraries(${execname} spc5) 
    if(USE_ARMPL)
        target_link_libraries(${execname} armpl)
    endif()
endforeach()

