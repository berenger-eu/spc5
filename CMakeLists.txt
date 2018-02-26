project(spc5 CXX)
cmake_minimum_required(VERSION 3.2)


# Set a default build type if none was specified
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")

SET(SPC5_EXTRA_FLAG "-m64 -std=c++11 -march=native -Wextra -Wnon-virtual-dtor -Wshadow -Wpointer-arith -Wcast-qual -Wconversion  -Wall -Wno-sign-conversion -pedantic -Woverloaded-virtual -Wpointer-arith -Wcast-qual -Wconversion -Wno-error" CACHE STRING "Additional flags"  )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SPC5_EXTRA_FLAG}")

#################################################

option(NO_FMADD "Set to true to avoid fmadd (and replace with mul and add)" FALSE)
message(STATUS "NO_FMADD = ${NO_FMADD}")
if(NO_FMADD)
    add_definitions("-DNO_FMADD")
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

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")    
    set(AVX512KNL_FLAGS "-xCOMMON-AVX512 -xMIC-AVX512")
    set(AVX512SKL_FLAGS "-xCOMMON-AVX512 -xCORE-AVX512")
    set(AVX512CNL_FLAGS "-xCOMMON-AVX512 -xCORE-AVX512")
else()
    set(AVX512KNL_FLAGS "-mavx512f -mavx512pf -mavx512er -mavx512cd")
    set(AVX512SKL_FLAGS "-mavx512f -mavx512cd -mavx512vl -mavx512bw -mavx512dq")
    set(AVX512CNL_FLAGS "-mavx512f -mavx512cd -mavx512vl -mavx512bw -mavx512dq -mavx512vbmi  -mavx512ifma")
endif()

set(CPU "KNL" CACHE STRING "Target hardware")
set_property(CACHE CPU PROPERTY STRINGS KNL SKL CNL)
message(STATUS "CPU = ${CPU}")

if(CPU STREQUAL "KNL")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${AVX512KNL_FLAGS}")
elseif(CPU STREQUAL "SKL")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${AVX512SKL_FLAGS}")
elseif(CPU STREQUAL "CNL")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${AVX512CNL_FLAGS}")
else()
    message(ERROR "unknown CPU type ${CPU}")
endif()

#################################################

message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")

file(GLOB_RECURSE cpp_for_lib ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE hpp_for_lib ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)
add_library(spc5 STATIC ${cpp_for_lib} ${hpp_for_lib})

#################################################

if(DEFINED ENV{MKLROOT})
    set(MKL_FOUND TRUE)
else()
    set(MKL_FOUND FALSE)
endif()
option(USE_MKL "Compile examples with MKL" ${MKL_FOUND})
message(STATUS "USE_MKL  = ${USE_MKL}")
if(DEFINED ENV{MKLROOT} AND USE_MKL)
    include_directories("$ENV{MKLROOT}/include")
    add_definitions("-DUSE_MKL")

    option(OMP_MKL "Set to true to use OpenMP MKL" FALSE)
endif()

#################################################


file(GLOB_RECURSE cpp_for_examples ${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp)
foreach(an_example ${cpp_for_examples})
    get_filename_component(execname ${an_example} NAME_WE)
    
    message(STATUS "Add example ${execname} for ${an_example}")

    add_executable(${execname} ${an_example})
        target_link_libraries(${execname} spc5) 
    if(USE_MKL)
        if(OMP_MKL)
            if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")  
                target_link_libraries(${execname} -Wl,--start-group $ENV{MKLROOT}/lib/intel64/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/intel64/libmkl_intel_thread.a $ENV{MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl)
            else()
                target_link_libraries(${execname} -Wl,--start-group $ENV{MKLROOT}/lib/intel64/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/intel64/libmkl_gnu_thread.a $ENV{MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl)
            endif()
        else()
            if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")  
                target_link_libraries(${execname} -Wl,--start-group $ENV{MKLROOT}/lib/intel64/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/intel64/libmkl_sequential.a $ENV{MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl)
            else()
                target_link_libraries(${execname} -Wl,--start-group $ENV{MKLROOT}/lib/intel64/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/intel64/libmkl_sequential.a $ENV{MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl)
            endif()
        endif()
    endif()
    # set_target_properties(${execname} PROPERTIES COMPILE_FLAGS "${AVX512_FLAGS}" )
endforeach()

