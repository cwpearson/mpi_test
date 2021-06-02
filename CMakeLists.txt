# 3.17+ for CMAKE_CUDA_KNOWN_FEATURES/cuda_std_11
# 3.18+ for CUDA_ARCHITECTURES
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(tempi LANGUAGES CXX VERSION 0.1.0.0)

include(CheckLanguage)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
message(STATUS "CMAKE_CUDA_ARCHITECTURES not defined, setting to OFF")
set(CMAKE_CUDA_ARCHITECTURES OFF CACHE STRING "")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

check_language(CUDA)
find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

if(CMAKE_CUDA_COMPILER)
  enable_language(Fortran)
else()
  message(STATUS "No Fortran support")
endif()

if (MPI_FOUND)
message(STATUS "MPI_VERSION:                 ${MPI_VERSION}")
message(STATUS "MPI_CXX_COMPILER:            ${MPI_CXX_COMPILER}")
message(STATUS "MPI_CXX_COMPILE_OPTIONS:     ${MPI_CXX_COMPILE_OPTIONS}")
message(STATUS "MPI_CXX_COMPILE_DEFINITIONS: ${MPI_CXX_COMPILE_DEFINITIONS}")
message(STATUS "MPI_CXX_INCLUDE_DIRS:        ${MPI_CXX_INCLUDE_DIRS}")
message(STATUS "MPI_CXX_LIBRARIES:           ${MPI_CXX_LIBRARIES}")
message(STATUS "MPI_CXX_LINK_FLAGS:          ${MPI_CXX_LINK_FLAGS}")
message(STATUS "MPIEXEC_EXECUTABLE:          ${MPIEXEC_EXECUTABLE}")
message(STATUS "MPIEXEC_NUMPROC_FLAG:        ${MPIEXEC_NUMPROC_FLAG}")
message(STATUS "MPIEXEC_MAX_NUMPROCS:        ${MPIEXEC_MAX_NUMPROCS}")
message(STATUS "MPIEXEC_PREFLAGS:            ${MPIEXEC_PREFLAGS}")
message(STATUS "MPIEXEC_POSTFLAGS:           ${MPIEXEC_POSTFLAGS}")
endif()

function(set_cxx_options target)
target_compile_options(${target} PRIVATE
$<$<COMPILE_LANGUAGE:CXX>:
 -Wall
 -Wextra
 -Wcast-align;
 -Wdisabled-optimization;
 -Wformat=2;
 -Winit-self;
 -Wlogical-op;
 -Wmissing-include-dirs;
 -Woverloaded-virtual;
 -Wpointer-arith;
 -Wshadow;
 -Wstrict-aliasing;
 -Wswitch-enum;
 -Wvla;
 >
)
endfunction()

function(set_cuda_options target)
target_compile_options(${target} PRIVATE
$<$<COMPILE_LANGUAGE:CUDA>:
--Wno-deprecated-gpu-targets;
--expt-extended-lambda;
-Xcompiler=-Wall;
-Xcompiler=-Wextra;
-Xcompiler=-Wcast-align;
-Xcompiler=-Wdisabled-optimization;
-Xcompiler=-Wformat=2;
-Xcompiler=-Winit-self;
-Xcompiler=-Wlogical-op;
-Xcompiler=-Wmissing-include-dirs;
-Xcompiler=-Woverloaded-virtual;
-Xcompiler=-Wpointer-arith;
-Xcompiler=-Wshadow;
-Xcompiler=-Wstrict-aliasing;
-Xcompiler=-Wswitch-enum;
-Xcompiler=-Wvla;
-Xptxas=-v;
>
)
endfunction()

function(set_cxx_standard target)
set_property(TARGET ${target} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${target} PROPERTY CXX_EXTENSIONS OFF)
set_property(TARGET ${target} PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET ${target} PROPERTY CUDA_STANDARD 11)
set_property(TARGET ${target} PROPERTY CUDA_STANDARD_REQUIRED ON)
endfunction()

# copy run-all.sh to build directory
configure_file(${CMAKE_CURRENT_LIST_DIR}/run-all.sh ${CMAKE_CURRENT_BINARY_DIR}/run-all.sh COPYONLY)

if (MPI_FOUND)
  add_executable(hello-world hello_world.cpp)
  target_link_libraries(hello-world MPI::MPI_CXX)
  # target_include_directories(hello-world PRIVATE ${MPI_CXX_INCLUDE_PATH})
  # target_compile_options(hello-world PRIVATE ${MPI_CXX_COMPILE_FLAGS})
  # target_link_libraries(hello-world ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
  set_cxx_options(hello-world)
  set_cxx_standard(hello-world)
endif()

if (MPI_FOUND)
  add_executable(one-sided one_sided.cpp)
  target_link_libraries(one-sided MPI::MPI_CXX)
  set_cxx_options(one-sided)
  set_cxx_standard(one-sided)
endif()

if (MPI_FOUND)
  add_executable(persistent persistent.cpp)
  target_link_libraries(persistent MPI::MPI_CXX)
  set_cxx_options(persistent)
  set_cxx_standard(persistent)
endif()

if (MPI_FOUND)
  add_executable(send-recv send_recv.cpp)
  target_link_libraries(send-recv MPI::MPI_CXX)
  set_cxx_options(send-recv)
  set_cxx_standard(send-recv)
endif()

if (MPI_FOUND AND CMAKE_CUDA_COMPILER)
  add_executable(one-sided-gpu one_sided_gpu.cpp)
  target_link_libraries(one-sided-gpu MPI::MPI_CXX)
  target_link_libraries(one-sided-gpu CUDA::cudart)
  set_cxx_options(one-sided-gpu)
  set_cxx_standard(one-sided-gpu)
endif()

if (MPI_FOUND AND CMAKE_CUDA_COMPILER)
  add_executable(       persistent-gpu persistent_gpu.cpp)
  target_link_libraries(persistent-gpu MPI::MPI_CXX)
  target_link_libraries(persistent-gpu CUDA::cudart)
  set_cxx_options(      persistent-gpu)
  set_cxx_standard(     persistent-gpu)
endif()

if (MPI_FOUND AND CMAKE_CUDA_COMPILER)
  add_executable(send-recv-gpu send_recv_gpu.cpp)
  target_link_libraries(send-recv-gpu MPI::MPI_CXX)
  target_link_libraries(send-recv-gpu CUDA::cudart)
  set_cxx_options(send-recv-gpu)
  set_cxx_standard(send-recv-gpu)
endif()
