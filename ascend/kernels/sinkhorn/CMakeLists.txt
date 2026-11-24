# ascend/kernels/sinkhorn/CMakeLists.txt - the kernel half of the Ascend C skeleton.
#
# Mirrors /vllm-workspace/vllm-ascend/CMakeLists.txt:49-66: CANN's `ascendc_library()` macro compiles
# Ascend C sources to a device .so, and a normal C++ build compiles `torch_binding.cpp` into the torch
# extension that registers them.  This file builds only the first half.
cmake_minimum_required(VERSION 3.16)
project(maya_ascend_sinkhorn CXX)

set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH})
set(SOC_VERSION "ascend910b" CACHE STRING "")

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake not found under ${ASCEND_CANN_PACKAGE_PATH}")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

ascendc_library(maya_ascend_kernels SHARED ${CMAKE_CURRENT_SOURCE_DIR}/sinkhorn.cpp)
