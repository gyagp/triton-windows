include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(lib/TritonWebGPUToLLVM)

if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonWebGPU
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_webgpu.cc
    LINK_LIBS TritonWebGPUToLLVM
  )
  target_link_libraries(TritonWebGPU PRIVATE Python3::Module pybind11::headers)
  target_include_directories(TritonWebGPU PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/webgpu-headers
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/TritonWebGPUToLLVM
  )
endif()
