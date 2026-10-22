cmake_minimum_required(VERSION 3.25)

# This project only targets WASI, so it picks the wasi-sdk toolchain itself.
if(NOT CMAKE_TOOLCHAIN_FILE)
  if(NOT DEFINED ENV{WASI_SDK_PATH})
    message(FATAL_ERROR "Set WASI_SDK_PATH to the wasi-sdk directory")
  endif()
  set(WASI_SDK_PREFIX "$ENV{WASI_SDK_PATH}" CACHE PATH "wasi-sdk directory")
  set(CMAKE_TOOLCHAIN_FILE "${WASI_SDK_PREFIX}/share/cmake/wasi-sdk-p3.cmake" CACHE FILEPATH "")
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "")
endif()

project(onnxruntime_wit LANGUAGES C CXX)

set(ORT_REPOSITORY "https://github.com/MendyBerger/onnxruntime" CACHE STRING "ONNX Runtime repository with WASI support")
set(ORT_COMMIT "9d4c4be3c8d6f828bfc4471b64fbf75d7b348c72" CACHE STRING "ONNX Runtime commit to build")
set(ORT_SHA256 "96ddead87b4ff9168d95c3eb6e033f46be77251c85964b15f323df5a19391b6e" CACHE STRING "SHA256 of the commit archive")
set(ORT_SOURCE_DIR "" CACHE PATH "Build this local ONNX Runtime checkout instead of downloading ORT_COMMIT")
option(ORT_USE_WEBGPU "Build ONNX Runtime's WebGPU execution provider" ON)

include(ExternalProject)

if(ORT_SOURCE_DIR)
  # Local checkouts change, so let ONNX Runtime's own build decide what is out of date.
  set(ort_source SOURCE_DIR "${ORT_SOURCE_DIR}" BUILD_ALWAYS ON)
else()
  set(ort_source URL "${ORT_REPOSITORY}/archive/${ORT_COMMIT}.tar.gz" URL_HASH SHA256=${ORT_SHA256})
endif()

set(ort_binary_dir "${CMAKE_BINARY_DIR}/onnxruntime")
# ONNX Runtime and the static libraries it depends on, bundled into one.
set(ort_library "${ort_binary_dir}/libonnxruntime_webassembly.a")

ExternalProject_Add(onnxruntime_external
  ${ort_source}
  SOURCE_SUBDIR cmake
  BINARY_DIR "${ort_binary_dir}"
  CMAKE_ARGS
    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
    -DWASI_SDK_PREFIX=${WASI_SDK_PREFIX}
    -DCMAKE_BUILD_TYPE=Release
    -Donnxruntime_BUILD_SHARED_LIB=OFF
    -Donnxruntime_BUILD_UNIT_TESTS=OFF
    -Donnxruntime_BUILD_WEBASSEMBLY_STATIC_LIB=ON
    -Donnxruntime_ENABLE_WEBASSEMBLY_SIMD=ON
    -Donnxruntime_USE_WEBGPU=${ORT_USE_WEBGPU}
  INSTALL_COMMAND ""
  # Stream ONNX Runtime's long download and build output instead of letting Ninja buffer it.
  USES_TERMINAL_DOWNLOAD ON
  USES_TERMINAL_CONFIGURE ON
  USES_TERMINAL_BUILD ON
  BUILD_BYPRODUCTS "${ort_library}"
)
ExternalProject_Get_Property(onnxruntime_external SOURCE_DIR)
set(ort_source_dir "${SOURCE_DIR}")

set(gen_dir "${CMAKE_BINARY_DIR}/gen")
add_custom_command(
  OUTPUT "${gen_dir}/onnxruntime.c" "${gen_dir}/onnxruntime.h" "${gen_dir}/onnxruntime_component_type.o"
  COMMAND wit-bindgen c "${CMAKE_CURRENT_SOURCE_DIR}/wit" --world onnxruntime --async=-all --out-dir "${gen_dir}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/wit/onnxruntime.wit"
)

add_executable(onnxruntime
  adapter/onnxruntime_component.cc
  "${gen_dir}/onnxruntime.c"
  "${gen_dir}/onnxruntime_component_type.o"
)
add_dependencies(onnxruntime onnxruntime_external)
target_include_directories(onnxruntime PRIVATE "${ort_source_dir}/include/onnxruntime/core/session" "${gen_dir}")
target_compile_options(onnxruntime PRIVATE -fwasm-exceptions "SHELL:-mllvm -wasm-use-legacy-eh=false")
target_link_options(onnxruntime PRIVATE
  -mexec-model=reactor
  -fwasm-exceptions "SHELL:-mllvm -wasm-use-legacy-eh=false"
  # MLAS needs more than wasm-ld's default 64 KiB stack.
  -Wl,-z,stack-size=5242880
)
target_link_libraries(onnxruntime PRIVATE
  "${ort_library}"
  unwind dl wasi-emulated-signal wasi-emulated-mman wasi-emulated-getpid wasi-emulated-process-clocks
)
set_target_properties(onnxruntime PROPERTIES SUFFIX ".wasm" LINK_DEPENDS "${ort_library}")
