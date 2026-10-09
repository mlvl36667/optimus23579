cmake_minimum_required(VERSION 3.18)
project(llm_alloc_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# LibTorch
# ---------------------------------------------------------------------------
# Point Torch_DIR at the cmake package shipped with the pip `torch` wheel if
# the caller did not supply one explicitly.  Override with:
#   cmake -DTorch_DIR=/path/to/torch/share/cmake/Torch ..
if(NOT DEFINED Torch_DIR)
  # Prefer a CPU-only LibTorch if present (avoids the CUDA toolkit requirement
  # that the pip cu121 wheel imposes).  Fall back to the pip torch package.
  set(_CPU_TORCH "$ENV{HOME}/.local/lib/libtorch/share/cmake/Torch")
  if(EXISTS "${_CPU_TORCH}")
    set(Torch_DIR "${_CPU_TORCH}")
    message(STATUS "Using CPU LibTorch: ${Torch_DIR}")
  else()
    execute_process(
      COMMAND python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'share', 'cmake', 'Torch'))"
      OUTPUT_VARIABLE _TORCH_CMAKE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _TORCH_FIND_RESULT
    )
    if(_TORCH_FIND_RESULT EQUAL 0 AND EXISTS "${_TORCH_CMAKE_DIR}")
      set(Torch_DIR "${_TORCH_CMAKE_DIR}")
      message(STATUS "Auto-detected Torch_DIR: ${Torch_DIR}")
    endif()
  endif()
endif()

find_package(Torch REQUIRED)

# The pip LibTorch wheel is built with the pre-C++11 string ABI.  Linking
# against it requires the same ABI flag, otherwise std::string symbols mismatch.
# TORCH_CXX_FLAGS carries -D_GLIBCXX_USE_CXX11_ABI=0 when needed.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ---------------------------------------------------------------------------
# CUDA runtime resolution for CUDA Torch builds.
# ---------------------------------------------------------------------------
# This project is CPU-only, but a CUDA Torch build (conda / pip cu* wheels)
# has libtorch.so NEEDED-linked against libtorch_cuda.so -> libc10_cuda.so,
# so the CUDA libraries are pulled in transitively and the linker must resolve
# their libcudart symbols even though we never call a CUDA op.
#
# The failure
#   libc10_cuda.so: undefined reference to cudaGetDriverEntryPointByVersion
# happens when the system libcudart (/usr/lib/.../libcudart.so.12) is OLDER
# than the one the wheel was built against.  Modern pip/conda torch ships the
# matching runtime in the `nvidia-cuda-runtime` package next to torch; point
# the linker (and rpath) at it so the correct symbols are found.
get_filename_component(_TORCH_LIBDIR "${TORCH_LIBRARY}" DIRECTORY)
if(_TORCH_LIBDIR)
  # torch/lib/../../nvidia/cuda_runtime/lib
  get_filename_component(_SITE_PKGS "${_TORCH_LIBDIR}/../.." ABSOLUTE)
  set(_NVIDIA_CUDART "${_SITE_PKGS}/nvidia/cuda_runtime/lib")
  if(EXISTS "${_NVIDIA_CUDART}/libcudart.so.12")
    message(STATUS "Using bundled CUDA runtime: ${_NVIDIA_CUDART}")
    link_directories(BEFORE "${_NVIDIA_CUDART}")
    list(APPEND CMAKE_BUILD_RPATH "${_NVIDIA_CUDART}")
    list(APPEND CMAKE_INSTALL_RPATH "${_NVIDIA_CUDART}")
  endif()
endif()

set(LLM_ALLOC_TORCH_LIBS "${TORCH_LIBRARIES}")

# ---------------------------------------------------------------------------
# nlohmann/json (header-only, system install at /usr/include/nlohmann)
# ---------------------------------------------------------------------------
find_package(nlohmann_json QUIET)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(llm_alloc STATIC
  src/domain.cpp
  src/simulator.cpp
  src/allocators.cpp
  src/metrics.cpp
  src/batched_simulator.cpp
  src/learning.cpp
)
target_include_directories(llm_alloc PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(llm_alloc PUBLIC "${LLM_ALLOC_TORCH_LIBS}")
if(nlohmann_json_FOUND)
  target_link_libraries(llm_alloc PUBLIC nlohmann_json::nlohmann_json)
endif()

# ---------------------------------------------------------------------------
# Executables
# ---------------------------------------------------------------------------
add_executable(llm_run       apps/run_main.cpp)
add_executable(llm_benchmark apps/benchmark_main.cpp)
add_executable(llm_sweep     apps/sweep_main.cpp)
foreach(_app llm_run llm_benchmark llm_sweep)
  target_link_libraries(${_app} PRIVATE llm_alloc)
endforeach()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
add_executable(llm_tests
  tests/test_main.cpp
  tests/test_core.cpp
  tests/test_batched.cpp
)
target_link_libraries(llm_tests PRIVATE llm_alloc)
add_test(NAME llm_tests COMMAND llm_tests)
