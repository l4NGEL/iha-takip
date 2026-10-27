cmake_minimum_required(VERSION 3.18)
project(iha_takip LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(IHA_WITH_TRT "TensorRT + CUDA dedektor ve canli hat" ON)
option(IHA_BUILD_TESTS "GoogleTest birim testleri" ON)
option(IHA_WERROR "Uyarilari hata say" ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(IHA_WARN -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
if(IHA_WERROR)
  list(APPEND IHA_WARN -Werror)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# --- Cekirdek: GPU'suz derlenir (takip, aci hatasi, PID, protokol, olcum) ---
add_library(iha_core
  src/geometry.cpp
  src/kalman.cpp
  src/hungarian.cpp
  src/tracker.cpp
  src/gimbal.cpp
  src/protocol.cpp
  src/latency.cpp
  src/sequence_io.cpp
)
target_include_directories(iha_core PUBLIC include)
target_link_libraries(iha_core PUBLIC Eigen3::Eigen)
target_compile_options(iha_core PRIVATE ${IHA_WARN})

add_library(iha_net src/udp.cpp)
target_link_libraries(iha_net PUBLIC iha_core)
target_compile_options(iha_net PRIVATE ${IHA_WARN})

add_executable(iha_track_offline apps/iha_track_offline.cpp)
target_link_libraries(iha_track_offline PRIVATE iha_core)
target_compile_options(iha_track_offline PRIVATE ${IHA_WARN})

add_executable(gimbal_sim apps/gimbal_sim.cpp)
target_link_libraries(gimbal_sim PRIVATE iha_core)
target_compile_options(gimbal_sim PRIVATE ${IHA_WARN})

# --- TensorRT dedektor + canli hat ---
if(IHA_WITH_TRT)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 86 87)  # Turing (RTX 20xx), Ampere, Jetson Orin
  endif()
  find_package(CUDAToolkit REQUIRED)
  find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)
  find_path(TENSORRT_INCLUDE NvInfer.h PATH_SUFFIXES include)
  find_library(TENSORRT_LIB nvinfer)
  find_library(TENSORRT_ONNX_LIB nvonnxparser)
  if(NOT TENSORRT_INCLUDE OR NOT TENSORRT_LIB OR NOT TENSORRT_ONNX_LIB)
    message(FATAL_ERROR "TensorRT bulunamadi; -DIHA_WITH_TRT=OFF ile yalniz cekirdek derlenebilir")
  endif()

  add_library(iha_trt
    trt/preprocess.cu
    trt/trt_detector.cpp
  )
  target_include_directories(iha_trt PUBLIC include ${TENSORRT_INCLUDE})
  target_link_libraries(iha_trt PUBLIC iha_core CUDA::cudart ${TENSORRT_LIB} ${TENSORRT_ONNX_LIB} ${OpenCV_LIBS})
  # TensorRT basliklari kendi uyarilarini uretiyor; -Werror yalniz bizim C++ kodumuza.
  target_compile_options(iha_trt PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)

  add_executable(iha_track apps/iha_track.cpp)
  target_link_libraries(iha_track PRIVATE iha_trt iha_net ${OpenCV_LIBS})
  target_compile_options(iha_track PRIVATE ${IHA_WARN})

  add_executable(trt_build apps/trt_build.cpp)
  target_link_libraries(trt_build PRIVATE iha_trt ${OpenCV_LIBS})

  add_executable(preprocess_check apps/preprocess_check.cpp)
  target_link_libraries(preprocess_check PRIVATE iha_trt ${OpenCV_LIBS})
endif()

if(IHA_BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)
  add_executable(iha_tests
    tests/test_geometry.cpp
    tests/test_kalman.cpp
    tests/test_hungarian.cpp
    tests/test_tracker.cpp
    tests/test_gimbal.cpp
    tests/test_protocol.cpp
    tests/test_latency.cpp
    tests/test_sequence_io.cpp
  )
  target_link_libraries(iha_tests PRIVATE iha_core iha_net GTest::gtest_main)
  target_compile_options(iha_tests PRIVATE ${IHA_WARN})
  include(GoogleTest)
  gtest_discover_tests(iha_tests)
endif()
