cmake_minimum_required(VERSION 3.16)
project(semantic_inference VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(SEMANTIC_INFERENCE_USE_TRT "Build with TensorRT" OFF)
option(SEMANTIC_INFERENCE_BUILD_TESTS "Build tests" ON)
option(BUILD_SHARED_LIBS "Build shared libs" ON)

find_package(config_utilities REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# ----------------------------------------------------------------------
# TensorRT manual discovery (Jetson: no TensorRTConfig.cmake available)
# ----------------------------------------------------------------------
if(SEMANTIC_INFERENCE_USE_TRT)
  find_path(TensorRT_INCLUDE_DIR NvInfer.h
            PATHS /usr/include /usr/include/aarch64-linux-gnu)

  find_library(TensorRT_LIBRARY nvinfer
               PATHS /usr/lib /usr/lib/aarch64-linux-gnu)

  find_library(TensorRT_ONNXPARSER_LIBRARY nvonnxparser
               PATHS /usr/lib /usr/lib/aarch64-linux-gnu)

  find_library(TensorRT_INFER_PLUGIN_LIBRARY nvinfer_plugin
               PATHS /usr/lib /usr/lib/aarch64-linux-gnu)

  if (TensorRT_INCLUDE_DIR AND TensorRT_LIBRARY)
    message(STATUS "Found TensorRT: ${TensorRT_LIBRARY}")
    set(ENABLE_TENSORRT_CXX_VALUE 1)

    # TensorRT core
    add_library(TensorRT::TensorRT UNKNOWN IMPORTED)
    set_target_properties(TensorRT::TensorRT PROPERTIES
        IMPORTED_LOCATION ${TensorRT_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${TensorRT_INCLUDE_DIR}
    )

    # TensorRT plugin
    if(TensorRT_INFER_PLUGIN_LIBRARY)
      add_library(TensorRT::nvinfer_plugin UNKNOWN IMPORTED)
      set_target_properties(TensorRT::nvinfer_plugin PROPERTIES
          IMPORTED_LOCATION ${TensorRT_INFER_PLUGIN_LIBRARY}
          INTERFACE_INCLUDE_DIRECTORIES ${TensorRT_INCLUDE_DIR}
      )
    endif()

    # TensorRT ONNX parser
    if(TensorRT_ONNXPARSER_LIBRARY)
      add_library(TensorRT::nvonnxparser UNKNOWN IMPORTED)
      set_target_properties(TensorRT::nvonnxparser PROPERTIES
          IMPORTED_LOCATION ${TensorRT_ONNXPARSER_LIBRARY}
          INTERFACE_INCLUDE_DIRECTORIES ${TensorRT_INCLUDE_DIR}
      )
    endif()
  else()
    message(WARNING "TensorRT headers or libs not found. Inference disabled.")
    set(ENABLE_TENSORRT_CXX_VALUE 0)
  endif()
else()
  message(STATUS "Disabling TensorRT! Inference will not work!")
  set(ENABLE_TENSORRT_CXX_VALUE 0)
endif()

configure_file(
  cmake/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/semantic_inference_config.h
)

add_library(
  ${PROJECT_NAME}
  src/image_recolor.cpp src/image_rotator.cpp src/image_utilities.cpp src/logging.cpp
  src/model_config.cpp src/segmenter.cpp src/panoptic_extractor.cpp
)
target_link_libraries(
  ${PROJECT_NAME} PUBLIC config_utilities::config_utilities ${OpenCV_LIBRARIES}
)
target_include_directories(
  ${PROJECT_NAME}
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> ${OpenCV_INCLUDE_DIRS}
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
add_library(
  semantic_inference::${PROJECT_NAME} ALIAS ${PROJECT_NAME}
)

if(ENABLE_TENSORRT_CXX_VALUE)
  target_sources(${PROJECT_NAME} PRIVATE src/trt_utilities.cpp src/model.cpp)
  target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})

  # Always link core TensorRT
  target_link_libraries(${PROJECT_NAME} PRIVATE TensorRT::TensorRT ${CUDA_LIBRARIES})

  # Only link plugin if it was found
  if(TARGET TensorRT::nvinfer_plugin)
    target_link_libraries(${PROJECT_NAME} PRIVATE TensorRT::nvinfer_plugin)
  endif()

  # Only link ONNX parser if it was found
  if(TARGET TensorRT::nvonnxparser)
    target_link_libraries(${PROJECT_NAME} PRIVATE TensorRT::nvonnxparser)
  endif()
endif()

if(SEMANTIC_INFERENCE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
  EXPORT ${PROJECT_NAME}-targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)
