cmake_minimum_required(VERSION 3.16)
project(semantic_inference_ros)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra)

find_package(catkin REQUIRED roscpp nodelet cv_bridge image_transport)
find_package(semantic_inference REQUIRED)
find_package(dynamic_reconfigure REQUIRED)

catkin_python_setup()

generate_dynamic_reconfigure_options(
  cfg/VLMForNavigation.cfg
)

catkin_package(
  CATKIN_DEPENDS cv_bridge image_transport nodelet roscpp semantic_inference_msgs
  DEPENDS semantic_inference
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
)

add_library(
  ${PROJECT_NAME}
  src/mask_nodelet.cpp src/rgbd_segmentation_nodelet.cpp src/segmentation_nodelet.cpp
  src/output_publisher.cpp src/recolor_nodelet.cpp src/republish_nodelet.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC include ${catkin_INCLUDE_DIRS})
target_link_libraries(
  ${PROJECT_NAME} ${catkin_LIBRARIES} semantic_inference::semantic_inference
)

install(
  TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY launch/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)
