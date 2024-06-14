cmake_minimum_required(VERSION 3.16)
project(ASDAP)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release" FORCE)
endif()
message(STATUS "cmake build type: ${CMAKE_BUILD_TYPE}")

# Libigl
option(LIBIGL_GLFW         "Build target igl::glfw"         ON)
option(LIBIGL_IMGUI        "Build target igl::imgui"        ON)
option(LIBIGL_OPENGL       "Build target igl::opengl"       ON)
option(LIBIGL_PNG          "Build target igl::png"          ON)
include(libigl)

add_subdirectory("geometry-central")

include(FetchContent)
FetchContent_Declare(
  tinyad
  GIT_REPOSITORY https://github.com/patr-schm/tinyad.git
  GIT_TAG 75093e14ef0d7bb39657c5f3b2aba1251afaa38c
)
#FetchContent_MakeAvailable(tinyad)

FetchContent_GetProperties(tinyad)
if(NOT tinyad_POPULATED)
  # Fetch the content using previously declared details
  FetchContent_Populate(tinyad)
  message(STATUS "tinyad_SOURCE_DIR: ${tinyad_SOURCE_DIR}")
  message(STATUS "tinyad_BINARY_DIR: ${tinyad_BINARY_DIR}")
  add_subdirectory(${tinyad_SOURCE_DIR} ${tinyad_BINARY_DIR})
endif()

SET(CMAKE_CXX_FLAGS "${BASE_CXX_FLAGS} ${DISABLED_WARNINGS} ${DCMAKE_CXX_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${DCMAKE_EXE_LINKER_FLAGS}")
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${DCMAKE_SHARED_LINKER_FLAGS}")

# Add your project files
file(GLOB SRC_FILES src/*.cpp)
file(GLOB HEADER_FILES include/*.hpp)

file(GLOB VIEWER_FILES viewer/*.cpp)

add_library(asdap_core STATIC ${SRC_FILES} ${HEADER_FILES})
target_include_directories(asdap_core PUBLIC include)
target_link_libraries(asdap_core PUBLIC geometry-central igl::core TinyAD)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC asdap_core)

add_executable(${PROJECT_NAME}_gui ${VIEWER_FILES})
target_link_libraries(${PROJECT_NAME}_gui PUBLIC asdap_core igl::glfw igl::opengl igl::imgui igl::png)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)