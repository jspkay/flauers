cmake_minimum_required(VERSION 3.25)
project(saffira_core VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS  "-Wall -Wextra")
# set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -lboost_numpy39 -lboost_python39")
set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(Boost COMPONENTS numpy3 REQUIRED)
find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)

include_directories(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
find_package(Boost 1.67 REQUIRED COMPONENTS python39)

message(STATUS "PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
message(STATUS "PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE}")
message(STATUS "PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
message(STATUS "Boost_LIBRARIES = ${Boost_LIBRARIES}")

add_library(saffira_core MODULE ./saffira.cpp)

target_link_libraries(saffira_core Boost::python Boost::numpy ${PYTHON_LIBRARIES})
set_target_properties(saffira_core
        PROPERTIES
            OUTPUT_NAME ".cpython-39-x86_64-linux-gnu"
            SUFFIX ".so"
            PREFIX "${PROJECT_NAME}"
        )

