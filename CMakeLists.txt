cmake_minimum_required(VERSION 2.8)
project(param_server)

include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})

include(FindMPI)
include_directories(${MPI_CXX_INCLUDE_PATH})

#set(SOURCES message.pb.cc mpi_send.cpp)
file(GLOB SOURCES "*.cpp" "*.cc")
add_executable(param_server ${SOURCES})

set (CMAKE_CXX_FLAGS "--std=gnu++11 ${CMAKE_CXX_FLAGS}")

target_link_libraries(param_server ${PROTOBUF_LIBRARY} ${MPI_CXX_LIBRARIES})
