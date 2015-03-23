cmake_minimum_required(VERSION 3.1)
project(nn)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(nn ${SOURCE_FILES} nn_batch.h nn_batch.cpp common.h nn_seq.h nn_seq.cpp)