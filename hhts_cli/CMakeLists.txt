# Copyright (c) Technische Hochschule Nürnberg, Game Tech Lab.
# All rights reserved.

cmake_minimum_required (VERSION 2.8)
project (superpixel_benchmark)

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS system filesystem program_options timer REQUIRED)

include_directories(../lib_eval/ ../lib_hhts/
    ${OpenCV_INCLUDE_DIRS} 
    ${Boost_INCLUDE_DIRS}
)
add_executable(hhts_cli main.cpp)
target_link_libraries(hhts_cli
    eval hhts
    ${Boost_LIBRARIES}
    ${OpenCV_LIBS}
)
