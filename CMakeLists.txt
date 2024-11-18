cmake_minimum_required(VERSION 2.8)
project( cv-algorithms )
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(SOURCES
    src/bilinear_interpolation.cc
)

set(WARNING_FLAGS "-Wall" "-Wextra" "-Wpedantic" "-Werror")

foreach(SOURCE_FILE ${SOURCES})
get_filename_component(EXE_NAME ${SOURCE_FILE} NAME_WE)
add_executable(target/${EXE_NAME} ${SOURCE_FILE})
target_link_libraries(target/${EXE_NAME} ${OpenCV_LIBS} )
endforeach()

