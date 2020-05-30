
cmake_minimum_required (VERSION 3.5.0)
project(GMS_Matching)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")


find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)


set(SOURCES_DIR "src/")
# Headers
set(${PROJECT_NAME}_headers
    )

# Sources
set(${PROJECT_NAME}_sources
    ${SOURCES_DIR}/main.cxx
    )




add_executable(${PROJECT_NAME}
    ${${PROJECT_NAME}_sources}
    ${${PROJECT_NAME}_headers}
    )



target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
    )

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
    )

