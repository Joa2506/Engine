# src/CMakeLists.txt


# set the project name
project("Hello_world")
cmake_minimum_required(VERSION 3.10)

#links libraries
#target_link_libraries(${TARGET_NAME} nvonnxparser)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/utils/cuda" )
set(TensorRT_DIR "/usr/src/tensorrt")
find_package(CUDA)
message("-- CUDA version: ${CUDA_VERSION}")

set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS}; 
    -O3 
	-gencode arch=compute_53,code=sm_53
	-gencode arch=compute_62,code=sm_62
)

if(CUDA_VERSION_MAJOR GREATER 9)
	message("-- CUDA ${CUDA_VERSION_MAJOR} detected, enabling SM_72")

	set(
		CUDA_NVCC_FLAGS
		${CUDA_NVCC_FLAGS}; 
		-gencode arch=compute_72,code=sm_72
	)
endif()

set(TensorRT_DIR "/usr/src/tensorrt")

# specify the executable and corresponding source file
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})  #This seems to have added the cuda_runtime api.
include_directories(${TensorRT_INCLUDE_DIR})
include_directories(${TensorRT_DIR}/samples/common)
add_executable(app main.cpp TensorEngine.cpp)

#links libraries
target_link_libraries(app nvonnxparser)
target_link_libraries(app nvinfer${NVINFER_LIBRARY})
target_link_libraries(app )
target_link_libraries(app ${CUDA_LIBRARIES})
