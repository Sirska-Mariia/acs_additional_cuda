cmake_minimum_required(VERSION 3.10)
project(integrate_parallel)

enable_language(CUDA)
#add_executable(integrate_parallel integrate_parallel.cu cuda.cu)
#set_target_properties(integrate_parallel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_61")


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(integrate_parallel cuda.cu integrate.cu)

target_link_libraries(integrate_parallel m)

set_target_properties(integrate_parallel PROPERTIES OUTPUT_NAME "integrate_parallel")
