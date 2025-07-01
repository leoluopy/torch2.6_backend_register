cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(extendBackEnd_FOR_CODE_INDEX)

# compile options
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 17)

include_directories("/home/leo/.pyenv/versions/anaconda3-2023.09-0/include/python3.11/")

# package
#find_package(Torch REQUIRED PATHS "/root/Downloads/libtorch-cxx11-abi-shared-with-deps-2.1.2+cpu/libtorch")
find_package(Torch REQUIRED PATHS "/home/leo/Downloads/dev/libtorch-cxx11-abi-shared-with-deps-2.1.2+cpu/share/cmake/Torch")

add_executable(extendBackEnd_FOR_CODE_INDEX torch_tcu.cpp)
# libtorch
target_link_libraries(extendBackEnd_FOR_CODE_INDEX ${TORCH_LIBRARIES})
target_link_libraries(extendBackEnd_FOR_CODE_INDEX pthread)
