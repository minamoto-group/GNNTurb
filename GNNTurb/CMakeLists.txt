cmake_minimum_required(VERSION 3.10)
project(GNNTurb)

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Xlinker --no-as-needed -Xlinker --add-needed")
list(APPEND CMAKE_PREFIX_PATH $ENV{TORCH_LIBRARIES})
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -m64 -Dlinux64 -DWM_ARCH_OPTION=64 -DWM_DP -DWM_LABEL_SIZE=32 -Wall -Wextra -Wold-style-cast -Wnon-virtual-dtor -Wno-unused-parameter -Wno-invalid-offsetof -Wno-attributes -O3  -DNoRepository -ftemplate-depth-100 -Wno-old-style-cast -fopenmp")

find_package(CUDA REQUIRED)
# libtorch
find_package(Torch REQUIRED)
find_package(TorchSparse REQUIRED)
find_package(TorchScatter REQUIRED)
find_package(TorchSplineConv REQUIRED)


# Make/files
add_library(GNNTurb SHARED makeTurbulenceModel.C)

# Make/options
target_link_directories(GNNTurb PUBLIC $ENV{FOAM_LIBBIN})
target_include_directories(GNNTurb PUBLIC $ENV{FOAM_SRC}/OpenFOAM/lnInclude)
target_include_directories(GNNTurb PUBLIC $ENV{FOAM_SRC}/OSspecific/POSIX/lnInclude)

# EXE_INC
target_include_directories(GNNTurb PUBLIC $ENV{FOAM_SRC}/TurbulenceModels/turbulenceModels/lnInclude)
target_include_directories(GNNTurb PUBLIC $ENV{FOAM_SRC}/TurbulenceModels/incompressible/lnInclude)
target_include_directories(GNNTurb PUBLIC $ENV{FOAM_SRC}/transportModels)
target_include_directories(GNNTurb PUBLIC $ENV{FOAM_SRC}/finiteVolume/lnInclude)
target_include_directories(GNNTurb PUBLIC $ENV{FOAM_SRC}/meshTools/lnInclude)

set(EXTRA_LIBS dl m)
target_link_libraries(GNNTurb PUBLIC $ENV{EXTRA_LIBS})
target_link_libraries(GNNTurb PUBLIC OpenFOAM)

# LIB_LIBS or EXE_LIBS
target_link_libraries(GNNTurb PUBLIC incompressibleTransportModels)
target_link_libraries(GNNTurb PUBLIC incompressibleTurbulenceModels)
target_link_libraries(GNNTurb PUBLIC turbulenceModels)
target_link_libraries(GNNTurb PUBLIC finiteVolume)
target_link_libraries(GNNTurb PUBLIC meshTools)

# libtorch
target_link_libraries(GNNTurb PUBLIC ${TORCH_LIBRARIES})
target_link_libraries(GNNTurb PUBLIC TorchSparse::TorchSparse)
target_link_libraries(GNNTurb PUBLIC TorchScatter::TorchScatter)
target_link_libraries(GNNTurb PUBLIC TorchSplineConv::TorchSplineConv)

target_compile_features(GNNTurb PUBLIC cxx_range_for)
set_property(TARGET GNNTurb PROPERTY CXX_STANDARD 14)

#https://www.cfd-online.com/Forums/openfoam-programming-development/205243-share-something-about-dev-openfoam-using-cmake-rather-than-wmake.html