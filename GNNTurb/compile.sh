export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/libtorch/lib
rm -r build
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="/opt/libtorch" ..
cmake --build .

mkdir -p $FOAM_USER_LIBBIN
cp libGNNTurb.so $FOAM_USER_LIBBIN