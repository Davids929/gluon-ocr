# C++ inference demo
This is a demo application which illustrates how to use existing GluonCV models in c++ environments given exported JSON and PARAMS files. 
Please checkout export for instructions of how to export pre-trained models.

## Build from source
We will go through with cpu versions, gpu versions of mxnet are similar but requires USE_CUDA=1 and USE_CUDNN=1 (optional). See MXNet website if interested.

### Linux
We use Ubuntu as example in Linux section.

1. Install build tools and git
```bash
sudo apt-get update
sudo apt-get install -y build-essential git
# install openblas
sudo apt-get install -y libopenblas-dev
# install cmake
sudo apt-get install -y cmake
```

2. Download opencv source and build shared library
```bash
cd ~
wget https://github.com/opencv/opencv/archive/3.4.7.tar.gz
tar -xf 3.4.7.tar.gz
cd opencv-3.4.7
mkdir build
cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=~/opencv3 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DWITH_IPP=OFF \
    -DBUILD_IPP_IW=OFF \
    -DWITH_LAPACK=OFF \
    -DWITH_EIGEN=OFF \
    -DCMAKE_INSTALL_LIBDIR=lib64 \
    -DWITH_ZLIB=ON \
    -DBUILD_ZLIB=ON \
    -DWITH_JPEG=ON \
    -DBUILD_JPEG=ON \
    -DWITH_PNG=ON \
    -DBUILD_PNG=ON \

make -j
make install
```

3. Download MXNet source and build shared library
```bash
cd ~
git clone --recursive https://github.com/apache/incubator-mxnet.git
cd incubator-mxnet
make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CPP_PACKAGE=1
```

4. Build gluon-ocr demo
```bash
cd ~
git clone https://github.com/Davids929/gluon-ocr.git
cd gluon-ocr/scripts/deploy/cpp-infer
mkdir build
cd build
cmake .. -DMXNET_ROOT=~/incubator-mxnet -DOPENCV_DIR=~/opencv3
make -j4
```

5. Demo usage
```bash
./gluon-ocr ../config.txt ../test.jpg
```



