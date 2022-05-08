#!/bin/bash

# g2opy
cd thirdparty/g2opy
mkdir -p build
cd build
cmake ..
make -j2
cd ..
python setup.py install


# lambdatwist PnP
cd ../lambdatwist
mkdir -p build
cd build
cmake ..
make -j2
cd ..
python setup.py install
