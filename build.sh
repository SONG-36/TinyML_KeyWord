#!/bin/bash
set -e  # 一旦出错就退出

#!/bin/bash
cmake -B build -DCMAKE_TOOLCHAIN_FILE=./toolchain-arm-none-eabi.cmake
cmake --build build -j$(nproc)

