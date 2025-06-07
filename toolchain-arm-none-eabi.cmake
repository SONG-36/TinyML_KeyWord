# toolchain-arm-none-eabi.cmake

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

# 编译器路径
set(CMAKE_C_COMPILER arm-none-eabi-gcc)
set(CMAKE_CXX_COMPILER arm-none-eabi-g++)
set(CMAKE_ASM_COMPILER arm-none-eabi-gcc)

# 设置 sysroot 为标准 C 库路径（解决 stdint.h / __STATIC_INLINE 问题核心）
set(CMAKE_SYSROOT /opt/gcc-arm-none-eabi/arm-none-eabi)
set(CMAKE_FIND_ROOT_PATH /opt/gcc-arm-none-eabi/arm-none-eabi)

# 加入标准头文件和 GCC 内部头文件路径（确保 stdint.h, stddef.h 被识别）
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I${CMAKE_SYSROOT}/include -I/opt/gcc-arm-none-eabi/lib/gcc/arm-none-eabi/10.3.1/include")

# 禁用 host 编译工具查找，仅使用交叉编译工具
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# 避免 CMake 默认执行 native 编译测试，改为静态库
set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
