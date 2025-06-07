# drivers/CMSIS/DSP/CMakeLists.txt

# 自动添加所有 DSP 源文件
file(GLOB_RECURSE DSP_SRCS
  Source/*/*.c
)

# 创建一个对象库（Object Library，不单独生成 .a/.so）
add_library(CMSISDSP OBJECT ${DSP_SRCS})

# 添加头文件路径（让主工程也能找到）
target_include_directories(CMSISDSP PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/Include
)

# 你也可以在这里添加特殊的编译选项（可选）
target_compile_definitions(CMSISDSP PRIVATE ARM_MATH_CM7)
target_compile_options(CMSISDSP PRIVATE -O3 -ffunction-sections -fdata-sections)
