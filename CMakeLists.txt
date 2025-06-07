cmake_minimum_required(VERSION 3.20)
project(TinyML_Keyword C ASM)

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR cortex-m7)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# 工具链设置
set(CMAKE_C_COMPILER /opt/gcc-arm-none-eabi/bin/arm-none-eabi-gcc)
set(CMAKE_ASM_COMPILER /opt/gcc-arm-none-eabi/bin/arm-none-eabi-gcc)
set(CMAKE_OBJCOPY /opt/gcc-arm-none-eabi/bin/arm-none-eabi-objcopy)

# 编译参数
set(CPU_FLAGS "-mcpu=cortex-m7 -mthumb -mfloat-abi=hard -mfpu=fpv5-d16")
set(COMMON_FLAGS "-Wall -O2 -ffunction-sections -fdata-sections")
set(CMAKE_C_FLAGS "${CPU_FLAGS} ${COMMON_FLAGS}")
set(CMAKE_ASM_FLAGS "${CPU_FLAGS} ${COMMON_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CPU_FLAGS} -T${CMAKE_SOURCE_DIR}/STM32F746XX_FLASH.ld -Wl,--gc-sections")

# 宏定义
add_definitions(-DARM_MATH_CM7 -DUSE_HAL_DRIVER -DSTM32F746xx)

# 包含路径
include_directories(
  core/inc
  drivers/CMSIS/Core/Include
  drivers/CMSIS/Device/ST/STM32F7xx/Include
  drivers/STM32F7xx_HAL_Driver/Inc
  drivers/STM32F7xx_HAL_Driver/Inc/Legacy
  drivers/CMSIS/DSP/Include
)

# 源码列表
file(GLOB CORE_SRCS      core/src/*.c)
file(GLOB STARTUP_SRCS   startup/*.s)
file(GLOB HAL_SRCS drivers/STM32F7xx_HAL_Driver/Src/*.c)

# 排除 template 文件（避免重定义冲突）
list(REMOVE_ITEM HAL_SRCS
  ${CMAKE_SOURCE_DIR}/drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_msp_template.c
  ${CMAKE_SOURCE_DIR}/drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_timebase_rtc_alarm_template.c
  ${CMAKE_SOURCE_DIR}/drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_timebase_rtc_wakeup_template.c
)


# ✅ 只包含你需要的 CMSIS-DSP 源码（避免重复定义）
set(DSP_SRCS
  drivers/CMSIS/DSP/Source/TransformFunctions/arm_cfft_q15.c
  drivers/CMSIS/DSP/Source/TransformFunctions/arm_cfft_radix4_q15.c
  drivers/CMSIS/DSP/Source/BasicMathFunctions/arm_mult_q15.c
  drivers/CMSIS/DSP/Source/BasicMathFunctions/arm_abs_q15.c
)

# 添加目标
add_executable(${PROJECT_NAME}
  ${CORE_SRCS}
  ${STARTUP_SRCS}
  ${HAL_SRCS}
  ${DSP_SRCS}
)

# 生成 bin/hex 文件
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
  COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:${PROJECT_NAME}> ${PROJECT_NAME}.bin
  COMMAND ${CMAKE_OBJCOPY} -O ihex   $<TARGET_FILE:${PROJECT_NAME}> ${PROJECT_NAME}.hex
)
