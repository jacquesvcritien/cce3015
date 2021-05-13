################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/jvel0131-Part1SerialImprovement.cpp 

OBJS += \
./src/jvel0131-Part1SerialImprovement.o 

CPP_DEPS += \
./src/jvel0131-Part1SerialImprovement.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/bin/nvcc -DNDEBUG -O3   -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/bin/nvcc -DNDEBUG -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


