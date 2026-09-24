cmake_minimum_required(VERSION 3.16)

project(
    OneEyedCobra 
    VERSION 1.0.0
    LANGUAGES C
)


set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)


add_executable(
	oec
	
	src/main.c
	src/oec_tensor.c
	src/oec_image.c
	src/oec_model.c
	src/oec_conv.c
    src/oec_activation.c
    src/oec_layer.c
    src/oec_loss.c
    src/oec_optimizer.c
    src/oec_dataset.c
)


add_executable(
	oec_run
	
	src/run.c
	src/oec_tensor.c
	src/oec_image.c
	src/oec_model.c
	src/oec_conv.c
    src/oec_activation.c
    src/oec_layer.c
    src/oec_loss.c
    src/oec_optimizer.c
    src/oec_dataset.c
)


target_include_directories(
    oec
    PRIVATE
        include
)

target_include_directories(
    oec_run
    PRIVATE
        include
)


target_link_libraries(
    oec
    PRIVATE
        m
)

target_link_libraries(
    oec_run
    PRIVATE
        m
)


if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
        oec
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -march=native
            -Rpass=loop-vectorize
            -Rpass-missed=loop-vectorize
            -O3
    )
    
    target_compile_options(
        oec_run
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -march=native
            -Rpass=loop-vectorize
            -Rpass-missed=loop-vectorize
            -O2
    )
endif()