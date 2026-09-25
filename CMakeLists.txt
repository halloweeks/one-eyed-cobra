cmake_minimum_required(VERSION 3.16)

project(
    OneEyedCobra 
    VERSION 1.0.0
    LANGUAGES C
)


set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_library(oec
	src/oec.c
    src/oec_tensor.c
    src/oec_image.c
    src/oec_model.c
    src/oec_conv.c
    src/oec_activation.c
    src/oec_layer.c
    src/oec_loss.c
    src/oec_optimizer.c
    src/oec_dataset.c
    src/oec_target.c
)

target_include_directories(oec PUBLIC include)

add_executable(oec_train
    examples/oec_train.c
)

add_executable(oec_predict
    examples/oec_predict.c
)

add_executable(oec_finetune
    examples/oec_finetune.c
)


target_link_libraries(oec_train oec m)

target_link_libraries(oec_finetune oec m)

target_link_libraries(oec_predict oec m)

if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
        oec_train
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -march=native
    )
    
    target_compile_options(
        oec_finetune
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -march=native
    )
    
    target_compile_options(
        oec_predict
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -march=native
    )
endif()