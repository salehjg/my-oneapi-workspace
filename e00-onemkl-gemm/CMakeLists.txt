cmake_minimum_required(VERSION 3.28)
project(e00 CXX)

set(BASE_FLAGS
        $<$<CONFIG:Debug>:-Wall -Wextra -Wno-unused-parameter -fno-common -g -ggdb -Wno-unknown-pragmas -fsycl -fsycl-link -Xsycl-target-backend --cuda-gpu-arch=sm_89># -fsycl-device-code-split=per_kernel > #$<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>
        $<$<CONFIG:Release>:-Wall -Wextra -Wno-unused-parameter -fno-common -O3 -funroll-loops -Wno-unknown-pragmas -fsycl -fsycl-link -Xsycl-target-backend --cuda-gpu-arch=sm_89> # -fsycl-device-code-split=per_kernel > #$<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>
)
set(BASE_LINKER_FLAGS
        $<$<CONFIG:Debug>:-fsycl  -Xsycl-target-backend --cuda-gpu-arch=sm_89> # -fsycl-device-code-split=per_kernel>
        $<$<CONFIG:Release>:-fsycl  -Xsycl-target-backend --cuda-gpu-arch=sm_89> # -fsycl-device-code-split=per_kernel>
)

include(FetchContent)
set(BUILD_FUNCTIONAL_TESTS OFF)
set(BUILD_EXAMPLES OFF)
set(BUILD_EXAMPLES OFF)
set(ENABLE_CUBLAS_BACKEND ON)
set(ENABLE_CUFFT_BACKEND ON)
set(ENABLE_CURAND_BACKEND ON)
set(ENABLE_OMP_OFFLOAD ON)
FetchContent_Declare(
        open_onemkl
        GIT_REPOSITORY https://github.com/oneapi-src/oneMKL.git
        GIT_TAG 028d8ba
)
FetchContent_MakeAvailable(open_onemkl)

add_executable(e00
        main.cpp
)
target_compile_options(e00 PRIVATE ${BASE_FLAGS})
target_link_options(e00 PRIVATE ${BASE_LINKER_FLAGS})
target_link_libraries(
        e00
        PRIVATE
        onemkl
        onemkl_blas
        onemkl_dft
        onemkl_lapack
        onemkl_sparse_blas
        onemkl_rng
)

