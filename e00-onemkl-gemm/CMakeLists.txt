project(e00 CXX)

set(BASE_FLAGS
        $<$<CONFIG:Debug>:-Wall -Wextra -Wno-unused-parameter -fno-common -g -ggdb -Wno-unknown-pragmas -fsycl -fsycl-link -Xsycl-target-backend --cuda-gpu-arch=sm_80># -fsycl-device-code-split=per_kernel > #$<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>
        $<$<CONFIG:Release>:-Wall -Wextra -Wno-unused-parameter -fno-common -O3 -funroll-loops -Wno-unknown-pragmas -fsycl -fsycl-link -Xsycl-target-backend --cuda-gpu-arch=sm_80> # -fsycl-device-code-split=per_kernel > #$<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>
)
set(BASE_LINKER_FLAGS
        $<$<CONFIG:Debug>:-fsycl  -Xsycl-target-backend --cuda-gpu-arch=sm_80> # -fsycl-device-code-split=per_kernel>
        $<$<CONFIG:Release>:-fsycl  -Xsycl-target-backend --cuda-gpu-arch=sm_80> # -fsycl-device-code-split=per_kernel>
)

find_package(MKL REQUIRED)


add_executable(e00
        main.cpp
)
target_compile_options(e00 PRIVATE ${BASE_FLAGS})
target_link_options(e00 PRIVATE ${BASE_LINKER_FLAGS})
target_link_libraries(
        e00
        PRIVATE
        sycl
        MKL::MKL_SYCL::BLAS
)

