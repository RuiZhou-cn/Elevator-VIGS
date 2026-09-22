cmake_minimum_required(VERSION 3.10)
project(imu_integrator_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
# Auto-detect pybind11 from the Python that runs scripts/install.sh (-DPYTHON_EXECUTABLE), or from the
# active environment when configured by hand
if(NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE python)
endif()
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED)

# ----------------------------------------------------------------------
# SOPHUS (header-only, vendored in thirdparty/Sophus -- no global install)
# ----------------------------------------------------------------------
# Sophus is used header-only, so we just add its include directory rather
# than find_package()-ing a system/global install. The headers ship with the
# repository as a git submodule.
set(SOPHUS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/Sophus")
if(NOT EXISTS "${SOPHUS_INCLUDE_DIR}/sophus/so3.hpp")
    message(FATAL_ERROR
        "Sophus headers not found at ${SOPHUS_INCLUDE_DIR}.\n"
        "Fetch the submodule from the repository root:\n"
        "  git submodule update --init thirdparty/Sophus")
endif()

pybind11_add_module(imu_integrator_cpp
    imu_integrator.cpp
    imu_integrator_bindings.cpp
)

target_include_directories(imu_integrator_cpp PRIVATE ${SOPHUS_INCLUDE_DIR})

target_link_libraries(imu_integrator_cpp
    PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX
)

# Optimization flags. Deliberately NOT -ffast-math and NOT -march=native:
#   -ffast-math implies -ffinite-math-only, under which the compiler may fold the dT
#     isnan/isinf guard in imu_integrator_bindings.cpp to a constant false, and
#     -funsafe-math-optimizations, which lets it reassociate the preintegration sums;
#   -march=native makes the built .so non-portable and its results machine-dependent.
# x86-64-v3 still gives AVX2 + FMA (what -march=native was here for) on any Haswell-or-later CPU.
target_compile_options(imu_integrator_cpp PRIVATE
    -O3                    # Maximum optimization
    -march=x86-64-v3       # AVX2, FMA -- fixed baseline, so the build is reproducible
    -funroll-loops         # Unroll loops
    -ftree-vectorize       # Auto-vectorization
)

target_compile_definitions(imu_integrator_cpp PRIVATE
    EIGEN_NO_DEBUG            # Disable Eigen debug checks
    NDEBUG                   # Disable assertions
    SOPHUS_USE_BASIC_LOGGING # Avoid Sophus's optional fmt dependency (use printf logging)
    SOPHUS_DISABLE_ENSURES   # Disable Sophus runtime checks (SO3-from-matrix orthogonality
                             # check is too strict for numerically-drifted rotations; the code
                             # in imu_integrator.cpp explicitly relies on this being disabled)
)
