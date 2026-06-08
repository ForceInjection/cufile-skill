cmake_minimum_required(VERSION 3.18)
project(cufile-examples LANGUAGES C CXX CUDA
    DESCRIPTION "cuFile / GPUDirect Storage programming examples"
    HOMEPAGE_URL "https://docs.nvidia.com/gpudirect-storage/"
)

# ── Prerequisite Checks ─────────────────────────────────────────

message(STATUS "=== cuFile Examples Build Configuration ===")
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES} (override with -DCMAKE_CUDA_ARCHITECTURES=\"80;89;90\")")

# Check for nvidia-fs kernel module (informational only — build can succeed without it)
execute_process(
    COMMAND bash -c "lsmod | grep -q nvidia_fs && echo 'YES' || echo 'NO'"
    OUTPUT_VARIABLE NFS_LOADED
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NFS_LOADED STREQUAL "YES")
    message(STATUS "nvidia-fs kernel module: LOADED")
else()
    message(WARNING "nvidia-fs kernel module NOT loaded. Build will succeed, "
                    "but examples will fail at runtime. Run: sudo modprobe nvidia-fs")
endif()

# Check for /etc/cufile.json
if(EXISTS "/etc/cufile.json")
    message(STATUS "Config file: /etc/cufile.json found")
else()
    message(WARNING "Config file /etc/cufile.json NOT found. "
                    "Create it or set CUFILE_CONFIG env var. See references/configuration.md")
endif()

# Require CUDA toolkit (provides cufile.h and libcufile)
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION} (found at ${CUDAToolkit_BIN_DIR})")

# Verify CUDA toolkit version meets minimum requirement
if(CUDAToolkit_VERSION VERSION_LESS "11.4")
    message(FATAL_ERROR
        "CUDA Toolkit ${CUDAToolkit_VERSION} is too old. "
        "cuFile requires CUDA 11.4 or newer. "
        "Please install CUDA Toolkit 12.x from https://developer.nvidia.com/cuda-downloads")
endif()

# cuFile library is bundled with CUDA toolkit
find_library(CUFILE_LIBRARY cufile
    PATHS ${CUDAToolkit_LIBRARY_DIR}
          /usr/local/cuda/lib64
          /usr/local/cuda/lib
          /opt/cuda/lib64
    DOC "cuFile library for GPUDirect Storage"
)
if(NOT CUFILE_LIBRARY)
    message(FATAL_ERROR
        "libcufile NOT found. cuFile is required to build these examples.\n"
        "Searched: ${CUDAToolkit_LIBRARY_DIR}, /usr/local/cuda/lib64, /usr/local/cuda/lib\n"
        "Install CUDA Toolkit 12.x: https://developer.nvidia.com/cuda-downloads\n"
        "Or install nvidia-gds package: sudo apt install nvidia-gds-12-0 (Ubuntu/Debian)")
endif()
message(STATUS "libcufile: ${CUFILE_LIBRARY}")

# CUDA Driver API (for cuMemAlloc etc.)
find_library(CUDA_LIBRARY cuda
    PATHS ${CUDAToolkit_LIBRARY_DIR}
          /usr/local/cuda/lib64
          /usr/local/cuda/lib
    DOC "CUDA Driver API library"
    REQUIRED
)

# ── GPU Architecture Detection ──────────────────────────────────

if(NOT CMAKE_CUDA_ARCHITECTURES)
    # Try to auto-detect from nvidia-smi
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE GPU_CCS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(GPU_CCS)
        # Convert "8.9\n9.0" → "89;90" format
        string(REGEX REPLACE "\\." "" GPU_ARCHS "${GPU_CCS}")
        string(REGEX REPLACE "\n" ";" GPU_ARCHS "${GPU_ARCHS}")
        list(REMOVE_DUPLICATES GPU_ARCHS)
        set(CMAKE_CUDA_ARCHITECTURES "${GPU_ARCHS}" CACHE STRING
            "CUDA architectures (auto-detected from nvidia-smi)")
        message(STATUS "Auto-detected GPU architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        # Default to Ampere (SM 8.0) as a safe baseline
        set(CMAKE_CUDA_ARCHITECTURES "80" CACHE STRING
            "CUDA architectures (could not auto-detect, defaulting to sm_80)")
        message(WARNING "Could not auto-detect GPU. Defaulting to sm_80. "
                        "Override with: -DCMAKE_CUDA_ARCHITECTURES=\"your_arch\"")
    endif()
endif()

# Verify architectures are GDS-compatible (SM 6.0+)
foreach(ARCH ${CMAKE_CUDA_ARCHITECTURES})
    if(ARCH VERSION_LESS "60")
        message(WARNING "Architecture sm_${ARCH} is < sm_60 (Pascal). "
                        "GDS requires Pascal (SM 6.0) or newer GPU.")
    endif()
endforeach()

# ── Compiler Flags ──────────────────────────────────────────────

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 --use_fast_math")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -Wall")

# Enable line info for profiling
if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo" OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif()

# ── Common Utility Library ──────────────────────────────────────

add_library(cufile_utils STATIC
    common/cufile_utils.cu
)
target_include_directories(cufile_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(cufile_utils PUBLIC
    ${CUFILE_LIBRARY}
    ${CUDA_LIBRARY}
    CUDA::cudart
)

# ── Individual Examples ─────────────────────────────────────────

# Each example is an independent executable linked against cufile_utils
# Compile: cmake -B build && cmake --build build -j$(nproc)
# Run:     sudo build/01-driver-init

# Example 01: Driver lifecycle
add_executable(01-driver-init 01-driver-init.cu)
target_link_libraries(01-driver-init PRIVATE cufile_utils)

# Example 02: Synchronous read/write
add_executable(02-sync-read-write 02-sync-read-write.cu)
target_link_libraries(02-sync-read-write PRIVATE cufile_utils)

# Example 03: Asynchronous I/O
add_executable(03-async-read-write 03-async-read-write.cu)
target_link_libraries(03-async-read-write PRIVATE cufile_utils)

# Example 04: Batch I/O
add_executable(04-batch-io 04-batch-io.cu)
target_link_libraries(04-batch-io PRIVATE cufile_utils)

# Example 05: End-to-end pipeline
add_executable(05-end-to-end-pipeline 05-end-to-end-pipeline.cu)
target_link_libraries(05-end-to-end-pipeline PRIVATE cufile_utils)

# Example 06: Alignment check
add_executable(06-alignment-check 06-alignment-check.cu)
target_link_libraries(06-alignment-check PRIVATE cufile_utils)

# ── Build Summary ───────────────────────────────────────────────

message(STATUS "")
message(STATUS "=== Build Configuration Complete ===")
message(STATUS "Examples will be built to: ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "To build and run:")
message(STATUS "  cmake -B build && cmake --build build -j$(nproc)")
message(STATUS "  sudo build/01-driver-init")
message(STATUS "  sudo build/02-sync-read-write /mnt/nvme/testfile")
message(STATUS "")
message(STATUS "If examples fail at runtime:")
message(STATUS "  1. Run: sudo modprobe nvidia-fs")
message(STATUS "  2. Run: gdscheck -p")
message(STATUS "  3. Run: bash ../scripts/check_gds.sh /mnt/nvme")
message(STATUS "  4. See: references/hardware-requirements.md")
