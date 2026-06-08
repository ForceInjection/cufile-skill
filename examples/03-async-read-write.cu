/*
 * 03-async-read-write.cu — Asynchronous cuFile I/O via CUDA Streams
 *
 * Demonstrates:
 *   1. cuFileStreamRegister / cuFileStreamDeregister — associate streams
 *   2. cuFileReadAsync / cuFileWriteAsync — non-blocking I/O
 *   3. Overlapping I/O with GPU compute kernels
 *   4. CUDA event-based synchronization between I/O and compute streams
 *
 * Compile: nvcc -O2 -o 03-async-read-write 03-async-read-write.cu common/cufile_utils.cu -lcufile -lcuda
 * Run:     sudo ./03-async-read-write /mnt/nvme/testfile
 *
 * NOTE: cuFile v1.13 async API uses pointer-to-size/offset semantics.
 *       The size_p, file_offset_p, bufPtr_offset_p and bytes_read_p
 *       parameters are all pointers — the cuFile library reads/writes
 *       through them at submission and completion time.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <cufile.h>
#include <cuda_runtime.h>
#include "common/cufile_utils.h"

#define TEST_SIZE (64 * 1024 * 1024)
#define IO_SIZE   (4 * 1024 * 1024)

/* ── Simple GPU kernel for compute overlap demo ────────────────── */

__global__ void process_kernel(unsigned char *buf, size_t size,
                               unsigned char xor_val) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        buf[idx] ^= xor_val;
    }
}

/* ── Main ─────────────────────────────────────────────────────── */

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <file_path_on_gds_mount>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char *file_path = argv[1];

    printf("=== Async I/O Example ===\n\n");

    /* ── Driver init ────────────────────────────────────────── */
    CUfileError_t st = cuFileDriverOpen();
    cuFileCheck(st, "cuFileDriverOpen");

    /* ── GPU buffer ─────────────────────────────────────────── */
    unsigned char *gpu_buf;
    cudaMalloc(&gpu_buf, TEST_SIZE);
    cuFileBufRegister(gpu_buf, TEST_SIZE, 0);

    /* ── File setup ─────────────────────────────────────────── */
    int fd = open_direct(file_path, O_CREAT | O_RDWR, 0644);
    preallocate_file(fd, TEST_SIZE);

    CUfileDescr_t descr;
    memset(&descr, 0, sizeof(descr));
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    descr.handle.fd = fd;
    CUfileHandle_t fh;
    st = cuFileHandleRegister(&fh, &descr);
    cuFileCheck(st, "cuFileHandleRegister");

    /* ── Create streams and events ──────────────────────────── */
    CUstream io_stream, compute_stream;
    cudaStreamCreate(&io_stream);
    cudaStreamCreate(&compute_stream);
    cudaEvent_t io_done;
    cudaEventCreate(&io_done);

    /* Register streams with cuFile (reduces per-IO setup overhead) */
    st = cuFileStreamRegister(io_stream, CU_FILE_STREAM_PAGE_ALIGNED_INPUTS);
    if (st.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "cuFileStreamRegister failed: %s (err=%d)\n",
                CUFILE_ERRSTR(st.err), st.err);
        fprintf(stderr, "(Continuing — stream registration is optional)\n");
    }

    /* ── Fill GPU buffer with initial data ──────────────────── */
    int block_size = 256;
    int grid_size = (TEST_SIZE + block_size - 1) / block_size;
    process_kernel<<<grid_size, block_size>>>(gpu_buf, TEST_SIZE, 0xFF);
    cudaDeviceSynchronize();

    /* ── Async read + compute overlap demo ──────────────────── */
    printf("Starting async read + compute overlap...\n");

    /* Non-blocking read on io_stream */
    size_t  read_size      = TEST_SIZE;
    off_t   read_offset    = 0;
    off_t   read_buf_off   = 0;
    ssize_t bytes_read     = 0;
    st = cuFileReadAsync(fh, gpu_buf, &read_size, &read_offset,
                         &read_buf_off, &bytes_read, io_stream);
    cuFileCheck(st, "cuFileReadAsync");
    printf("  Async read submitted (%zu bytes pending)\n", read_size);

    /* While I/O is running, launch compute on a different stream */
    cudaStreamWaitEvent(compute_stream, io_done, 0);  // Will wait later
    process_kernel<<<grid_size, block_size, 0, compute_stream>>>(
        gpu_buf + TEST_SIZE/2, TEST_SIZE/2, 0xAA);
    printf("  Compute kernel launched on separate stream\n");

    /* Record event when I/O completes */
    cudaEventRecord(io_done, io_stream);

    /* Wait for both to finish */
    cudaStreamSynchronize(io_stream);
    cudaStreamSynchronize(compute_stream);
    printf("  Async read completed: %zd bytes\n", bytes_read);
    printf("  Compute kernel completed\n");

    /* ── Cleanup ────────────────────────────────────────────── */
    cuFileStreamDeregister(io_stream);
    cudaEventDestroy(io_done);
    cudaStreamDestroy(io_stream);
    cudaStreamDestroy(compute_stream);
    cuFileHandleDeregister(fh);
    close(fd);
    cuFileBufDeregister(gpu_buf);
    cudaFree(gpu_buf);
    cuFileDriverClose();

    printf("\n=== Example Complete ===\n");
    return EXIT_SUCCESS;
}
