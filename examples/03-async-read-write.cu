/*
 * 03-async-read-write.cu — Asynchronous cuFile I/O via CUDA Streams
 *
 * Demonstrates:
 *   1. cuFileStreamRegister — associate CUDA streams with cuFile
 *   2. cuFileReadAsync / cuFileWriteAsync — non-blocking I/O
 *   3. Overlapping I/O with GPU compute kernels
 *   4. CUDA event-based synchronization between I/O and compute streams
 *
 * Compile: nvcc -O2 -o 03-async-read-write 03-async-read-write.cu common/cufile_utils.cu -lcufile -lcuda
 * Run:     sudo ./03-async-read-write /mnt/nvme/testfile
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
    const char *filepath = (argc > 1) ? argv[1] : "/mnt/nvme/testfile";
    printf("=== Asynchronous cuFile I/O Example ===\n\n");

    int threads = 256;
    int blocks = (TEST_SIZE + threads - 1) / threads;

    /* ── 1. Initialize ────────────────────────────────────── */
    printf("1. Initializing...\n");
    cuFileDriverOpen();
    check_gds_available();

    CUdeviceptr devPtr;
    cuMemAlloc(&devPtr, TEST_SIZE);
    cuFileBufRegister(devPtr, TEST_SIZE, 0);

    int fd = open(filepath, O_DIRECT | O_RDWR | O_CREAT, 0644);
    ftruncate(fd, TEST_SIZE);

    CUfileDescr_t fh = {0};
    fh.cookie = (CUfileDriverCookie)(uintptr_t)fd;
    cuFileHandleRegister(&fh, NULL);
    printf("   Initialized.\n\n");

    /* ── 2. Create streams ────────────────────────────────── */
    printf("2. Creating CUDA streams...\n");
    cudaStream_t io_stream, compute_stream;
    cudaStreamCreate(&io_stream);
    cudaStreamCreate(&compute_stream);
    printf("   io_stream=%p, compute_stream=%p\n",
           (void *)io_stream, (void *)compute_stream);

    /* ── 3. Register streams with cuFile ──────────────────── */
    printf("3. Registering streams with cuFile...\n");
    CUfileError_t st = cuFileStreamRegister(fh, io_stream);
    if (st.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "   FAILED: %s\n", cuFileGetErrorString(st));
        return EXIT_FAILURE;
    }
    printf("   Stream registered.\n\n");

    /* ── 4. Async read + compute overlap demo ─────────────── */
    printf("4. Async read with compute overlap...\n");
    cudaEvent_t read_done, compute_done;
    cudaEventCreate(&read_done);
    cudaEventCreate(&compute_done);

    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);

    for (size_t off = 0; off < TEST_SIZE; off += IO_SIZE) {
        size_t chunk = (off + IO_SIZE <= TEST_SIZE) ? IO_SIZE : (TEST_SIZE - off);

        /* Submit async read on io_stream */
        st = cuFileReadAsync(fh, devPtr + off, chunk, off, 0, io_stream);
        if (st.err != CU_FILE_SUCCESS) {
            fprintf(stderr, "   cuFileReadAsync failed at %zu: %s\n",
                    off, cuFileGetErrorString(st));
            break;
        }

        /* Record event on io_stream when read completes */
        cudaEventRecord(read_done, io_stream);

        /* Compute stream waits for read to finish before processing */
        cudaStreamWaitEvent(compute_stream, read_done, 0);

        /* Launch kernel on compute stream (data already on GPU!) */
        process_kernel<<<blocks, threads, 0, compute_stream>>>(
            (unsigned char *)devPtr + off, chunk, 0xFF);

        /* Record compute completion */
        cudaEventRecord(compute_done, compute_stream);

        /* Next read can start — no dependency from compute
         * (compute operates on data already in buffer at 'off') */
    }

    /* Wait for all work to complete */
    cudaStreamSynchronize(io_stream);
    cudaStreamSynchronize(compute_stream);

    clock_gettime(CLOCK_MONOTONIC, &t2);

    double elapsed = (t2.tv_sec - t1.tv_sec) +
                     (t2.tv_nsec - t1.tv_nsec) / 1e9;
    double bw = measure_bandwidth(TEST_SIZE, elapsed);
    printf("   Complete: %.2f GB/s (read + process, overlapped)\n\n", bw);

    /* ── 5. Async write demo ──────────────────────────────── */
    printf("5. Async write back to file...\n");

    clock_gettime(CLOCK_MONOTONIC, &t1);

    for (size_t off = 0; off < TEST_SIZE; off += IO_SIZE) {
        size_t chunk = (off + IO_SIZE <= TEST_SIZE) ? IO_SIZE : (TEST_SIZE - off);

        /* Write directly from GPU memory — no CPU copy needed */
        st = cuFileWriteAsync(fh, devPtr + off, chunk, off, 0, io_stream);
        if (st.err != CU_FILE_SUCCESS) {
            fprintf(stderr, "   cuFileWriteAsync failed: %s\n",
                    cuFileGetErrorString(st));
            break;
        }
    }

    cudaStreamSynchronize(io_stream);

    clock_gettime(CLOCK_MONOTONIC, &t2);
    double write_elapsed = (t2.tv_sec - t1.tv_sec) +
                           (t2.tv_nsec - t1.tv_nsec) / 1e9;
    double write_bw = measure_bandwidth(TEST_SIZE, write_elapsed);
    printf("   Write complete: %.2f GB/s\n\n", write_bw);

    /* ── 6. Cleanup ───────────────────────────────────────── */
    printf("6. Cleanup...\n");
    cudaEventDestroy(read_done);
    cudaEventDestroy(compute_done);
    cudaStreamDestroy(io_stream);
    cudaStreamDestroy(compute_stream);

    cuFileHandleDeregister(fh);
    close(fd);
    cuFileBufDeregister(devPtr);
    cuMemFree(devPtr);
    cuFileDriverClose();

    printf("\n=== Example Complete ===\n");
    printf("Read+Process: %.2f GB/s | Write: %.2f GB/s\n", bw, write_bw);
    return EXIT_SUCCESS;
}
