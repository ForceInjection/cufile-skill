/*
 * 05-end-to-end-pipeline.cu — Double-Buffered Prefetch Pipeline
 *
 * Demonstrates:
 *   1. Double-buffered async read + compute pipeline using two GPU buffers
 *   2. cuFileStreamRegister / StreamDeregister for async I/O
 *   3. cuFileReadAsync with pointer-to-size semantics (cuFile v1.13)
 *   4. CUDA event-based pipeline synchronization
 *
 * Pattern:
 *   Stream I/O:  [Read B0]            [Read B1]            [Read B2]
 *   Stream Comp:          [Process B0]         [Process B1]         [Process B2]
 *
 * Compile: nvcc -O2 -o 05-end-to-end-pipeline 05-end-to-end-pipeline.cu common/cufile_utils.cu -lcufile -lcuda
 * Run:     sudo ./05-end-to-end-pipeline /mnt/nvme/testfile
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <cufile.h>
#include <cuda_runtime.h>
#include "common/cufile_utils.h"

#define CHUNK_SIZE   (8 * 1024 * 1024)  // 8 MB per chunk
#define NUM_CHUNKS   4                    // Process 4 chunks

/* Simple compute kernel: XOR every byte */
__global__ void transform_kernel(unsigned char *buf, size_t size,
                                  unsigned char key) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        buf[idx] ^= key;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <file_path_on_gds_mount>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char *file_path = argv[1];

    printf("=== Double-Buffered Pipeline Example ===\n\n");

    /* ── Init ───────────────────────────────────────────────── */
    CUfileError_t st = cuFileDriverOpen();
    cuFileCheck(st, "cuFileDriverOpen");

    /* Allocate TWO GPU buffers for double-buffering */
    unsigned char *buf[2];
    cudaMalloc(&buf[0], CHUNK_SIZE);
    cudaMalloc(&buf[1], CHUNK_SIZE);
    cuFileBufRegister(buf[0], CHUNK_SIZE, 0);
    cuFileBufRegister(buf[1], CHUNK_SIZE, 0);

    int fd = open_direct(file_path, O_CREAT | O_RDWR, 0644);
    preallocate_file(fd, NUM_CHUNKS * CHUNK_SIZE);

    CUfileDescr_t descr;
    memset(&descr, 0, sizeof(descr));
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    descr.handle.fd = fd;
    CUfileHandle_t fh;
    st = cuFileHandleRegister(&fh, &descr);
    cuFileCheck(st, "cuFileHandleRegister");

    /* ── Create streams ─────────────────────────────────────── */
    CUstream io_stream, comp_stream;
    cudaStreamCreate(&io_stream);
    cudaStreamCreate(&comp_stream);
    cuFileStreamRegister(io_stream, CU_FILE_STREAM_PAGE_ALIGNED_INPUTS);

    cudaEvent_t io_done[2];     // One event per buffer slot
    cudaEventCreate(&io_done[0]);
    cudaEventCreate(&io_done[1]);

    int block_size = 256;
    int grid_size  = (CHUNK_SIZE + block_size - 1) / block_size;

    /* ── Pipeline loop ──────────────────────────────────────── */
    printf("Starting pipeline: %d chunks × %s\n",
           NUM_CHUNKS, format_size(CHUNK_SIZE));

    for (int i = 0; i < NUM_CHUNKS; i++) {
        int cur  = i % 2;          // Current buffer index
        off_t off = (off_t)(i * CHUNK_SIZE);

        /* If not the first chunk, wait for previous I/O to finish */
        if (i > 0) {
            cudaStreamWaitEvent(comp_stream, io_done[cur], 0);
        }

        /* Launch async read for this chunk */
        size_t  read_sz  = CHUNK_SIZE;
        off_t   rd_off   = off;
        off_t   rd_buf   = 0;
        ssize_t rd_bytes = 0;
        st = cuFileReadAsync(fh, buf[cur], &read_sz, &rd_off,
                             &rd_buf, &rd_bytes, io_stream);
        if (st.err != CU_FILE_SUCCESS) {
            fprintf(stderr, "ReadAsync chunk %d failed: %s\n",
                    i, CUFILE_ERRSTR(st.err));
        }
        cudaEventRecord(io_done[cur], io_stream);

        /* While I/O runs, process the PREVIOUS buffer on comp_stream */
        if (i > 0) {
            int prev = (i - 1) % 2;
            transform_kernel<<<grid_size, block_size, 0, comp_stream>>>(
                buf[prev], CHUNK_SIZE, (unsigned char)(i * 7 + 3));
        }

        printf("  Chunk %d: read submitted, prev chunk processing...\n", i);
    }

    /* Process the last chunk */
    cudaStreamSynchronize(io_stream);
    int last = (NUM_CHUNKS - 1) % 2;
    transform_kernel<<<grid_size, block_size>>>(buf[last], CHUNK_SIZE,
                                                  (unsigned char)(NUM_CHUNKS * 7 + 3));
    cudaDeviceSynchronize();

    printf("Pipeline complete: %d chunks processed\n", NUM_CHUNKS);

    /* ── Cleanup ────────────────────────────────────────────── */
    cuFileStreamDeregister(io_stream);
    cudaEventDestroy(io_done[0]);
    cudaEventDestroy(io_done[1]);
    cudaStreamDestroy(io_stream);
    cudaStreamDestroy(comp_stream);
    cuFileHandleDeregister(fh);
    close(fd);
    cuFileBufDeregister(buf[0]);
    cuFileBufDeregister(buf[1]);
    cudaFree(buf[0]);
    cudaFree(buf[1]);
    cuFileDriverClose();

    printf("\n=== Example Complete ===\n");
    return EXIT_SUCCESS;
}
