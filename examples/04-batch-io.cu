/*
 * 04-batch-io.cu — Batch I/O for High-Throughput Small I/O
 *
 * Demonstrates:
 *   1. cuFileBatchIOSetUp — create batch handle
 *   2. cuFileBatchIOSubmit — submit multiple I/Os in one call
 *   3. cuFileBatchIOGetStatus — poll for completion
 *   4. cuFileBatchIODestroy — tear down batch handle
 *
 * Compile: nvcc -O2 -o 04-batch-io 04-batch-io.cu common/cufile_utils.cu -lcufile -lcuda
 * Run:     sudo ./04-batch-io /mnt/nvme/testfile
 *
 * NOTE: cuFile v1.13 batch API uses CUfileIOParams_t with mode=CUFILE_BATCH
 *       and CUfileIOEvents_t for completion status.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <cufile.h>
#include <cuda_runtime.h>
#include "common/cufile_utils.h"

#define BATCH_SIZE  16
#define CHUNK_SIZE  (256 * 1024)   // 256 KB per chunk
#define TOTAL_SIZE  (BATCH_SIZE * CHUNK_SIZE)

__global__ void fill_chunk_kernel(unsigned char *buf, size_t offset,
                                   size_t size, unsigned char val) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        buf[offset + idx] = val;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <file_path_on_gds_mount>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char *file_path = argv[1];

    printf("=== Batch I/O Example ===\n\n");

    /* ── Init ───────────────────────────────────────────────── */
    CUfileError_t st = cuFileDriverOpen();
    cuFileCheck(st, "cuFileDriverOpen");

    unsigned char *gpu_buf;
    cudaMalloc(&gpu_buf, TOTAL_SIZE);
    cuFileBufRegister(gpu_buf, TOTAL_SIZE, 0);

    int fd = open_direct(file_path, O_CREAT | O_RDWR, 0644);
    preallocate_file(fd, TOTAL_SIZE);

    CUfileDescr_t descr;
    memset(&descr, 0, sizeof(descr));
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    descr.handle.fd = fd;
    CUfileHandle_t fh;
    st = cuFileHandleRegister(&fh, &descr);
    cuFileCheck(st, "cuFileHandleRegister");

    /* ── Fill GPU buffer ────────────────────────────────────── */
    int block_size = 256;
    for (int i = 0; i < BATCH_SIZE; i++) {
        int grid_size = (CHUNK_SIZE + block_size - 1) / block_size;
        fill_chunk_kernel<<<grid_size, block_size>>>(
            gpu_buf, i * CHUNK_SIZE, CHUNK_SIZE, (unsigned char)(i + 1));
    }
    cudaDeviceSynchronize();

    /* ── Batch write ────────────────────────────────────────── */
    printf("Submitting batch write: %d chunks × %s...\n",
           BATCH_SIZE, format_size(CHUNK_SIZE));

    CUfileBatchHandle_t batch;
    st = cuFileBatchIOSetUp(&batch, BATCH_SIZE);
    cuFileCheck(st, "cuFileBatchIOSetUp");

    CUfileIOParams_t io_params[BATCH_SIZE];
    for (int i = 0; i < BATCH_SIZE; i++) {
        io_params[i].mode            = CUFILE_BATCH;
        io_params[i].u.batch.devPtr_base = gpu_buf + i * CHUNK_SIZE;
        io_params[i].u.batch.file_offset = (off_t)(i * CHUNK_SIZE);
        io_params[i].u.batch.devPtr_offset = 0;
        io_params[i].u.batch.size     = CHUNK_SIZE;
        io_params[i].fh              = fh;
        io_params[i].opcode          = CUFILE_WRITE;
        io_params[i].cookie          = (void *)(uintptr_t)i;  // user tag
    }

    st = cuFileBatchIOSubmit(batch, BATCH_SIZE, io_params, 0);
    cuFileCheck(st, "cuFileBatchIOSubmit");

    /* ── Poll for completion ────────────────────────────────── */
    CUfileIOEvents_t events[BATCH_SIZE];
    unsigned completed_total = 0;
    while (completed_total < BATCH_SIZE) {
        unsigned nr_done = 0;
        st = cuFileBatchIOGetStatus(batch, 1, &nr_done, events, NULL);
        if (st.err != CU_FILE_SUCCESS) {
            fprintf(stderr, "BatchGetStatus error: %s\n",
                    CUFILE_ERRSTR(st.err));
            break;
        }
        for (unsigned i = 0; i < nr_done; i++) {
            int tag = (int)(uintptr_t)events[i].cookie;
            if (events[i].status == CUFILE_COMPLETE) {
                printf("  Chunk %d: COMPLETE (%zu bytes)\n",
                       tag, events[i].ret);
            } else if (events[i].status == CUFILE_FAILED) {
                printf("  Chunk %d: FAILED\n", tag);
            }
        }
        completed_total += nr_done;
    }

    printf("Batch write completed: %u/%d chunks\n",
           completed_total, BATCH_SIZE);

    /* ── Cleanup ────────────────────────────────────────────── */
    cuFileBatchIODestroy(batch);
    cuFileHandleDeregister(fh);
    close(fd);
    cuFileBufDeregister(gpu_buf);
    cudaFree(gpu_buf);
    cuFileDriverClose();

    printf("\n=== Example Complete ===\n");
    return EXIT_SUCCESS;
}
