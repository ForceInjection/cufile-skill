/*
 * 05-end-to-end-pipeline.cu — Double-Buffered Prefetch Pipeline
 *
 * Demonstrates:
 *   1. Double-buffered async read: prefetch next chunk while processing current
 *   2. CUDA event-based synchronization between I/O and compute streams
 *   3. Zero-copy GPU data processing pipeline (NVMe → GPU → Kernel → GPU)
 *   4. Throughput measurement for the complete pipeline
 *
 * This is the most important production pattern for AI/ML data pipelines.
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

#define BUF_SIZE    (16 * 1024 * 1024)   // 16 MB per buffer
#define NUM_CHUNKS  16                    // Total chunks to process
#define TOTAL_SIZE  (BUF_SIZE * NUM_CHUNKS)

/* ── Simulated processing kernel ───────────────────────────────── */

/* A meaningful compute kernel: normalize + threshold + statistics */
__global__ void process_chunk(const float *input, float *output,
                              size_t num_elements, float threshold) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float val = input[idx];
        /* Simulate a realistic preprocessing pipeline */
        val = val * 2.0f - 1.0f;       // Normalize to [-1, 1]
        val = val > threshold ? val : 0.0f; // Threshold
        output[idx] = val;
    }
}

/* ── Initialization kernel ─────────────────────────────────────── */

__global__ void init_data(float *buf, size_t num_elements, int chunk_id) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        buf[idx] = (float)((idx + chunk_id * 1000) % 10000) / 10000.0f;
    }
}

/* ── Double-Buffered Pipeline ──────────────────────────────────── */

typedef struct {
    CUfileDescr_t input_fh;
    CUfileDescr_t output_fh;
    CUdeviceptr buf[2];        /* [0] = current, [1] = next */
    CUdeviceptr result[2];     /* Processed results */
    size_t buf_size;
    cudaStream_t io_stream;
    cudaStream_t compute_stream;
    cudaEvent_t io_done[2];
    cudaEvent_t compute_done[2];
    int initialized;
} DoubleBufferPipeline;

int pipeline_init(DoubleBufferPipeline *p,
                  const char *input_path, const char *output_path) {
    memset(p, 0, sizeof(*p));
    p->buf_size = BUF_SIZE;

    /* Allocate double buffers */
    size_t num_floats = BUF_SIZE / sizeof(float);
    for (int i = 0; i < 2; i++) {
        cuMemAlloc(&p->buf[i], BUF_SIZE);
        cuMemAlloc(&p->result[i], BUF_SIZE);
        cuFileBufRegister(p->buf[i], BUF_SIZE, 0);
        cuFileBufRegister(p->result[i], BUF_SIZE, 0);
        cudaEventCreate(&p->io_done[i]);
        cudaEventCreate(&p->compute_done[i]);
    }

    /* Open input file */
    int ifd = open(input_path, O_DIRECT | O_RDONLY);
    if (ifd < 0) { perror("open input"); return -1; }
    p->input_fh.cookie = (CUfileDriverCookie)(uintptr_t)ifd;
    cuFileHandleRegister(&p->input_fh, NULL);

    /* Open output file */
    int ofd = open(output_path, O_DIRECT | O_WRONLY | O_CREAT, 0644);
    if (ofd < 0) { perror("open output"); return -1; }
    ftruncate(ofd, TOTAL_SIZE);
    p->output_fh.cookie = (CUfileDriverCookie)(uintptr_t)ofd;
    cuFileHandleRegister(&p->output_fh, NULL);

    /* Create streams */
    cudaStreamCreate(&p->io_stream);
    cudaStreamCreate(&p->compute_stream);

    /* Register I/O stream with both handles */
    cuFileStreamRegister(p->input_fh, p->io_stream);
    cuFileStreamRegister(p->output_fh, p->io_stream);

    p->initialized = 1;
    return 0;
}

void pipeline_run(DoubleBufferPipeline *p) {
    int threads = 256;
    size_t num_floats = BUF_SIZE / sizeof(float);
    int blocks = (num_floats + threads - 1) / threads;
    float threshold = 0.5f;

    printf("Starting double-buffered pipeline (%d chunks)...\n", NUM_CHUNKS);
    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    /* Prefetch first chunk into buf[0] */
    cuFileReadAsync(p->input_fh, p->buf[0], p->buf_size, 0,
                    0, p->io_stream);
    cudaEventRecord(p->io_done[0], p->io_stream);

    for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
        int cur = chunk % 2;
        int next = (chunk + 1) % 2;

        /* ── Wait for current chunk read to complete ──── */
        cudaStreamWaitEvent(p->compute_stream, p->io_done[cur], 0);

        /* ── Process current chunk ────────────────────── */
        process_chunk<<<blocks, threads, 0, p->compute_stream>>>(
            (const float *)p->buf[cur],
            (float *)p->result[cur],
            num_floats, threshold);
        cudaEventRecord(p->compute_done[cur], p->compute_stream);

        /* ── Prefetch next chunk (while compute runs) ─── */
        if (chunk + 1 < NUM_CHUNKS) {
            /* Make sure next buffer isn't still in use */
            if (chunk >= 1) {
                cudaStreamWaitEvent(p->io_stream,
                                    p->compute_done[next], 0);
            }
            cuFileReadAsync(p->input_fh, p->buf[next], p->buf_size,
                            (chunk + 1) * p->buf_size, 0, p->io_stream);
            cudaEventRecord(p->io_done[next], p->io_stream);
        }

        /* ── Write results (after compute completes) ──── */
        cudaStreamWaitEvent(p->io_stream, p->compute_done[cur], 0);
        cuFileWriteAsync(p->output_fh, p->result[cur], p->buf_size,
                         chunk * p->buf_size, 0, p->io_stream);
    }

    /* Wait for final writes to complete */
    cudaStreamSynchronize(p->io_stream);
    cudaStreamSynchronize(p->compute_stream);

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed = (t_end.tv_sec - t_start.tv_sec) +
                     (t_end.tv_nsec - t_start.tv_nsec) / 1e9;
    double bw = measure_bandwidth(NUM_CHUNKS * BUF_SIZE * 2, elapsed);
    /* ×2 because we measure both read AND write throughput */

    printf("Pipeline complete:\n");
    printf("  Chunks processed:  %d\n", NUM_CHUNKS);
    printf("  Data moved:        %s read + %s written\n",
           format_size(TOTAL_SIZE), format_size(TOTAL_SIZE));
    printf("  Time:              %.3f seconds\n", elapsed);
    printf("  Effective BW:      %.2f GB/s\n", bw);
    printf("  Per-chunk latency: %.2f ms\n",
           (elapsed / NUM_CHUNKS) * 1000.0);
}

void pipeline_destroy(DoubleBufferPipeline *p) {
    if (!p->initialized) return;

    cudaStreamDestroy(p->io_stream);
    cudaStreamDestroy(p->compute_stream);

    for (int i = 0; i < 2; i++) {
        cudaEventDestroy(p->io_done[i]);
        cudaEventDestroy(p->compute_done[i]);
        cuFileBufDeregister(p->buf[i]);
        cuFileBufDeregister(p->result[i]);
        cuMemFree(p->buf[i]);
        cuMemFree(p->result[i]);
    }

    cuFileHandleDeregister(p->input_fh);
    close((int)(uintptr_t)p->input_fh.cookie);

    cuFileHandleDeregister(p->output_fh);
    close((int)(uintptr_t)p->output_fh.cookie);
}

/* ── Main ─────────────────────────────────────────────────────── */

int main(int argc, char *argv[]) {
    const char *input_path = (argc > 1) ? argv[1] : "/mnt/nvme/input.bin";
    char output_path[512];
    snprintf(output_path, sizeof(output_path), "%s.output", input_path);

    printf("=== End-to-End Pipeline Example ===\n");
    printf("Input:  %s\n", input_path);
    printf("Output: %s\n", output_path);
    printf("Buffer: %s × 2 (double-buffered)\n\n",
           format_size(BUF_SIZE));

    /* ── Initialize cuFile ────────────────────────────────── */
    cuFileDriverOpen();
    check_gds_available();

    /* ── Prepare input file with test data ────────────────── */
    printf("Preparing input file...\n");
    {
        int fd = open(input_path, O_DIRECT | O_RDWR | O_CREAT, 0644);
        ftruncate(fd, TOTAL_SIZE);

        CUfileDescr_t tmp_fh = {0};
        tmp_fh.cookie = (CUfileDriverCookie)(uintptr_t)fd;
        cuFileHandleRegister(&tmp_fh, NULL);

        CUdeviceptr tmp_buf;
        cuMemAlloc(&tmp_buf, BUF_SIZE);
        cuFileBufRegister(tmp_buf, BUF_SIZE, 0);

        int threads = 256;
        size_t num_floats = BUF_SIZE / sizeof(float);
        int blocks = (num_floats + threads - 1) / threads;

        for (int i = 0; i < NUM_CHUNKS; i++) {
            init_data<<<blocks, threads>>>((float *)tmp_buf,
                                           num_floats, i);
            cudaDeviceSynchronize();
            cuFileWrite(tmp_fh, tmp_buf, BUF_SIZE, i * BUF_SIZE, 0);
        }

        cuFileBufDeregister(tmp_buf);
        cuMemFree(tmp_buf);
        cuFileHandleDeregister(tmp_fh);
        close(fd);
        printf("  Wrote %s test data.\n\n", format_size(TOTAL_SIZE));
    }

    /* ── Run double-buffered pipeline ─────────────────────── */
    DoubleBufferPipeline pipeline;
    if (pipeline_init(&pipeline, input_path, output_path) != 0) {
        fprintf(stderr, "Pipeline init failed.\n");
        cuFileDriverClose();
        return EXIT_FAILURE;
    }

    pipeline_run(&pipeline);
    pipeline_destroy(&pipeline);

    /* ── Cleanup ──────────────────────────────────────────── */
    cuFileDriverClose();
    printf("\n=== Example Complete ===\n");
    return EXIT_SUCCESS;
}
