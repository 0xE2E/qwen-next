#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

// one cudaMemcpyBatchAsync of host-to-device copies on the compute stream of the backend, no synchronization;
// also reachable as the proc address "ggml_backend_cuda_memcpy_batch_async" of the CUDA backend registry.
// returns false on any error or if count == 0
GGML_BACKEND_API bool ggml_backend_cuda_memcpy_batch_async(ggml_backend_t backend, void ** dsts, const void ** srcs, const size_t * sizes, size_t count);

// QNEXT_PF_OVERLAP: non-blocking test whether a CUDA backend event has completed (cudaEventQuery);
// also reachable as the proc address "ggml_backend_cuda_event_query" of the CUDA backend registry
GGML_BACKEND_API bool ggml_backend_cuda_event_query(ggml_backend_event_t event);

// QNEXT_ELASTIC_KV: elastic device buffers (CUDA virtual memory). Each granularity-sized chunk of the buffer's
// virtual range is either mapped to its own physical allocation or unmapped (owns no VRAM). An elastic buffer is a
// CUDA buffer for every other purpose (same buffer interface, same buffer type name). All are also reachable as proc
// addresses of the CUDA backend registry under the same names.
// the elastic buffer type of the device of a plain CUDA device buffer type, nullptr if not supported (no VMM)
// its buffers are fully mapped at allocation, like cudaMalloc, and have the plain CUDA tensor layout
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_elastic_buffer_type(int device);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_elastic_buffer_type_of(ggml_backend_buffer_type_t buft);
// an elastic buffer of `size` bytes with no chunk mapped (a parking range for chunks), buft must be elastic
GGML_BACKEND_API ggml_backend_buffer_t ggml_backend_cuda_elastic_buffer_reserve(ggml_backend_buffer_type_t buft, size_t size);
GGML_BACKEND_API bool   ggml_backend_cuda_buffer_is_elastic(ggml_backend_buffer_t buffer);
GGML_BACKEND_API size_t ggml_backend_cuda_elastic_granularity(ggml_backend_buffer_t buffer); // 0 if not elastic
GGML_BACKEND_API size_t ggml_backend_cuda_elastic_n_chunks(ggml_backend_buffer_t buffer);
GGML_BACKEND_API size_t ggml_backend_cuda_elastic_n_mapped(ggml_backend_buffer_t buffer);
GGML_BACKEND_API bool   ggml_backend_cuda_elastic_chunk_mapped(ggml_backend_buffer_t buffer, size_t chunk);
// move the physical memory of n mapped chunks of src to n unmapped chunks of dst (same device): per chunk unmap then map,
// then access is set on dst; zero: fill the moved chunks with 0 (synchronous). The caller guarantees that no pending GPU
// work reads or writes the src chunks. Returns false on error (a failed map puts the chunk back at its source).
// granularity of the chunks of an elastic buffer type (0 if not elastic)
GGML_BACKEND_API size_t ggml_backend_cuda_elastic_buft_granularity(ggml_backend_buffer_type_t buft);
// give the unmapped chunks their own new physical allocation and map them, zero filled (synchronous)
GGML_BACKEND_API bool   ggml_backend_cuda_elastic_map_new(ggml_backend_buffer_t buffer, const size_t * chunks, size_t n);
// wait for all work on the device of the buffer (all streams), before chunks that kernels may read are moved
GGML_BACKEND_API void   ggml_backend_cuda_elastic_sync(ggml_backend_buffer_t buffer);
GGML_BACKEND_API bool   ggml_backend_cuda_elastic_move(ggml_backend_buffer_t src, const size_t * src_chunks,
                                                       ggml_backend_buffer_t dst, const size_t * dst_chunks, size_t n, bool zero);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
