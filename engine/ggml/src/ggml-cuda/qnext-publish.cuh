#pragma once

#include "common.cuh"

// QNEXT_MOE_MISS_OVERLAP: GGML_OP_QNEXT_PUBLISH, see ggml.h and qnext-publish.cu
void ggml_cuda_op_qnext_publish(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_qnext_publish_supported(const ggml_tensor * op);

// proc addresses "ggml_backend_cuda_qnext_publish_alloc" / "ggml_backend_cuda_qnext_publish_free" of the CUDA registry:
// n_slots slots of GGML_QNEXT_PUBLISH_SLOT bytes (ggml.h) in mapped pinned host memory (zeroed; *host and *slots_dev are the host
// and device addresses of slot 0) and n_slots uint32 counters in device memory (zeroed), on the device of backend.
// false on any failure (nothing allocated, error cleared)
bool ggml_backend_cuda_qnext_publish_alloc(ggml_backend_t backend, size_t n_slots, void ** host, void ** slots_dev, void ** ctrs_dev);
void ggml_backend_cuda_qnext_publish_free(ggml_backend_t backend, void * host, void * ctrs_dev);
