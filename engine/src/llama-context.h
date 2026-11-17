#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-moe-cache.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <array>
#include <map>
#include <vector>

struct llama_model;
class llama_batch_allocr;
class llama_moe_cache;
class llama_compute_elastic;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    float * get_embeddings_nextn();
    float * get_embeddings_nextn_ith(int32_t i);

    float * get_embeddings_layer_inp(uint32_t lid);

    ggml_tensor * get_layer_inp_dev() const { return layer_inp_dev; }
    ggml_tensor * get_h_nextn_dev()   const { return h_nextn_dev; }

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    bool get_causal_attn() const;

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_embeddings_nextn(bool value, bool masked);
    void set_embeddings_layer_inp(uint32_t lid, bool enable);
    void set_embeddings_layer_inp_dev(bool value);

    // whether decode accepts entries whose embd is a row of ctx_other's layer input device buffer
    bool supports_embd_row() const;

    // h_nextn device buffer, see llama_set_embeddings_nextn_dev()
    bool set_embeddings_nextn_dev(bool value);
    bool get_h_nextn_rows(int64_t row0, int64_t n_rows, float * out);
    bool set_h_nextn_rows(int64_t row0, int64_t n_rows, const float * data);

    // state rows read from the h_nextn device buffer of src, see llama_set_embd_state_src()
    bool set_state_src(llama_context * src);
    bool supports_state_row() const;
    void set_nextn_layer_offset(int32_t offset);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch_ext & batch_inp);
    int decode(const llama_batch_ext & batch_inp);

    // compat version
    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);

    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    // async-copy enabled layer-input tensors (per cparams.output_layer_inp)
    // from backend into host-side embd_layer_inp buffers
    bool extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens);

    // (re)allocate the device buffer for cparams.embeddings_layer_inp_dev
    void layer_inp_dev_alloc();

    // (re)allocate the device buffer for cparams.embeddings_nextn_dev, false if it does not fit
    bool h_nextn_dev_alloc();

    ggml_backend_t backend_from_dev(ggml_backend_dev_t dev) const;

    //
    // graph
    //

public:
    uint32_t graph_max_nodes() const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

private:
    llm_graph_result * get_gf_res_prev();

    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    llm_graph_cb graph_get_cb() const;

    // parameters defining the backend scheduler instance
    struct sched_params {
        size_t max_nodes  = 0;
        bool   parallel   = false;
        bool   op_offload = false;

        // a scheduler can serve any graph of up to max_nodes nodes
        bool can_reuse(const sched_params & other) const {
            return parallel   == other.parallel
                && op_offload == other.op_offload
                && max_nodes  <= other.max_nodes;
        }
    };

    // create a new backend scheduler with the given parameters, unless the current one can be reused
    void sched_new(const sched_params & params);

    // ggml_backend_sched copy callback, copies only the experts used by MUL_MAT_ID and updates the MoE cache
    static bool sched_copy_experts(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, void * user_data);

    // ggml_backend_sched realloc callback, counts and traces the graph reallocations of this context
    static void sched_graph_realloc(ggml_backend_sched_t sched, ggml_cgraph * gf, bool backend_ids_changed, bool graph_size_changed, void * user_data);

    // disable auto fused ops (Flash Attention, Gated Delta Net) whose op lands on a device
    // that differs from the layer it belongs to (usually due to missing backend support)
    void resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs);

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams cparams;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    llama_memory_ptr memory;
    llama_moe_cache_ptr moe_cache;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    // hidden state required by the nextn layers (2-dimensional array: [n_outputs][n_embd])
    // populated only when cparams.embeddings_nextn is enabled and the model graph
    // sets llm_graph_result::t_h_nextn
    buffer_view<float> embd_nextn = {nullptr, 0};

    // host buffers for output layer input embeddings, per layer
    // populated when cparams.output_layer_inp[il] is true
    std::vector<buffer_view<float>> embd_layer_inp;
    std::vector<int32_t> embd_batch_idxs; // extracted index -> original batch index

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_input_tensors = 0; // number of tensors marked as input during the last graph reserve
    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    sched_params sched_params_cur;

    bool sched_need_reserve = true;

    // whether the reserve graph has a logits tensor, set in sched_reserve()
    // true until then, because the first output_reserve() runs before any graph is reserved
    bool graph_has_logits = true;

    // state of sched_copy_experts, reset before each graph compute
    struct copy_experts_info {
        const ggml_tensor *  ids = nullptr;
        std::vector<int32_t> ids_data;
        std::vector<bool>    used;

        // per graph compute, logged by graph_compute if QNEXT_MOE_STATS is set
        int64_t n_from_cache     = 0; // experts copied device-to-device from the MoE cache
        size_t  bytes_from_cache = 0;
        int64_t n_from_host      = 0; // experts uploaded from host memory (bytes include the MMQ padding)
        size_t  bytes_from_host  = 0;
        int64_t n_tensors        = 0; // expert tensors handled by sched_copy_experts

        void reset() {
            ids = nullptr;
            ids_data.clear();
            used.clear();
            n_from_cache     = 0;
            bytes_from_cache = 0;
            n_from_host      = 0;
            bytes_from_host  = 0;
            n_tensors        = 0;
        }
    };

    copy_experts_info copy_experts;

    // graph reallocations during inference where the backend assignments did not change, split by whether the
    // graph size is the same as the previously scheduled one (unexpected) or different
    int32_t n_reallocs_same = 0;
    int32_t n_reallocs_diff = 0;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // QNEXT_PF_OVERLAP=1 (default off), see llama-context.cpp: large-batch expert tensors are uploaded whole, ahead of
    // their use, into a ring of device slots on a second stream; declared after backends so it is destroyed before them
    struct pf_overlap_state;
    struct pf_overlap_deleter { void operator()(pf_overlap_state * p) const; };
    std::unique_ptr<pf_overlap_state, pf_overlap_deleter> pf_overlap;
    uint32_t pf_overlap_n_tokens = 0; // ubatch size of the graph process_ubatch is computing, 0 for other graphs

    void pf_overlap_init();
    void pf_overlap_graph_begin(ggml_cgraph * gf);
    void pf_overlap_graph_end();
    std::map<ggml_backend_buffer_type_t, size_t> pf_overlap_memory_breakdown() const;

    // QNEXT_STEP_EVENTS=N (default off, measurement only), see llama-context.cpp: CUDA timing events at the stream
    // level around the split graph launches and the MoE cache miss uploads of small-batch evaluations; logged as
    // qnext_step_events lines. Declared after backends so it is destroyed before them
    struct step_events_state;
    struct step_events_deleter { void operator()(step_events_state * p) const; };
    std::unique_ptr<step_events_state, step_events_deleter> step_events;

    void step_events_init();
    void step_events_graph_begin();
    void step_events_graph_end();
    static void step_events_copy_mark(void * ud, ggml_backend_t backend, int which);

    // QNEXT_ELASTIC_COMPUTE=1 (default off), see llama-elastic.h: the target context's compute buffer on elastic
    // memory, its top lent to the MoE cache while only small graphs run. Declared after sched and moe_cache so it is
    // destroyed before them (its destructor only logs)
    struct el_cmp_deleter { void operator()(llama_compute_elastic * p) const; };
    std::unique_ptr<llama_compute_elastic, el_cmp_deleter> el_cmp;
    ggml_backend_t             el_cmp_backend = nullptr; // the GPU backend whose compute buffer is lent
    size_t                     el_cmp_idx     = 0;       // its index in backend_ptrs / backend_buft
    ggml_backend_buffer_type_t el_cmp_buft    = nullptr; // its elastic buffer type, given to the scheduler
    int64_t                    el_cmp_ntok    = 0;       // tokens of the ubatch process_ubatch is allocating, 0 otherwise
    size_t                     el_cmp_logged_keep = 0;   // the keep region of the last init line (logged again when it changes)
    uint64_t                   el_cmp_n_reserves  = 0;   // sched_reserve measurements so far

    bool     mtp_crop_offload_logged = false; // QNEXT_MTP_CROP: one-time error line for an uncropped offloaded draft graph

    // QNEXT_COMPUTE_FILL=<byte> (debug, default off), see llama-context.cpp: fill the scheduler's compute buffers
    int      cmp_fill       = -2; // -2: not read yet, -1: off, else the byte
    bool     cmp_fill_every = false;
    bool     cmp_fill_warned = false;
    uint64_t cmp_fill_n     = 0;
    std::vector<ggml_backend_buffer_t> cmp_fill_last; // [backend] the buffer filled last
    void cmp_fill_init();
    void cmp_fill_apply(ggml_backend_sched_t s);

    void el_cmp_init();
    void el_cmp_give();
    void el_cmp_take(const char * why);
    static void el_cmp_alloc_cb(ggml_backend_sched_t sched, void * user_data);
    static void el_cmp_buffer_free_cb(ggml_backend_sched_t sched, ggml_backend_t backend, ggml_backend_buffer_t chunk0, int n_chunks, void * user_data);

    // QNEXT_STEP_EVENTS / QNEXT_IDS_ASYNC: the scheduler's split callback, set only if one of them is on
    static void sched_split_cb(enum ggml_backend_sched_split_phase phase, const ggml_backend_sched_split_info * info, void * user_data);

    // QNEXT_MTP_PRUNE=1 (default off): draft (MTP) batches with no outputs are pruned to their state writes, see
    // llm_graph_context::qnext_prune_to_state_writes; counted per evaluated ubatch, logged as qnext_mtp_prune lines
    uint64_t mtp_prune_n_pruned = 0;
    uint64_t mtp_prune_n_full   = 0;
    uint64_t mtp_prune_n_logged = 0;
    std::map<std::string, uint64_t> mtp_prune_reasons; // full ubatches per reason
    // QNEXT_MTP_STATE_HASH=N (debug): 64-bit hash of the draft's K, V and indexer rows of the used cells (syncs)
    int64_t mtp_hash_every = -1; // -1: not read yet, 0: off
    uint64_t mtp_hash_ubatches = 0;
    bool mtp_hash_prev_outputs = true;
    void qnext_mtp_state_hash(const char * when, uint32_t n_tokens, uint32_t n_outputs_ub);
    std::string mtp_prune_reasons_str() const {
        std::string s;
        for (const auto & [why, n] : mtp_prune_reasons) {
            s += " | " + why + ": " + std::to_string(n);
        }
        return s;
    }

    // QNEXT_MTP_CROP=1 (default off): draft (MTP) graphs keep only their output rows before the FFN/MoE, see graph_mtp
    // (qwen4exp.cpp); counted per evaluated ubatch, logged as qnext_mtp_crop lines; the compute buffer sizes after each
    // newly built draft graph are compared with the largest so far (starting from the reservation) to log any growth
    uint64_t mtp_crop_n_cropped = 0;
    uint64_t mtp_crop_n_full    = 0;
    uint64_t mtp_crop_n_logged  = 0;
    std::map<std::string, uint64_t> mtp_crop_reasons; // not-cropped ubatches per reason
    std::vector<size_t> mtp_crop_buf_max;              // [backend] largest compute buffer size seen
    std::string mtp_crop_reasons_str() const {
        std::string s;
        for (const auto & [why, n] : mtp_crop_reasons) {
            s += " | " + why + ": " + std::to_string(n);
        }
        return s;
    }

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    // Separate arenas give batches with and without outputs distinct CUDA graph cache keys.
    std::array<llm_graph_result_ptr, 2> gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    llm_graph_result * gf_res_prev_active = nullptr;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // device buffer with enabled layer inputs, [n_layer_inp*n_embd, n_batch], row = batch index
    ggml_context_ptr        ctx_layer_inp_dev;
    ggml_backend_buffer_ptr buf_layer_inp_dev;
    ggml_tensor *           layer_inp_dev = nullptr;
    ggml_backend_dev_t      layer_inp_dev_dev = nullptr;
    ggml_backend_t          layer_inp_backend = nullptr;

    // ordering with the reader context (no host sync):
    //   written - recorded by this context after the batch, waited on by the reader
    //   read    - recorded by the reader after it consumed the rows, waited on by this context
    ggml_backend_event_ptr ev_layer_inp_written;
    ggml_backend_event_ptr ev_layer_inp_read;
    bool ev_layer_inp_read_recorded = false;

    // device buffer with the h_nextn rows, F32 [n_embd_out, n_batch + n_seq_max]:
    // row r < n_batch is the token at batch index r of the last decode, row n_batch + s is set by the host for seq s
    // ordered with the reader context like the layer input buffer
    ggml_context_ptr        ctx_h_nextn_dev;
    ggml_backend_buffer_ptr buf_h_nextn_dev;
    ggml_tensor *           h_nextn_dev = nullptr;
    ggml_backend_dev_t      h_nextn_dev_dev = nullptr;
    ggml_backend_t          h_nextn_backend = nullptr;
    ggml_backend_event_ptr  ev_h_nextn_written;
    ggml_backend_event_ptr  ev_h_nextn_read;
    bool ev_h_nextn_written_recorded = false;
    bool ev_h_nextn_read_recorded    = false;

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // true after the prism.hadamard coverage check passes on a reserved graph
    bool hadamard_verified = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
