#pragma once

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <vector>

struct ggml_moe_stream_op {
    int split;
    ggml_tensor * source;
    ggml_tensor * source_ids;
    ggml_tensor * weights;
    ggml_tensor * ids;
};

struct ggml_moe_stream {
    bool enabled;
    bool cpu;
    int64_t max_experts;
    size_t cache_limit;
    std::vector<ggml_moe_stream_op> ops;

    ggml_moe_stream();
    ~ggml_moe_stream();
    bool candidate(const ggml_tensor * node) const;
    bool fits(const ggml_tensor * node) const;
    bool prepare(const ggml_moe_stream_op & op, ggml_backend_t backend, ggml_backend_t ids_backend);

private:
    struct entry;
    std::list<std::unique_ptr<entry>> cache;
    size_t cache_size = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    uint64_t uploaded = 0;
    bool trace;
};
