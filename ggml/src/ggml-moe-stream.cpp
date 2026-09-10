#include "ggml-moe-stream.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

static size_t env_size(const char * name, size_t fallback, size_t maximum) {
    const char * value = getenv(name);
    if (!value) {
        return fallback;
    }
    char * end = nullptr;
    errno = 0;
    const unsigned long long n = strtoull(value, &end, 10);
    if (!*value || *value == '-' || *end || errno || n > maximum) {
        GGML_ABORT("invalid %s: expected an integer in [0, %zu]", name, maximum);
    }
    return (size_t) n;
}

struct ggml_moe_stream::entry {
    const void * source;
    int32_t expert;
    ggml_backend_t backend;
    ggml_context * ctx = nullptr;
    ggml_tensor * tensor = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    size_t size = 0;

    ~entry() {
        ggml_backend_synchronize(backend);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
};

ggml_moe_stream::ggml_moe_stream() :
    enabled(env_size("GGML_MOE_STREAM", 0, 1) != 0),
    cpu(env_size("GGML_MOE_STREAM_CPU", 0, 1) != 0),
    max_experts(env_size("GGML_MOE_MAX_EXPERTS", 32, 65536)),
    cache_limit(env_size("GGML_MOE_CACHE_MIB", 256, SIZE_MAX / (1024 * 1024)) * 1024 * 1024),
    trace(env_size("GGML_MOE_TRACE", 0, 1) != 0) {
    if (enabled && max_experts == 0) {
        GGML_ABORT("GGML_MOE_MAX_EXPERTS must be positive");
    }
}

ggml_moe_stream::~ggml_moe_stream() {
    if (enabled && trace) {
        GGML_LOG_INFO("moe-stream: cache hits=%llu misses=%llu evictions=%llu upload=%.2f MiB cache=%.2f MiB\n",
            (unsigned long long) hits, (unsigned long long) misses, (unsigned long long) evictions,
            uploaded / 1048576.0, cache_size / 1048576.0);
    }
}

bool ggml_moe_stream::candidate(const ggml_tensor * node) const {
    if (!enabled || node->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }
    const ggml_tensor * w = node->src[0];
    return node->src[2] && node->src[2]->type == GGML_TYPE_I32 &&
        node->src[2]->ne[2] == 1 && node->src[2]->ne[3] == 1 && w && w->buffer && ggml_backend_buffer_is_host(w->buffer) &&
        ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_is_contiguous(w) && w->ne[2] > 1 && w->ne[3] == 1;
}

bool ggml_moe_stream::fits(const ggml_tensor * node) const {
    const ggml_tensor * ids = node->src[2];
    return ids->ne[0] > 0 && ids->ne[1] > 0 &&
        ids->ne[1] <= max_experts / ids->ne[0];
}

bool ggml_moe_stream::prepare(const ggml_moe_stream_op & op, ggml_backend_t backend, ggml_backend_t ids_backend) {
    // Prior users can share the allocator's staging region.
    ggml_backend_synchronize(backend);
    ggml_backend_synchronize(ids_backend);

    std::vector<uint8_t> raw(ggml_nbytes(op.source_ids));
    ggml_backend_tensor_get(op.source_ids, raw.data(), 0, raw.size());
    std::vector<int32_t> ids;
    for (int64_t t = 0; t < op.ids->ne[1]; ++t) {
        for (int64_t k = 0; k < op.ids->ne[0]; ++k) {
            int32_t id;
            memcpy(&id, raw.data() + t * op.source_ids->nb[1] + k * op.source_ids->nb[0], sizeof(id));
            if (id < 0 || id >= op.source->ne[2]) {
                GGML_LOG_ERROR("moe-stream: expert index out of range: %d\n", id);
                return false;
            }
            ids.push_back(id);
        }
    }
    std::vector<int32_t> unique = ids;
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    if (unique.size() > (size_t) op.weights->ne[2]) {
        return false;
    }

    const size_t bytes = op.source->nb[2];
    for (size_t slot = 0; slot < unique.size(); ++slot) {
        ggml_tensor view = *op.weights;
        view.ne[2] = 1;
        view.nb[3] = bytes;
        view.data = (char *) op.weights->data + slot * bytes;
        view.view_src = op.weights;
        view.view_offs = slot * bytes;

        auto found = std::find_if(cache.begin(), cache.end(), [&](const std::unique_ptr<entry> & e) {
            return e->source == op.source->data && e->expert == unique[slot] && e->backend == backend &&
                e->tensor->type == view.type && e->tensor->ne[0] == view.ne[0] && e->tensor->ne[1] == view.ne[1];
        });
        if (found != cache.end()) {
            ggml_backend_tensor_copy((*found)->tensor, &view);
            cache.splice(cache.begin(), cache, found);
            ++hits;
            continue;
        }

        ++misses;
        const void * data = (const char *) op.source->data + (size_t) unique[slot] * bytes;
        ggml_backend_tensor_set(&view, data, 0, bytes);
        uploaded += bytes;
        if (bytes > cache_limit || cache_limit == 0) {
            continue;
        }

        auto e = std::unique_ptr<entry>(new entry);
        e->source = op.source->data;
        e->expert = unique[slot];
        e->backend = backend;
        e->ctx = ggml_init({ ggml_tensor_overhead(), nullptr, true });
        if (!e->ctx) {
            return false;
        }
        e->tensor = ggml_dup_tensor(e->ctx, &view);
        const auto buft = ggml_backend_get_default_buffer_type(backend);
        const size_t alignment = ggml_backend_buft_get_alignment(buft);
        size_t expected = ggml_backend_buft_get_alloc_size(buft, e->tensor);
        expected = GGML_PAD(expected, alignment);
        if (expected > cache_limit) {
            continue;
        }
        while (!cache.empty() && expected > cache_limit - cache_size) {
            cache_size -= cache.back()->size;
            cache.pop_back();
            ++evictions;
        }
        e->buffer = ggml_backend_alloc_ctx_tensors(e->ctx, backend);
        if (!e->buffer) {
            // Caching is optional; the active expert has already been loaded.
            continue;
        }
        e->size = ggml_backend_buffer_get_size(e->buffer);
        if (e->size > cache_limit) {
            continue;
        }
        while (!cache.empty() && e->size > cache_limit - cache_size) {
            cache_size -= cache.back()->size;
            cache.pop_back();
            ++evictions;
        }
        ggml_backend_tensor_copy(&view, e->tensor);
        cache_size += e->size;
        cache.push_front(std::move(e));
    }
    for (int32_t & id : ids) {
        id = std::lower_bound(unique.begin(), unique.end(), id) - unique.begin();
    }
    ggml_backend_tensor_set(op.ids, ids.data(), 0, ids.size() * sizeof(int32_t));
    return true;
}
