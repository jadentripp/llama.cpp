#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static bool run(ggml_backend_t target, ggml_backend_t cpu, ggml_type type, int batch) {
    constexpr int width = 256, rows = 128, experts = 16, used = 2;
    ggml_context * wc = ggml_init({ ggml_tensor_overhead(), nullptr, true });
    ggml_tensor * w = ggml_new_tensor_3d(wc, type, width, rows, experts);
    ggml_backend_buffer_t wb = ggml_backend_alloc_ctx_tensors(wc, cpu);
    ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> raw(width * rows * experts);
    for (size_t i = 0; i < raw.size(); ++i) {
        raw[i] = std::sin(float(i) * 0.17f) * 0.1f;
    }
    if (type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(w, raw.data(), 0, raw.size() * sizeof(float));
    } else {
        std::vector<uint8_t> q(ggml_nbytes(w));
        ggml_quantize_chunk(type, raw.data(), q.data(), 0, rows * experts, width, nullptr);
        ggml_backend_tensor_set(w, q.data(), 0, q.size());
    }
    ggml_backend_t backends[] = { target, cpu };
    const int count = target == cpu ? 1 : 2;
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, count, 128, false, true);
    bool ok = true;
    for (int step = 0; step < 20 && ok; ++step) {
        ggml_context * ctx = ggml_init({ 32 * ggml_tensor_overhead() + 2 * ggml_graph_overhead_custom(64, false), nullptr, true });
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, used, batch);
        ggml_tensor * full_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used + 1, batch);
        ggml_tensor * ids = ggml_view_2d(ctx, full_ids, used, batch, full_ids->nb[1], sizeof(int32_t));
        ggml_set_input(x);
        ggml_set_input(full_ids);
        ggml_tensor * y = ggml_mul_mat_id(ctx, w, x, ids);
        ggml_set_output(y);
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(graph, y);
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            ok = false;
            break;
        }
        std::vector<float> input(width * used * batch);
        std::vector<int32_t> indices((used + 1) * batch, -1);
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = std::cos(float(i + step) * 0.03f);
        }
        for (int t = 0; t < batch; ++t) {
            for (int k = 0; k < used; ++k) {
                indices[t * (used + 1) + k + 1] = (step / 2 * 3 + t + (step % 5 == 0 ? 0 : k)) % experts;
            }
        }
        ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
        ggml_backend_tensor_set(full_ids, indices.data(), 0, indices.size() * sizeof(int32_t));
        if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS) {
            ok = false;
            break;
        }
        // A reused execution graph must refill staging with the latest router IDs.
        indices[1] = (indices[1] + 7) % experts;
        ggml_backend_tensor_set(full_ids, indices.data(), 0, indices.size() * sizeof(int32_t));
        if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS) {
            ok = false;
            break;
        }
        std::vector<float> actual(rows * used * batch);
        ggml_backend_tensor_get(y, actual.data(), 0, actual.size() * sizeof(float));
        if (batch * used <= 4 && y->src[0]->ne[2] != batch * used) {
            fprintf(stderr, "expert tensor was not compacted\n");
            ok = false;
        }
        if (batch * used <= 4 && ggml_backend_sched_get_tensor_backend(sched, y) != target) {
            fprintf(stderr, "compact operation did not execute on the requested backend\n");
            ok = false;
        }
        if (batch * used > 4 && (y->src[0] != w || ggml_backend_sched_get_tensor_backend(sched, y) != cpu)) {
            fprintf(stderr, "oversized operation did not preserve the CPU fallback\n");
            ok = false;
        }

        ggml_context * ref = ggml_init({ 16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false), nullptr, true });
        ggml_tensor * rx = ggml_new_tensor_3d(ref, GGML_TYPE_F32, width, used, batch);
        ggml_tensor * ri = ggml_new_tensor_2d(ref, GGML_TYPE_I32, used, batch);
        ggml_tensor * ry = ggml_mul_mat_id(ref, w, rx, ri);
        ggml_cgraph * rg = ggml_new_graph_custom(ref, 64, false);
        ggml_build_forward_expand(rg, ry);
        ggml_backend_buffer_t rb = ggml_backend_alloc_ctx_tensors(ref, cpu);
        std::vector<int32_t> packed(used * batch);
        for (int t = 0; t < batch; ++t) {
            for (int k = 0; k < used; ++k) {
                packed[t * used + k] = indices[t * (used + 1) + k + 1];
            }
        }
        ggml_backend_tensor_set(rx, input.data(), 0, input.size() * sizeof(float));
        ggml_backend_tensor_set(ri, packed.data(), 0, packed.size() * sizeof(int32_t));
        ok = ok && ggml_backend_graph_compute(cpu, rg) == GGML_STATUS_SUCCESS;
        std::vector<float> expected(actual.size());
        ggml_backend_tensor_get(ry, expected.data(), 0, expected.size() * sizeof(float));
        double error = 0, scale = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            if (!std::isfinite(actual[i])) {
                ok = false;
            }
            error += std::pow(actual[i] - expected[i], 2);
            scale += std::pow(expected[i], 2);
        }
        if (error / std::max(scale, 1e-12) > 5e-4) {
            fprintf(stderr, "step %d NMSE=%g\n", step, error / scale);
            ok = false;
        }
        ggml_backend_buffer_free(rb);
        ggml_free(ref);
        ggml_backend_sched_reset(sched);
        ggml_free(ctx);
    }
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(wb);
    ggml_free(wc);
    fprintf(stderr, "%s batch=%d: %s\n", ggml_type_name(type), batch, ok ? "OK" : "FAIL");
    return ok;
}

int main(int argc, char ** argv) {
    env("GGML_MOE_STREAM", "1");
    env("GGML_MOE_STREAM_CPU", "1");
    env("GGML_MOE_MAX_EXPERTS", "4");
    env("GGML_MOE_CACHE_MIB", "1");
    env("GGML_MOE_TRACE", "1");
    ggml_backend_load_all();
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t target = argc > 1 ? ggml_backend_init_by_name(argv[1], nullptr) : cpu;
    if (!cpu || !target) {
        fprintf(stderr, "requested backend is unavailable\n");
        return 1;
    }
    bool ok = true;
    for (const char * cache : { "0", "1" }) {
        env("GGML_MOE_CACHE_MIB", cache);
        for (ggml_type type : { GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_Q4_1 }) {
            for (int batch : { 1, 2, 9 }) {
                ok = run(target, cpu, type, batch) && ok;
            }
        }
    }
    if (target != cpu) {
        ggml_backend_free(target);
    }
    ggml_backend_free(cpu);
    return ok ? 0 : 1;
}
