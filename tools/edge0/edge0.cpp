#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// Deliberately owns one sequence. Routing state belongs to this runner, never
// to a shared model or server slot. GGUF identity binds all three release files.
struct selection {
    std::vector<int32_t> ids;
    std::vector<float> scores;
};

static selection select_experts(const std::vector<float> & logits, int k, bool ling) {
    const int n = logits.size();
    std::vector<float> scores(n);
    const float maximum = *std::max_element(logits.begin(), logits.end());
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(logits[i])) {
            throw std::runtime_error("non-finite prerouter output");
        }
        scores[i] = ling ? 1.0f / (1.0f + std::exp(-logits[i])) : std::exp(logits[i] - maximum);
    }
    std::vector<int> candidates(n);
    std::iota(candidates.begin(), candidates.end(), 0);
    auto better = [&](int a, int b) { return scores[a] == scores[b] ? a < b : scores[a] > scores[b]; };
    if (ling) {
        if (n != 128) {
            throw std::runtime_error("Ling release requires 128 experts");
        }
        std::vector<int> groups(8);
        std::iota(groups.begin(), groups.end(), 0);
        float group_scores[8];
        for (int g = 0; g < 8; ++g) {
            auto start = candidates.begin() + g * 16;
            std::partial_sort(start, start + 2, start + 16, better);
            group_scores[g] = scores[start[0]] + scores[start[1]];
        }
        std::partial_sort(groups.begin(), groups.begin() + 4, groups.end(), [&](int a, int b) {
            return group_scores[a] == group_scores[b] ? a < b : group_scores[a] > group_scores[b];
        });
        std::vector<int> kept;
        for (int g = 0; g < 4; ++g) {
            for (int e = 0; e < 16; ++e) {
                kept.push_back(groups[g] * 16 + e);
            }
        }
        candidates = std::move(kept);
    }
    std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(), better);
    selection out;
    float sum = 0;
    for (int i = 0; i < k; ++i) {
        out.ids.push_back(candidates[i]);
        out.scores.push_back(scores[candidates[i]]);
        sum += out.scores.back();
    }
    if (!(sum > 0)) {
        throw std::runtime_error("zero prerouter probability mass");
    }
    // The graph applies Ling's routed_scaling_factor after this normalization.
    for (auto & score : out.scores) {
        score /= sum;
    }
    return out;
}

static std::string field(const gguf_context * meta, const char * key) {
    const int64_t i = gguf_find_key(meta, key);
    if (i < 0 || gguf_get_kv_type(meta, i) != GGUF_TYPE_STRING) {
        throw std::runtime_error(std::string("missing GGUF string: ") + key);
    }
    return gguf_get_val_str(meta, i);
}

static uint32_t number(const gguf_context * meta, const char * key) {
    const int64_t i = gguf_find_key(meta, key);
    if (i < 0 || gguf_get_kv_type(meta, i) != GGUF_TYPE_UINT32) {
        throw std::runtime_error(std::string("missing GGUF integer: ") + key);
    }
    return gguf_get_val_u32(meta, i);
}

struct router_state {
    std::vector<std::vector<float>> hidden, current, previous;
    std::vector<selection> predictions;
};

class router {
    using gguf_ptr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
    using ggml_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
    ggml_ptr weights { nullptr, ggml_free };
    gguf_ptr metadata { nullptr, gguf_free };
    backend_ptr backend { nullptr, ggml_backend_free };
    int dim, experts, layers, top_k;
    bool ling;
    std::vector<int> owners;
    bool decode = false;
    int batch_tokens = 0;
    std::string error;

    static std::vector<float> floats(ggml_tensor * t, int64_t column) {
        if (t->type != GGML_TYPE_F32 || t->nb[0] != sizeof(float)) {
            throw std::runtime_error("expected contiguous F32 feature rows");
        }
        std::vector<float> result(t->ne[0]);
        ggml_backend_tensor_get(t, result.data(), column * t->nb[1], result.size() * sizeof(float));
        return result;
    }

    std::vector<float> onehot(ggml_tensor * t, int64_t column) const {
        if (t->type != GGML_TYPE_I32 || t->nb[0] != sizeof(int32_t)) {
            throw std::runtime_error("expected I32 expert IDs");
        }
        std::vector<int32_t> ids(t->ne[0]);
        ggml_backend_tensor_get(t, ids.data(), column * t->nb[1], ids.size() * sizeof(int32_t));
        std::vector<float> result(experts, 0.0f);
        for (int id : ids) {
            if (id < 0 || id >= experts) {
                throw std::runtime_error("invalid executed expert ID");
            }
            result[id] += 1.0f;
        }
        return result;
    }

    bool observe(ggml_tensor * t, bool ask) {
        const std::string name(t->name);
        const auto dash = name.rfind('-');
        if (dash == std::string::npos) {
            return ask ? false : true;
        }
        const std::string kind = name.substr(0, dash);
        const bool feature = kind == (ling ? "ffn_norm" : "attn_post_norm");
        const bool ids = kind == "ffn_moe_topk";
        const bool scores = kind == "ffn_moe_weights_norm";
        if (!feature && !ids && !scores) {
            return ask ? false : true;
        }
        const int layer = std::stoi(name.substr(dash + 1));
        if (layer < 0 || layer >= layers || (!ling && batch_tokens != 1)) {
            return ask ? false : true;
        }
        const bool owner = std::find(owners.begin(), owners.end(), layer) != owners.end();
        // The pinned Qwen installer patches owner blocks 6..38 only. Layer 39
        // retains its native router even though head 38 produces a prediction.
        const bool replace = decode && !state.predictions[layer].ids.empty() && (ling || layer < layers - 1);
        const bool need = (feature && owner) || (ids && (owner || replace)) || (scores && replace);
        if (ask) {
            return need;
        }
        if (feature && owner) {
            state.hidden[layer] = floats(t, t->ne[1] - 1);
        } else if (ids) {
            if (replace) {
                if (t->ne[0] != top_k || t->ne[1] != 1) {
                    throw std::runtime_error("unexpected decode expert-ID shape");
                }
                ggml_backend_tensor_set(t, state.predictions[layer].ids.data(), 0, top_k * sizeof(int32_t));
            }
            if (owner) {
                state.current[layer] = onehot(t, t->ne[1] - 1);
                if (!decode && ling) {
                    state.previous[layer] = t->ne[1] > 1 ? onehot(t, t->ne[1] - 2) : std::vector<float>(experts, 0.0f);
                }
            }
        } else if (scores && replace) {
            if (ggml_nelements(t) != top_k || t->type != GGML_TYPE_F32) {
                throw std::runtime_error("unexpected decode mixture-weight shape");
            }
            ggml_backend_tensor_set(t, state.predictions[layer].scores.data(), 0, top_k * sizeof(float));
        }
        return true;
    }

    std::vector<float> head(int owner) {
        // The CPU owns this separate graph; it never recursively executes a
        // scheduler graph. Cast boundaries retain the release's fp16 head math.
        ggml_ptr ctx(ggml_init({ 2 * 1024 * 1024, nullptr, false }), ggml_free);
        if (!ctx) {
            throw std::runtime_error("cannot allocate head graph");
        }
        const int width = dim + 2 * experts;
        ggml_tensor * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, 1);
        auto * data = static_cast<float *>(x->data);
        std::copy(state.hidden[owner].begin(), state.hidden[owner].end(), data);
        std::copy(state.current[owner].begin(), state.current[owner].end(), data + dim);
        std::fill(data + dim + experts, data + width, 0.0f);
        if (!state.previous[owner].empty()) {
            std::copy(state.previous[owner].begin(), state.previous[owner].end(), data + dim + experts);
        }
        auto half = [&](ggml_tensor * value) {
            return ggml_cast(ctx.get(), ggml_cast(ctx.get(), value, GGML_TYPE_F16), GGML_TYPE_F32);
        };
        auto weight = [&](const char * part) {
            return ggml_get_tensor(weights.get(), ("edge0." + std::to_string(owner) + "." + part).c_str());
        };
        x = half(x);
        ggml_tensor * hidden = half(ggml_mul_mat(ctx.get(), weight("fc1"), x));
        hidden = half(ggml_gelu_erf(ctx.get(), hidden));
        ggml_tensor * output = half(ggml_add(ctx.get(),
            half(ggml_mul_mat(ctx.get(), weight("fc2"), hidden)),
            half(ggml_mul_mat(ctx.get(), weight("linear_init"), x))));
        ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 64, false);
        ggml_build_forward_expand(graph, output);
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("prerouter head computation failed");
        }
        const float * result = static_cast<const float *>(output->data);
        return { result, result + experts };
    }

public:
    router_state state;
    std::string bundle;

    router(const std::string & path, int threads) {
        ggml_context * ctx = nullptr;
        metadata.reset(gguf_init_from_file(path.c_str(), { false, &ctx }));
        weights.reset(ctx);
        if (!metadata || !weights || number(metadata.get(), "edge0.format_version") != 1) {
            throw std::runtime_error("invalid Edge0 prerouter GGUF");
        }
        bundle = field(metadata.get(), "edge0.bundle_id");
        const std::string family = field(metadata.get(), "edge0.family");
        ling = family == "ling3";
        if (!ling && family != "qwen35") {
            throw std::runtime_error("unsupported Edge0 family");
        }
        dim = number(metadata.get(), "edge0.hidden_size");
        experts = number(metadata.get(), "edge0.expert_count");
        layers = number(metadata.get(), "edge0.layer_count");
        top_k = number(metadata.get(), "edge0.top_k");
        if (dim != (ling ? 1536 : 2048) || experts != (ling ? 128 : 256) ||
                layers != (ling ? 24 : 40) || top_k != (ling ? 8 : 4)) {
            throw std::runtime_error("prerouter dimensions do not match the release profile");
        }
        for (int i = ling ? 7 : 6; i < layers - 1; ++i) {
            owners.push_back(i);
            for (const char * part : { "fc1", "fc2", "linear_init" }) {
                const auto * w = ggml_get_tensor(weights.get(), ("edge0." + std::to_string(i) + "." + part).c_str());
                const bool fc2 = strcmp(part, "fc2") == 0;
                const int rows = strcmp(part, "fc1") == 0 ? 512 : experts;
                if (!w || w->type != GGML_TYPE_F16 || w->ne[0] != (fc2 ? 512 : dim + 2 * experts) ||
                        w->ne[1] != rows || w->ne[2] != 1 || w->ne[3] != 1) {
                    throw std::runtime_error("invalid head tensor layout");
                }
            }
        }
        backend.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
        if (!backend) {
            throw std::runtime_error("CPU backend required for learned heads");
        }
        auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        auto set_threads = reinterpret_cast<void (*)(ggml_backend_t, int)>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
        if (set_threads) {
            set_threads(backend.get(), threads);
        }
        reset();
    }

    void reset() {
        state = { std::vector<std::vector<float>>(layers), std::vector<std::vector<float>>(layers),
                  std::vector<std::vector<float>>(layers), std::vector<selection>(layers) };
        error.clear();
    }

    void begin(bool is_decode, int tokens) {
        decode = is_decode;
        batch_tokens = tokens;
        for (auto & row : state.hidden) {
            row.clear();
        }
        for (auto & row : state.current) {
            row.clear();
        }
    }

    void finish(bool stage) {
        if (!error.empty()) {
            throw std::runtime_error(error);
        }
        if (!stage) {
            return;
        }
        for (int owner : owners) {
            if (state.hidden[owner].empty()) {
                continue;
            }
            if (state.hidden[owner].size() != size_t(dim) || state.current[owner].size() != size_t(experts)) {
                throw std::runtime_error("missing prerouter features");
            }
            state.predictions[owner + 1] = select_experts(head(owner), top_k, ling);
            state.previous[owner] = state.current[owner];
        }
    }

    static bool callback(ggml_tensor * t, bool ask, void * user) {
        auto & self = *static_cast<router *>(user);
        try {
            return self.observe(t, ask);
        } catch (const std::exception & e) {
            self.error = e.what();
            return false;
        }
    }
};

static int run(int argc, char ** argv) {
    std::string directory, prompt = "The capital of France is", logits_path;
    int tokens = 32, ngl = 0, threads = 4, context = 2048, batch_size = 128;
    bool replay = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printf("llama-edge0 --bundle DIR [--prompt TEXT | --prompt-file FILE]\n"
                   "            [--tokens 32] [--gpu-layers 0] [--threads 4]\n"
                   "            [--context 2048] [--batch 128] [--verify-replay]\n"
                   "            [--logits-file FILE] (raw native-endian F32 rows)\n"
                   "Experimental matched Edge0 base + Recover-LoRA + cross-token heads.\n"
                   "One sequence; greedy completion of a raw prompt. Streaming is selected\n"
                   "with GGML_MOE_STREAM=1. See docs/backend/EDGE0-PORT.md.\n");
            return 0;
        }
        if (arg == "--verify-replay") {
            replay = true;
            continue;
        }
        if (i + 1 >= argc) {
            throw std::runtime_error("missing value for " + arg);
        }
        const std::string value = argv[++i];
        if (arg == "--bundle") directory = value;
        else if (arg == "--prompt") prompt = value;
        else if (arg == "--prompt-file") {
            std::ifstream file(value);
            if (!file) throw std::runtime_error("cannot read prompt file");
            prompt = { std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>() };
        }
        else if (arg == "--tokens") tokens = std::stoi(value);
        else if (arg == "--gpu-layers") ngl = std::stoi(value);
        else if (arg == "--threads") threads = std::stoi(value);
        else if (arg == "--context") context = std::stoi(value);
        else if (arg == "--batch") batch_size = std::stoi(value);
        else if (arg == "--logits-file") logits_path = value;
        else throw std::runtime_error("unknown option: " + arg);
    }
    if (directory.empty() || tokens <= 0 || threads <= 0 || context <= 0 || batch_size <= 0 || ngl < 0) {
        throw std::runtime_error("--bundle and positive run parameters are required; see --help");
    }
    std::ofstream logits_file;
    if (!logits_path.empty()) {
        logits_file.open(logits_path, std::ios::binary);
        if (!logits_file) throw std::runtime_error("cannot open logits file");
    }
    ggml_backend_load_all();
    llama_backend_init();
    router routing(directory + "/prerouter.gguf", threads);
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
    model_ptr model(llama_model_load_from_file((directory + "/base.gguf").c_str(), mp), llama_model_free);
    if (!model) throw std::runtime_error("cannot load base.gguf");
    char identity[128];
    if (llama_model_meta_val_str(model.get(), "edge0.bundle_id", identity, sizeof(identity)) < 0 || routing.bundle != identity) {
        throw std::runtime_error("base/prerouter bundle mismatch; reconvert the complete release together");
    }
    using lora_ptr = std::unique_ptr<llama_adapter_lora, decltype(&llama_adapter_lora_free)>;
    lora_ptr lora(llama_adapter_lora_init(model.get(), (directory + "/lora.gguf").c_str()), llama_adapter_lora_free);
    if (!lora || llama_adapter_meta_val_str(lora.get(), "edge0.bundle_id", identity, sizeof(identity)) < 0 || routing.bundle != identity) {
        throw std::runtime_error("base/LoRA bundle mismatch; reconvert the complete release together");
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = context;
    cp.n_batch = cp.n_ubatch = batch_size;
    cp.n_threads = cp.n_threads_batch = threads;
    cp.op_offload = ngl > 0;
    cp.cb_eval = router::callback;
    cp.cb_eval_user_data = &routing;
    cp.no_perf = false;
    using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
    context_ptr ctx(llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) throw std::runtime_error("cannot initialize context");
    auto * adapter = lora.get();
    float scale = 1.0f;
    if (llama_set_adapters_lora(ctx.get(), &adapter, 1, &scale)) throw std::runtime_error("cannot apply Recover-LoRA");
    const auto * vocab = llama_model_get_vocab(model.get());
    const int count = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    if (count <= 0 || int64_t(count) + tokens > context) throw std::runtime_error("prompt plus generation exceeds context");
    std::vector<llama_token> input(count);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), input.data(), count, true, true) != count) {
        throw std::runtime_error("tokenization failed");
    }
    for (int offset = 0; offset < count; offset += batch_size) {
        const int n = std::min(batch_size, count - offset);
        routing.begin(false, n);
        auto batch = llama_batch_get_one(input.data() + offset, n);
        if (llama_decode(ctx.get(), batch)) throw std::runtime_error("prefill failed");
        routing.finish(offset + n == count);
    }
    auto evaluate = [&](llama_token token) {
        routing.begin(true, 1);
        auto batch = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx.get(), batch)) throw std::runtime_error("decode failed");
        routing.finish(true);
    };
    const int vocab_size = llama_vocab_n_tokens(vocab);
    auto greedy = [&]() -> llama_token {
        const float * logits = llama_get_logits_ith(ctx.get(), -1);
        for (int i = 0; i < vocab_size; ++i) {
            if (!std::isfinite(logits[i])) throw std::runtime_error("non-finite model logits");
        }
        return std::max_element(logits, logits + vocab_size) - logits;
    };
    if (replay) {
        const llama_token next = greedy();
        std::vector<uint8_t> checkpoint(llama_state_get_size(ctx.get()));
        if (llama_state_get_data(ctx.get(), checkpoint.data(), checkpoint.size()) != checkpoint.size()) {
            throw std::runtime_error("cannot snapshot model state");
        }
        const auto route_checkpoint = routing.state;
        evaluate(next);
        const float * first = llama_get_logits_ith(ctx.get(), -1);
        const std::vector<float> expected(first, first + vocab_size);
        if (llama_state_set_data(ctx.get(), checkpoint.data(), checkpoint.size()) != checkpoint.size()) {
            throw std::runtime_error("cannot restore model state");
        }
        routing.state = route_checkpoint;
        evaluate(next);
        const float * actual = llama_get_logits_ith(ctx.get(), -1);
        double error = 0, magnitude = 0;
        for (int i = 0; i < vocab_size; ++i) {
            if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) throw std::runtime_error("replay produced non-finite logits");
            error += std::pow(actual[i] - expected[i], 2);
            magnitude += std::pow(expected[i], 2);
        }
        if (error / std::max(magnitude, 1e-12) > 1e-7) throw std::runtime_error("checkpoint replay mismatch");
        fprintf(stderr, "Edge0 checkpoint replay OK, normalized squared error %.3g\n", error / std::max(magnitude, 1e-12));
        if (llama_state_set_data(ctx.get(), checkpoint.data(), checkpoint.size()) != checkpoint.size()) throw std::runtime_error("cannot restore prompt");
        routing.state = route_checkpoint;
    }
    for (int i = 0; i < tokens; ++i) {
        const llama_token token = greedy();
        if (logits_file.is_open()) {
            logits_file.write(reinterpret_cast<const char *>(llama_get_logits_ith(ctx.get(), -1)), vocab_size * sizeof(float));
            if (!logits_file) throw std::runtime_error("cannot write logits file");
        }
        if (llama_vocab_is_eog(vocab, token)) break;
        std::vector<char> piece(128);
        int n = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true);
        if (n < 0) {
            piece.resize(-n);
            n = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true);
        }
        if (n < 0) throw std::runtime_error("cannot decode token text");
        fwrite(piece.data(), 1, n, stdout);
        fflush(stdout);
        if (i + 1 < tokens) evaluate(token);
    }
    printf("\n");
    llama_perf_context_print(ctx.get());
    return 0;
}

int main(int argc, char ** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception & e) {
        fprintf(stderr, "llama-edge0: %s\n", e.what());
        return 1;
    }
}
