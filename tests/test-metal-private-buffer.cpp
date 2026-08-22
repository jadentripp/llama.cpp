#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static bool check_equal(const uint8_t * actual, const uint8_t * expected, size_t size, const char * label) {
    for (size_t i = 0; i < size; ++i) {
        if (actual[i] != expected[i]) {
            std::fprintf(stderr, "%s: mismatch at byte %zu: got 0x%02x, expected 0x%02x\n",
                    label, i, actual[i], expected[i]);
            return false;
        }
    }

    return true;
}

int main() {
    ggml_backend_load_all();

    ggml_backend_dev_t dev = ggml_backend_dev_by_name("MTL0");
    if (dev == nullptr) {
        std::puts("SKIP: MTL0 is not available");
        return 0;
    }

    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    if (backend == nullptr) {
        std::fprintf(stderr, "failed to initialize MTL0\n");
        return 1;
    }

    const std::string buft_name = ggml_backend_buft_name(ggml_backend_get_default_buffer_type(backend.get()));
    const bool is_private = buft_name.find("Private") != std::string::npos;

    constexpr size_t staging_probe_size = 16*1024*1024;
    constexpr size_t tensor_size = 2*staging_probe_size + 65537;
    constexpr uint8_t sentinel = 0xa5;

    ggml_init_params params = {
        /* .mem_size   = */ 3*ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create ggml context\n");
        return 1;
    }

    // Start the test tensor at a nonzero buffer offset.
    ggml_tensor * pad = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, 256);
    ggml_tensor * tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, tensor_size);
    GGML_UNUSED(pad);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (buffer == nullptr) {
        std::fprintf(stderr, "failed to allocate %s test buffer\n", buft_name.c_str());
        return 1;
    }

    struct transfer_case {
        size_t host_offset;
        size_t tensor_offset;
        size_t size;
    };

    const transfer_case cases[] = {
        { 1,    1,      1 },
        { 31,   31,     4 },
        { 4095, 4095,   4095 },
        { 1,    8193,   4096 },
        { 31,   16385,  4097 },
        { 1,    65537,  607744 },
        { 1,    31,     staging_probe_size - 1 },
        { 31,   4095,   staging_probe_size },
        { 1,    8193,   staging_probe_size + 1 },
        { 31,   staging_probe_size - 31, 127 },
        { 1,    8193,   staging_probe_size + 4097 },
        { 31,   31,     2*staging_probe_size + 4097 },
    };

    for (size_t case_index = 0; case_index < sizeof(cases)/sizeof(cases[0]); ++case_index) {
        const transfer_case & tc = cases[case_index];

        std::vector<uint8_t> source(tc.host_offset + tc.size + 1);
        for (size_t i = 0; i < tc.size; ++i) {
            source[tc.host_offset + i] = (uint8_t) ((i*131 + case_index*17) & 0xff);
        }

        ggml_backend_tensor_memset(tensor, sentinel, 0, tensor_size);
        ggml_backend_tensor_set(tensor, source.data() + tc.host_offset, tc.tensor_offset, tc.size);

        constexpr uint8_t host_canary = 0x6d;
        std::vector<uint8_t> range_out(tc.host_offset + tc.size + 1, host_canary);
        ggml_backend_tensor_get(tensor, range_out.data() + tc.host_offset, tc.tensor_offset, tc.size);
        if (!check_equal(range_out.data() + tc.host_offset, source.data() + tc.host_offset, tc.size, "range transfer")) {
            return 1;
        }
        for (size_t i = 0; i < tc.host_offset; ++i) {
            if (range_out[i] != host_canary) {
                std::fprintf(stderr, "case %zu: leading host canary changed at byte %zu\n", case_index, i);
                return 1;
            }
        }
        if (range_out.back() != host_canary) {
            std::fprintf(stderr, "case %zu: trailing host canary changed\n", case_index);
            return 1;
        }

        // Verify bytes outside the partial write.
        std::vector<uint8_t> all_out(tensor_size + 1, 0);
        ggml_backend_tensor_get(tensor, all_out.data() + 1, 0, tensor_size);
        for (size_t i = 0; i < tensor_size; ++i) {
            const bool in_range = i >= tc.tensor_offset && i < tc.tensor_offset + tc.size;
            const uint8_t expected = in_range ? source[tc.host_offset + i - tc.tensor_offset] : sentinel;
            if (all_out[1 + i] != expected) {
                std::fprintf(stderr, "case %zu: sentinel mismatch at tensor byte %zu\n", case_index, i);
                return 1;
            }
        }
    }

    std::vector<uint8_t> cycle_source(tensor_size + 2);
    std::vector<uint8_t> cycle_snapshot(tensor_size + 2, 0x7b);
    std::vector<uint8_t> cycle_restored(tensor_size + 2, 0x7b);
    const size_t n_transfer_cycles = is_private ? 16 : 2;
    for (size_t cycle = 0; cycle < n_transfer_cycles; ++cycle) {
        for (size_t i = 0; i < tensor_size; ++i) {
            cycle_source[1 + i] = (uint8_t) ((i*131 + cycle*17) & 0xff);
        }

        ggml_backend_tensor_set(tensor, cycle_source.data() + 1, 0, tensor_size);
        ggml_backend_tensor_get(tensor, cycle_snapshot.data() + 1, 0, tensor_size);
        if (!check_equal(cycle_snapshot.data() + 1, cycle_source.data() + 1, tensor_size, "checkpoint snapshot")) {
            return 1;
        }

        ggml_backend_tensor_memset(tensor, (uint8_t) cycle, 0, tensor_size);
        ggml_backend_tensor_set(tensor, cycle_snapshot.data() + 1, 0, tensor_size);
        ggml_backend_tensor_get(tensor, cycle_restored.data() + 1, 0, tensor_size);
        if (!check_equal(cycle_restored.data() + 1, cycle_source.data() + 1, tensor_size, "checkpoint restore")) {
            return 1;
        }
        if (cycle_snapshot.front() != 0x7b || cycle_snapshot.back() != 0x7b ||
            cycle_restored.front() != 0x7b || cycle_restored.back() != 0x7b) {
            std::fprintf(stderr, "checkpoint cycle %zu changed a host canary\n", cycle);
            return 1;
        }
    }

    constexpr size_t n_threads = 4;
    constexpr size_t concurrent_size = 1024*1024 + 257;
    constexpr size_t concurrent_stride = concurrent_size + 31;
    std::atomic<bool> concurrent_ok(true);
    std::vector<std::thread> workers;
    for (size_t thread = 0; thread < n_threads; ++thread) {
        workers.emplace_back([&, thread]() {
            std::vector<uint8_t> source(concurrent_size + 2);
            std::vector<uint8_t> output(concurrent_size + 2, 0x4e);
            const size_t tensor_offset = 123 + thread*concurrent_stride;

            const size_t n_concurrent_cycles = is_private ? 8 : 2;
            for (size_t cycle = 0; cycle < n_concurrent_cycles; ++cycle) {
                for (size_t i = 0; i < concurrent_size; ++i) {
                    source[1 + i] = (uint8_t) ((i*67 + thread*29 + cycle*43) & 0xff);
                }

                ggml_backend_tensor_set(tensor, source.data() + 1, tensor_offset, concurrent_size);
                ggml_backend_tensor_get(tensor, output.data() + 1, tensor_offset, concurrent_size);
                if (std::memcmp(output.data() + 1, source.data() + 1, concurrent_size) != 0) {
                    std::fprintf(stderr, "concurrent transfer mismatch in thread %zu, cycle %zu\n", thread, cycle);
                    concurrent_ok.store(false);
                    return;
                }
                if (output.front() != 0x4e || output.back() != 0x4e) {
                    std::fprintf(stderr, "concurrent transfer changed a host canary in thread %zu, cycle %zu\n", thread, cycle);
                    concurrent_ok.store(false);
                    return;
                }
            }
        });
    }
    for (auto & worker : workers) {
        worker.join();
    }
    if (!concurrent_ok.load()) {
        return 1;
    }

    if (is_private) {
        constexpr size_t atomic_offset = 97;
        constexpr size_t atomic_size = staging_probe_size + 4097;
        std::vector<uint8_t> atomic_source(atomic_size + 1);
        std::vector<uint8_t> atomic_output(atomic_size);

        for (size_t cycle = 0; cycle < 4; ++cycle) {
            const uint8_t fill = (uint8_t) (0xd0 + cycle);
            for (size_t i = 0; i < atomic_size; ++i) {
                atomic_source[1 + i] = (uint8_t) ((i*73 + cycle*19) & 0xff);
            }

            std::atomic<int> ready(0);
            std::atomic<bool> start(false);
            std::thread writer([&]() {
                ready.fetch_add(1);
                while (!start.load()) {
                    std::this_thread::yield();
                }
                ggml_backend_tensor_set(tensor, atomic_source.data() + 1, atomic_offset, atomic_size);
            });
            std::thread filler([&]() {
                ready.fetch_add(1);
                while (!start.load()) {
                    std::this_thread::yield();
                }
                ggml_backend_tensor_memset(tensor, fill, atomic_offset, atomic_size);
            });
            while (ready.load() != 2) {
                std::this_thread::yield();
            }
            start.store(true);
            writer.join();
            filler.join();

            ggml_backend_tensor_get(tensor, atomic_output.data(), atomic_offset, atomic_size);
            if (std::memcmp(atomic_output.data(), atomic_source.data() + 1, atomic_size) != 0 &&
                !std::all_of(atomic_output.begin(), atomic_output.end(), [fill](uint8_t value) { return value == fill; })) {
                std::fprintf(stderr, "queue interleaving produced a torn transfer in cycle %zu\n", cycle);
                return 1;
            }
        }
    }

    // Check that a partial fill does not overwrite following bytes.
    ggml_backend_tensor_memset(tensor, sentinel, 0, tensor_size);
    ggml_backend_tensor_memset(tensor, 0x3c, 32, 100);
    std::vector<uint8_t> memset_out(tensor_size + 1, 0);
    ggml_backend_tensor_get(tensor, memset_out.data() + 1, 0, tensor_size);
    for (size_t i = 0; i < tensor_size; ++i) {
        const uint8_t expected = i >= 32 && i < 132 ? 0x3c : sentinel;
        if (memset_out[1 + i] != expected) {
            std::fprintf(stderr, "partial memset mismatch at tensor byte %zu\n", i);
            return 1;
        }
    }

    // Exercise the generic strided async fallback.
    constexpr size_t n_copies = 3;
    constexpr size_t row_size = 37;
    constexpr size_t tensor_stride = 101;
    constexpr size_t host_stride = 53;
    constexpr size_t tensor_offset = 7;

    std::vector<uint8_t> source_2d(1 + (n_copies - 1)*host_stride + row_size, 0);
    for (size_t row = 0; row < n_copies; ++row) {
        for (size_t i = 0; i < row_size; ++i) {
            source_2d[1 + row*host_stride + i] = (uint8_t) (row*67 + i);
        }
    }

    ggml_backend_tensor_memset(tensor, sentinel, 0, tensor_size);
    ggml_backend_tensor_set_2d_async(backend.get(), tensor, source_2d.data() + 1,
            tensor_offset, row_size, n_copies, tensor_stride, host_stride);
    ggml_backend_synchronize(backend.get());

    std::vector<uint8_t> output_2d(1 + (n_copies - 1)*host_stride + row_size, 0);
    ggml_backend_tensor_get_2d_async(backend.get(), tensor, output_2d.data() + 1,
            tensor_offset, row_size, n_copies, tensor_stride, host_stride);
    ggml_backend_synchronize(backend.get());

    for (size_t row = 0; row < n_copies; ++row) {
        if (!check_equal(output_2d.data() + 1 + row*host_stride,
                         source_2d.data() + 1 + row*host_stride, row_size, "2D async transfer")) {
            return 1;
        }
    }

    std::printf("OK: %s %s transfers\n", ggml_backend_dev_description(dev), is_private ? "private-buffer" : "shared-buffer");
    return 0;
}
