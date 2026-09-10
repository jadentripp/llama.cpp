# Instructions for jadentripp/llama.cpp

## Autonomous work in this fork

The repository owner authorizes agents to complete requested work autonomously in this fork. This includes investigation, design, implementation, dependency setup, meaningful tests, documentation, upstream merges and conflict resolution, branches, commits, and pushes to `jadentripp/llama.cpp`. Do not ask for separate permission for these steps or routine implementation choices. Carry the requested work through to a usable result.

Use a task branch for substantial changes unless the owner requests another branch. Preserve existing user work and the Intel Mac discrete-Metal fixes when integrating upstream. Keep unrelated changes separate. Do not force-push shared history or delete unrelated work as a routine implementation step.

The autonomy above applies to this fork. It does not authorize initiating contributions, messages, or pull requests to `ggml-org/llama.cpp` or other people's repositories. Upstream contribution guidance in `CONTRIBUTING.md` applies when the owner specifically requests upstream contribution; it is not an approval gate for development in this fork.

## Implementation

- Read the relevant source before changing it. Reuse the existing graph, backend, model, and build infrastructure.
- Keep model behavior explicit. Experimental routing or quantization changes must be opt-in and must validate compatible weights and metadata.
- Respect tensor shape, stride, alignment, lifetime, and asynchronous device-ordering requirements. Bound staging and cache allocations.
- Preserve CPU fallback and backend portability. Use runtime capabilities rather than model names for hardware decisions.
- Keep comments concise and use ASCII in code and comments. Do not hard-wrap prose mid-sentence.
- Prefer small, reviewable commits. Use an `Assisted-by: Codex` trailer when committing on the owner's behalf.

## Verification and reporting

Run tests that address the concrete risks of the change. Reuse existing tests when suitable; add focused tests when they provide missing coverage. No separate permission is needed to add or run tests.

Distinguish compilation, synthetic tests, real-model generation, and target-device verification. Do not claim a Mac or Android result based only on Linux tests. Report remaining limitations candidly and provide reproducible build/run commands.

Keep the owner informed with concise progress updates, and continue work without repeated approval requests.

## Useful references

- `docs/build.md` and `docs/android.md`
- `docs/backend/INTEL-AMD-METAL.md`
- `docs/development/HOWTO-add-model.md`
- `tools/server/README-dev.md`
- `skills/` for relevant repository workflows
