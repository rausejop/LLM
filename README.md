<!-- markdownlint-disable MD013 MD033 -->

# Dynamic-Load LLM Inference Engine — `llama.cpp` via `ctypes`

> A dependency-free, standard-library-only inference engine for **Llama 3.2 (1B, Q4_K_M / GGUF)** that runs inside a *hardened Windows environment where executables are blocked but dynamic libraries can still be mapped into a process*. Inference is driven entirely from Python through the C ABI of `llama.cpp`, loaded with `ctypes`.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12%2B-blue.svg">
  <img alt="Dependencies" src="https://img.shields.io/badge/dependencies-stdlib--only-success.svg">
  <img alt="Backend" src="https://img.shields.io/badge/backend-llama.cpp%20b9518-orange.svg">
  <img alt="Platform" src="https://img.shields.io/badge/platform-Windows%20x64-lightgrey.svg">
  <img alt="Style" src="https://img.shields.io/badge/style-PEP8%20%2B%20Google-9cf.svg">
</p>

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [The Constrained-Environment Problem](#2-the-constrained-environment-problem)
3. [System Architecture](#3-system-architecture)
4. [Theoretical Background: How a Decoder-Only LLM Generates Text](#4-theoretical-background-how-a-decoder-only-llm-generates-text)
5. [The GGUF Model and K-Quant Quantization](#5-the-gguf-model-and-k-quant-quantization)
6. [`carga.py` — The Provisioning Stage](#6-cargapy--the-provisioning-stage)
7. [`inferencia.py` — The Inference Engine](#7-inferenciapy--the-inference-engine)
8. [The Foreign Function Interface: Why ctypes Works Here](#8-the-foreign-function-interface-why-ctypes-works-here)
9. [The Struct-by-Value Problem and Its Solution](#9-the-struct-by-value-problem-and-its-solution)
10. [Runtime Backend Loading (the ggml plugin model)](#10-runtime-backend-loading-the-ggml-plugin-model)
11. [The Generation Loop, Step by Step](#11-the-generation-loop-step-by-step)
12. [Memory, Lifetimes, and the GIL](#12-memory-lifetimes-and-the-gil)
13. [Performance Characteristics](#13-performance-characteristics)
14. [Security Model and Threat Considerations](#14-security-model-and-threat-considerations)
15. [Reproducibility](#15-reproducibility)
16. [Usage Reference](#16-usage-reference)
17. [Troubleshooting](#17-troubleshooting)
18. [Limitations and Future Work](#18-limitations-and-future-work)
19. [Glossary](#19-glossary)
20. [References](#20-references)

---

## 1. Abstract

This project implements a complete autoregressive text-generation pipeline for a quantized Large Language Model **without invoking a single external process and without any third-party Python package**. The only Python runtime requirement is the standard library; the only native requirement is the set of `llama.cpp` dynamic libraries, which are loaded into the interpreter's address space at runtime via `ctypes`.

The work is split into two scripts with a deliberately strict separation of concerns:

- **`carga.py`** ("load/provision") is the *acquisition* stage. It fetches the model weights and the native libraries over HTTPS and lays out the working directory. It performs only I/O — it never executes a binary.
- **`inferencia.py`** ("inference") is the *execution* stage. It maps the `llama.cpp` C API into Python, loads the model, and runs the token-by-token decoding loop, emitting the generated text to a file and, optionally, a maximally verbose debug trace to standard output.

The remainder of this document explains, at a graduate level, **how** the engine was built, **how** it works internally, **why** the approach is sound at the level of the application binary interface (ABI), and **how** LLM inference can be realized at all inside an environment that forbids ordinary program execution.

---

## 2. The Constrained-Environment Problem

### 2.1 The restriction

The target is a **locked-down Windows host** on which application-execution policy (e.g., *AppLocker* or *Windows Defender Application Control / WDAC*) prevents users from launching arbitrary `.exe` files. This is a standard enterprise hardening posture: it shrinks the attack surface by ensuring only signed, allow-listed executables run.

The canonical way to run a local LLM — downloading `llama-cli.exe`, `ollama.exe`, or a packaged Python interpreter and double-clicking it — is therefore **unavailable**.

### 2.2 The opening

Application-control policies almost universally govern **executable image launch** (`CreateProcess` of a new `.exe`). They typically do **not** prohibit an *already-trusted, already-running* process from mapping a **dynamic-link library (DLL)** into its own address space via `LoadLibrary`. In our case the already-trusted process is the organization-sanctioned **Python interpreter** itself.

This asymmetry is the entire basis of the design:

> If we cannot *start* a new program, we can instead *teach an existing, allowed program* (Python) to perform the computation by loading native code as a library.

`llama.cpp` is distributed not only as command-line executables but also as a set of **shared libraries** (`llama.dll`, `ggml.dll`, `ggml-base.dll`, and a family of CPU backend plugins). These libraries expose a stable **C API**. Python's `ctypes` module can `LoadLibrary` a DLL and call its exported C functions directly. Hence: **no `.exe`, no compiler, no pip — yet full local inference.**

### 2.3 Why `ctypes` specifically

`ctypes` is part of the CPython standard library. It provides:

- `ctypes.CDLL(path)` → a thin wrapper over the platform loader (`LoadLibraryExW` on Windows), returning a handle whose attributes resolve to exported symbols.
- A type system (`c_int32`, `c_void_p`, `Structure`, `CFUNCTYPE`, …) that lets Python describe C declarations precisely enough for the FFI marshaller to lay out arguments and interpret return values.
- Callback synthesis (`CFUNCTYPE`) so C code can call *back into* Python — essential for routing `llama.cpp`'s logging.

Because `ctypes` ships with CPython, using it adds **zero** to the dependency footprint, which is the project's hard constraint.

---

## 3. System Architecture

```
                         ┌──────────────────────────────────────────────┐
                         │                 carga.py                      │
                         │  (provisioning — network + unzip only)        │
                         │                                               │
   Hugging Face  ───────▶│  • download GGUF weights (~770 MiB)           │
   GitHub Releases ─────▶│  • download pinned llama.cpp b9518 (CPU x64)  │
                         │  • flatten *.dll into ./lib                    │
                         └───────────────────────┬──────────────────────┘
                                                 │  writes
                                                 ▼
   ./llama-3.2-1b-q4_k_m.gguf      ./lib/llama.dll, ggml*.dll, ggml-cpu-*.dll
                                                 │
                                                 │  read by
                                                 ▼
                         ┌──────────────────────────────────────────────┐
                         │              inferencia.py                    │
                         │   (stdlib + ctypes; never spawns a process)   │
                         │                                               │
   prompt.txt  ─────────▶│  load_library → bind_api → run_inference      │──▶ output.txt
                         │                                               │
                         │  ctypes.CDLL ──FFI──▶ llama.dll ──▶ ggml ──▶ CPU SIMD
                         └───────────────────────────────────────────────┘
                                                 │ (verbose)
                                                 ▼
                                          stdout debug trace
```

The two stages are decoupled by the **filesystem**: `carga.py` produces a directory layout, and `inferencia.py` consumes it. Either can be re-run independently; `carga.py` is idempotent (it skips artifacts that already exist).

| File | Role | External I/O | Native code |
|------|------|--------------|-------------|
| `carga.py` | Acquire model + libraries | HTTPS download, unzip | none |
| `inferencia.py` | Run autoregressive decoding | reads model/prompt, writes output | `llama.dll` via `ctypes` |
| `prompt.txt` | Default input prompt | — | — |
| `output.txt` | Default generated output | — | — |
| `lib/` | Native libraries (generated) | — | the DLLs themselves |

---

## 4. Theoretical Background: How a Decoder-Only LLM Generates Text

To understand what the engine is actually orchestrating, it helps to restate the mechanics of a modern transformer language model. Llama 3.2 1B is a **decoder-only, causal transformer**. The metadata embedded in the GGUF file (extracted verbatim from a verbose run) is:

| Hyperparameter | Value | Symbol |
|----------------|-------|--------|
| Layers (blocks) | 16 | `n_layer` |
| Embedding width | 2048 | `n_embd` |
| Attention heads | 32 | `n_head` |
| KV heads (grouped) | 8 | `n_head_kv` |
| Head dimension | 64 | `n_embd_head` |
| GQA group factor | 4 | `n_gqa = n_head / n_head_kv` |
| Feed-forward width | 8192 | `n_ff` |
| Vocabulary | 128 256 | `n_vocab` |
| Trained context | 131 072 | `n_ctx_train` |
| RoPE base frequency | 500 000 | `freq_base` |
| Parameters | 1.24 B | — |

### 4.1 Autoregressive factorization

A language model defines a probability distribution over token sequences and factorizes it by the chain rule:

```
P(x₁, x₂, …, x_T) = ∏_{t=1}^{T} P(x_t | x_1, …, x_{t-1})
```

Generation is the act of sampling (or, here, taking the mode of) each conditional `P(x_t | x_{<t})` one token at a time, appending the result, and repeating. This is precisely the loop implemented in `run_inference`.

### 4.2 The forward pass

For each position the model computes:

1. **Embedding lookup** — token id → a 2048-dimensional vector (`token_embd.weight`).
2. **16 transformer blocks**, each applying:
   - **RMSNorm** (Llama uses RMS normalization, `f_norm_rms_eps = 1e-5`).
   - **Grouped-Query Attention (GQA)** with **Rotary Position Embeddings (RoPE)**. With 32 query heads but only 8 key/value heads (`n_gqa = 4`), every group of four query heads shares one K/V head, cutting the KV-cache size by 4× relative to full multi-head attention — a key reason a 1B model is comfortable on CPU.
   - **SwiGLU feed-forward** network of inner width 8192 (`ffn_gate`, `ffn_up`, `ffn_down`).
   - Residual connections around both sublayers.
3. **Final RMSNorm** and an **output projection** (`output.weight`, tied or untied) producing a **logit vector** of length `n_vocab = 128 256`.

### 4.3 Rotary Position Embeddings

Llama has no learned positional table; instead RoPE rotates the query/key vectors by an angle proportional to the absolute position, with per-dimension frequencies derived from `freq_base = 500 000`. The dot product of two rotated vectors depends only on their **relative** offset, giving the model relative-position awareness and the ability to extrapolate. Crucially for us, this means **position must be tracked correctly across decode calls** — a responsibility `llama.cpp` assumes automatically when we feed tokens via `llama_batch_get_one` (see §11).

### 4.4 The KV cache

Naïvely, generating token *t* would recompute attention over all previous positions from scratch — `O(t²)` work per step. Instead, the **key and value projections of every past token are cached**. At each step only the new token's Q/K/V are computed; its Q attends against the cached K/V. This converts per-step cost from quadratic to linear in context length.

For this model the cache footprint (also taken from a live run) is:

```
llama_kv_cache: size = 64.00 MiB (2048 cells, 16 layers, 1 seq),
                K (f16): 32.00 MiB, V (f16): 32.00 MiB
```

We can derive this. Per layer, the cache stores K and V of width `n_embd_k_gqa = n_head_kv × n_embd_head = 8 × 64 = 512` each, in FP16 (2 bytes):

```
bytes = 2 (K&V) × n_layer × n_ctx × n_embd_k_gqa × 2 bytes
      = 2 × 16 × 2048 × 512 × 2
      = 67 108 864 B = 64 MiB     ✓
```

The `--n-ctx` flag directly scales this number; halving the context window halves the KV cache.

---

## 5. The GGUF Model and K-Quant Quantization

### 5.1 GGUF

**GGUF** (GGML Universal File) is `llama.cpp`'s self-describing single-file model format. It contains a key/value metadata header (architecture, hyperparameters, tokenizer, chat template) followed by the tensor data. Our file reports `GGUF V3`, 147 tensors, 29 metadata pairs, and a tokenizer of type `gpt2` with `llama-bpe` pre-tokenization and **280 147 BPE merges**.

Because the metadata is embedded, the engine does **not** hardcode the model's shape — `llama.cpp` reads `n_layer`, `n_embd`, `n_vocab`, etc. from the file at load time. Our Python code only needs to query `n_vocab` (to size the logits read) and `n_ctx`.

### 5.2 Q4_K_M ("K-quant, medium")

The weights are stored in the **Q4_K** K-quant scheme at the "medium" mixing level. K-quants partition each tensor row into **super-blocks** (256 weights) subdivided into 16 sub-blocks of 16 weights. Each sub-block has its own 4-bit scale and minimum, themselves quantized against a super-block-level FP16 scale/min — a two-level scheme that preserves dynamic range far better than naïve per-tensor scaling.

"Medium" (`_M`) means a *mixed* assignment: the most error-sensitive tensors (attention `wv` and the feed-forward `w2`/`down` projections) are kept at higher precision (Q6_K) while the bulk use Q4_K. The GGUF reports exactly this:

```
type  f32:  34 tensors      (norms, biases — full precision)
type q4_K:  96 tensors      (bulk weights — 4-bit)
type q6_K:  17 tensors      (sensitive weights — 6-bit)
file size = 762.81 MiB (5.18 bits-per-weight effective)
```

At **5.18 bits per weight**, a 1.24 B-parameter model fits in ~763 MiB instead of ~2.5 GiB (FP16), making CPU inference with `mmap` practical.

### 5.3 Runtime repacking

On load you will see `repack: ... with q4_K_8x8` lines. The CPU backend **re-lays-out** Q4_K blocks into an interleaved `8x8` arrangement that matches the SIMD register width of the detected microarchitecture, so that the quantized matrix-vector kernels can stream weights with aligned, vectorized loads. This is a pure performance transform of the in-memory representation; the on-disk file is untouched.

---

## 6. `carga.py` — The Provisioning Stage

### 6.1 Responsibilities

1. Download the GGUF weights from Hugging Face.
2. Download the `llama.cpp` Windows-x64 **CPU** build and extract its DLLs.
3. Produce the directory layout `inferencia.py` expects.

All using only `urllib`, `json`, `zipfile`, `re`, and `pathlib`.

### 6.2 Pinned build with API fallback

The native libraries are version-sensitive (see §9), so the primary source is a **pinned release**:

```python
LLAMA_ZIP_URL = (
    'https://github.com/ggml-org/llama.cpp/releases/download/'
    'b9518/llama-b9518-bin-win-cpu-x64.zip'
)
```

If that exact URL ever 404s (builds are periodically pruned), `find_llama_asset()` falls back to the GitHub **`releases/latest`** API and selects an asset by an ordered preference list — generic `cpu-x64` first, then `avx2`, `avx`, `noavx` — while **excluding** hardware-specific builds (`cuda`, `vulkan`, `hip`, `sycl`, `arm64`, …) that would demand drivers or runtimes absent from a hardened host.

### 6.3 Robust downloads

- **Idempotent**: a file already present with non-zero size is skipped, so re-running never re-downloads ~800 MiB.
- **Atomic**: bytes stream into a `*.part` file that is `Path.replace()`-d into place only on success, so an interrupted download cannot masquerade as a complete one.
- **User-Agent header**: GitHub's API rejects requests without one; redirects (Hugging Face → CDN) are followed transparently by `urllib`.

### 6.4 DLL flattening

The release archive nests binaries under a folder; `extract_dlls()` keeps only `*.dll` members and writes each to `lib/<basename>`, then asserts that `llama.dll` is present. Flattening guarantees a single, predictable search directory for the loader (§10).

---

## 7. `inferencia.py` — The Inference Engine

The engine is organized into four sections, mirrored in the source:

1. **ctypes structs** passed/returned by value (`LlamaModelParams`, `LlamaContextParams`, `LlamaBatch`).
2. **DLL loading & C-API binding** (`load_library`, `_bind`, `bind_api`).
3. **Inference logic** (`tokenize_text`, `token_to_bytes`, `run_inference`).
4. **CLI** (`build_parser`, `main`).

The high-level call sequence is:

```
main
 └─ load_library(lib_dir)            # map DLLs, install log cb, register backends
     └─ ctypes.CDLL(llama.dll)
 └─ bind_api(lib)                    # set restype/argtypes for every C function
 └─ run_inference(...)
     ├─ llama_backend_init
     ├─ llama_model_load_from_file
     ├─ llama_init_from_model        # create context (KV cache lives here)
     ├─ llama_tokenize               # prompt → token ids
     ├─ llama_decode                 # prefill
     └─ loop: get_logits_ith → argmax → token_to_piece → batch_get_one → decode
```

The **C-API surface** the engine binds is summarized below. Each row is a real C declaration mapped to a `ctypes` signature in `bind_api`.

| C function (with common aliases) | Purpose | `ctypes` return / args |
|---|---|---|
| `llama_backend_init` / `_free` | Global init/teardown | `None` / `()` |
| `llama_model_default_params` | Default model params (by value) | `LlamaModelParams` / `()` |
| `llama_model_load_from_file` *(alias `llama_load_model_from_file`)* | Load GGUF | `c_void_p` / `(c_char_p, LlamaModelParams)` |
| `llama_model_get_vocab` | Get vocab handle | `c_void_p` / `(c_void_p)` |
| `llama_context_default_params` | Default ctx params (by value) | `LlamaContextParams` / `()` |
| `llama_init_from_model` *(alias `llama_new_context_with_model`)* | Create context + KV cache | `c_void_p` / `(c_void_p, LlamaContextParams)` |
| `llama_n_ctx` | Effective context size | `c_uint32` / `(c_void_p)` |
| `llama_vocab_n_tokens` *(alias `llama_n_vocab`)* | Vocabulary size | `c_int32` / `(c_void_p)` |
| `llama_tokenize` | Text → token ids | `c_int32` / `(vocab, char*, len, int32*, max, bool, bool)` |
| `llama_token_to_piece` | Token id → UTF-8 bytes | `c_int32` / `(vocab, token, char*, len, lstrip, special)` |
| `llama_vocab_is_eog` *(alias `llama_token_is_eog`)* | End-of-generation test | `c_bool` / `(vocab, token)` |
| `llama_batch_get_one` | Wrap tokens in a batch (by value) | `LlamaBatch` / `(int32*, n)` |
| `llama_decode` | Run one forward pass | `c_int32` / `(ctx, LlamaBatch)` |
| `llama_get_logits_ith` | Pointer to a row of logits | `POINTER(c_float)` / `(ctx, i)` |
| `ggml_backend_load_all_from_path` | Register CPU backends | `None` / `(c_char_p)` |
| `ggml_log_set` / `llama_log_set` | Install log callback | `None` / `(LOG_CALLBACK, c_void_p)` |

The **alias** mechanism in `_bind` is a deliberate hedge against API churn: `llama.cpp` has renamed several of these symbols over time, and binding by *primary name with fallbacks* lets the same script work across a range of builds.

---

## 8. The Foreign Function Interface: Why ctypes Works Here

Calling C from Python is not magic; it is the disciplined construction of a stack frame that the native function expects. Three things must agree between caller and callee:

1. **Symbol resolution** — the function must be found by name in the DLL's export table.
2. **Type marshalling** — each Python value must be converted to the C representation of the declared parameter type, and the return value interpreted accordingly.
3. **Calling convention / ABI** — argument *placement* (registers vs. stack), *ownership*, and *cleanup* must match.

`ctypes` handles (1) via attribute lookup on the `CDLL` handle, and (2) via the `restype`/`argtypes` we declare in `bind_api`. Point (3) is where this project's subtlety lives.

### 8.1 One calling convention to rule them all (on x64)

On 32-bit Windows there were multiple conventions (`cdecl`, `stdcall`, `fastcall`), and choosing wrongly corrupts the stack. On **x86-64 Windows there is exactly one** convention (the *Microsoft x64 calling convention*): the first four integer/pointer arguments go in `RCX, RDX, R8, R9`, floating-point in `XMM0–3`, the rest on the stack, and the **caller** cleans up. `ctypes.CDLL` uses this convention, and so does every function `llama.cpp` exports. This uniformity is *why* a hand-written `ctypes` binding can be correct without us ever specifying a convention — provided we target x64, which `carga.py` guarantees by downloading the `*-x64` package.

### 8.2 Scalars, pointers, and strings

- **Scalars** (`c_int32`, `c_uint32`, `c_bool`, `c_float`) map 1:1 to C integers/floats.
- **Opaque handles** (`llama_model*`, `llama_context*`, `llama_vocab*`) are declared `c_void_p`: we never dereference them in Python; we only pass them back to the library.
- **Strings**: Python `bytes` marshal to `const char*`. We always `.encode('utf-8')` explicitly, which matters because the model paths and prompts may contain non-ASCII characters.
- **Output buffers**: for `llama_tokenize` and `llama_token_to_piece` we allocate a `ctypes` array / `create_string_buffer`, pass it in, and read back the count the function returns. Both functions follow the C idiom of returning a **negative required-size** when the buffer is too small; the engine honors this by reallocating and retrying.

---

## 9. The Struct-by-Value Problem and Its Solution

This is the most technically delicate part of the project and the one most likely to crash a naïve binding.

### 9.1 The problem

`llama.cpp` configures the model and context through **structs passed by value**:

```c
struct llama_model_params   llama_model_default_params(void);
struct llama_context_params llama_context_default_params(void);
llama_model* llama_model_load_from_file(const char*, struct llama_model_params);
llama_context* llama_init_from_model(llama_model*, struct llama_context_params);
```

To pass or receive a struct *by value*, `ctypes` must know its **exact size and field offsets**. But these structs are **not stable across versions** — `llama.cpp` regularly adds, reorders, or removes fields (`n_gpu_layers`, `rpc_servers`, `tensor_buft_overrides`, flash-attention flags, etc.). A `ctypes.Structure` that mismatches the DLL's real layout will:

- read configuration from the **wrong offsets**, and/or
- under-allocate the return buffer, so the callee **writes past** our memory — undefined behavior, typically a crash.

Hardcoding the full struct for one exact commit is brittle; the binding would break the next time the libraries are updated.

### 9.2 The solution: opaque blob + stable-prefix-with-padding

The engine sidesteps layout fragility with two complementary techniques.

**(a) Treat `llama_model_params` as an opaque blob.**

```python
class LlamaModelParams(ctypes.Structure):
    _fields_ = [('_raw', ctypes.c_byte * 256)]
```

We never read or write its fields. We obtain it from `llama_model_default_params()` and pass it straight back to `llama_model_load_from_file`. For a CPU build the defaults are already correct (`n_gpu_layers == 0`). The 256-byte size is comfortably larger than any real version of the struct, so the by-value **return** always fits.

**(b) Declare only the *stable leading fields* of `llama_context_params`, then pad.**

```python
class LlamaContextParams(ctypes.Structure):
    _fields_ = [
        ('n_ctx',           ctypes.c_uint32),
        ('n_batch',         ctypes.c_uint32),
        ('n_ubatch',        ctypes.c_uint32),
        ('n_seq_max',       ctypes.c_uint32),
        ('n_threads',       ctypes.c_int32),
        ('n_threads_batch', ctypes.c_int32),
        ('_rest',           ctypes.c_byte * 512),   # padding for the tail
    ]
```

The leading fields of `llama_context_params` (`n_ctx`, batch sizes, thread counts) have been **stable for years** and sit at fixed offsets `0,4,8,…`. We set only those. Everything after them — rope/yarn parameters, pooling/attention enums, FP-type selectors, boolean flags — is absorbed by the 512-byte `_rest` blob, populated for us by `llama_context_default_params()`.

### 9.3 Why this is *correct*, not just lucky

The padding trick is safe **because of the x64 ABI**, not in spite of it.

> On the Microsoft x64 ABI, an aggregate larger than 8 bytes (or whose size is not 1/2/4/8) is **never** passed in registers. It is passed by an **implicit reference to a caller-allocated temporary**, and a large struct **return** is materialized through a hidden pointer (in `RCX`) to caller-allocated space.

Consequences:

- **Receiving** a struct by value: `ctypes` allocates `sizeof(OurStruct)` and hands its address to the callee as the hidden return pointer. The callee writes `sizeof(RealStruct)` bytes. As long as `sizeof(OurStruct) ≥ sizeof(RealStruct)` — guaranteed by our padding — every write lands inside our buffer. Trailing padding is simply left untouched or filled with real tail fields; either way it is *inside our allocation*.
- **Passing** a struct by value: `ctypes` places a pointer to our buffer in the argument slot. The callee reads `sizeof(RealStruct)` bytes from it. Since our leading fields are at the correct offsets and the tail was filled by `default_params`, the callee sees a *valid, fully-initialized* struct.

The one thing this scheme cannot tolerate is a new field being inserted **before** a field we explicitly set. That is why we are conservative: we only set the long-stable prefix of `context_params` and we set *nothing* in `model_params`. The result is a binding that survives library upgrades that would shatter a fully-specified struct.

### 9.4 `llama_batch` is declared exactly

By contrast, `llama_batch` has a stable, public layout and is small enough that we declare it field-for-field:

```python
class LlamaBatch(ctypes.Structure):
    _fields_ = [
        ('n_tokens', ctypes.c_int32),
        ('token',    ctypes.POINTER(ctypes.c_int32)),
        ('embd',     ctypes.POINTER(ctypes.c_float)),
        ('pos',      ctypes.c_void_p),
        ('n_seq_id', ctypes.c_void_p),
        ('seq_id',   ctypes.c_void_p),
        ('logits',   ctypes.c_void_p),
    ]
```

`llama_batch_get_one` returns it by value and `llama_decode` consumes it by value; both are >8 bytes, so both go through the hidden-pointer path described above.

---

## 10. Runtime Backend Loading (the ggml plugin model)

### 10.1 The failure that motivated it

A first, "obvious" implementation loaded `llama.dll` (with `ggml.dll` as a dependency) and immediately tried to load the model. It failed:

```
llama_model_load_from_file_impl: no backends are loaded.
hint: use ggml_backend_load() or ggml_backend_load_all() ...
```

### 10.2 Why

Modern `ggml` decouples the *tensor library* from the *compute backends*. The CPU kernels live in **separate plugin DLLs**, one per microarchitecture tier:

```
ggml-cpu-sandybridge.dll   ggml-cpu-haswell? / sse42.dll   ggml-cpu-skylakex.dll
ggml-cpu-icelake.dll       ggml-cpu-sapphirerapids.dll     ggml-cpu-zen4.dll   ...
```

Merely `LoadLibrary`-ing these files does **not** register them: a backend advertises itself only through a registration entry point that the host must invoke. The correct call is:

```c
void ggml_backend_load_all_from_path(const char* dir);
```

which scans a directory, loads every `ggml-cpu-*.dll`, queries each one's CPU-feature requirements, and **selects the best variant the running CPU supports**. On the development machine that selection was:

```
load_backend: loaded RPC backend from ...\ggml-rpc.dll
load_backend: loaded CPU backend from ...\ggml-cpu-icelake.dll
```

i.e. the Ice Lake AVX-512 kernels were chosen automatically.

### 10.3 The fix, and the symbol-location subtlety

`load_library` therefore, **before loading the model**, calls `ggml_backend_load_all_from_path(lib_dir)`. A second subtlety surfaced during debugging: the symbol is **not** exported by `ggml-base.dll` (the first dependency one might guess) but by **`ggml.dll`**. This was confirmed by scanning the export strings of each library. Accordingly, `_register_backends` searches the loaded ggml handles **in reverse** (so `ggml.dll` is tried first) and raises a clear `InferenceError` if no registration symbol is found at all.

### 10.4 Windows DLL search path

For `llama.dll`'s dependencies (`ggml*.dll`) to resolve, the engine:

1. Calls `os.add_dll_directory(lib_dir)` (the modern, secure replacement for mutating `PATH`; available since Python 3.8). A `PATH` fallback covers exotic environments.
2. Pre-loads `ggml-base.dll` and `ggml.dll` explicitly before `llama.dll`, so the loader already has them resident.

---

## 11. The Generation Loop, Step by Step

The heart of `run_inference` is the autoregressive loop. Annotated:

```python
# Prefill: encode the whole prompt in one forward pass.
token_array = (ctypes.c_int32 * len(tokens))(*tokens)
batch = api.batch_get_one(token_array, len(tokens))
api.decode(context, batch)                 # KV cache now holds the prompt

single = (ctypes.c_int32 * 1)()            # reused 1-token buffer
for _ in range(n_predict):
    logits_ptr = api.get_logits_ith(context, -1)   # logits of last position
    logits = logits_ptr[:n_vocab]                  # 128 256 floats
    best = max(range(n_vocab), key=logits.__getitem__)   # greedy argmax
    if api.is_eog(vocab, best):
        break
    emit(token_to_text(best))              # to file (+ stdout if verbose)
    single[0] = best
    batch = api.batch_get_one(single, 1)   # position tracked automatically
    api.decode(context, batch)             # extend KV cache by one
```

### 11.1 Prefill vs. decode

The first `llama_decode` processes all prompt tokens together (**prefill**) — efficient because the matrix multiplications batch over the sequence dimension. Thereafter each step decodes a **single** token, the per-token cost being one pass that attends against the growing KV cache.

### 11.2 Automatic position tracking

`llama_batch_get_one(tokens, n)` produces a batch with `pos = NULL`; `llama_decode` then assigns positions **sequentially from the context's internal cursor**. This is why we can feed one token at a time without ever computing positions ourselves — and why RoPE (§4.3) still receives correct absolute positions. Mixing manual and automatic positioning would corrupt this cursor; the engine never does.

### 11.3 Greedy decoding and the softmax shortcut

We select `argmax(logits)` directly, **without** computing a softmax. This is mathematically exact for greedy decoding: softmax is strictly monotonic, so

```
argmax_i softmax(logits)_i = argmax_i logits_i .
```

Computing the 128 256-way softmax every step would be pure waste. We also deliberately **bypass `llama.cpp`'s sampler-chain API**, which is comparatively volatile and would require additional by-value structs; argmax over the raw logits pointer is both simpler and more robust. Temperature, top-k, or nucleus sampling could be added entirely in Python over the same logits slice (see §18).

### 11.4 End-of-generation

`llama_vocab_is_eog` recognizes any end-of-generation token. For Llama 3.2 the EOG set (from the run) is `<|end_of_text|>` (128001), `<|eom_id|>` (128008), and `<|eot_id|>` (128009). Hitting any of them terminates the loop early.

### 11.5 Incremental UTF-8 decoding

A single token's bytes (`llama_token_to_piece`) may be a **fragment** of a multi-byte UTF-8 character — common with non-ASCII text where one glyph spans two BPE tokens. Decoding each fragment eagerly would raise `UnicodeDecodeError` or emit replacement characters. The engine instead **accumulates bytes** in a `bytearray` and flushes only when `bytes.decode('utf-8')` succeeds, holding back incomplete sequences until the next token completes them. Any genuinely truncated tail at the end of generation is flushed with `errors='replace'`.

### 11.6 Output routing and the log callback

Two output channels are cleanly separated:

- **The generated text** always goes to the output file (`output.txt` / `--o`).
- **The native debug log** — hundreds of `llama.cpp`/`ggml` lines — is routed through a `ctypes` **callback** installed via `ggml_log_set` and `llama_log_set`. With `--verbose` the callback writes everything to stdout (maximum-debug trace); without it, the callback **discards** every line, leaving stdout pristine apart from a final `Output written to …` confirmation.

This is why the same binary behavior yields either a silent, file-only run or a fully instrumented trace, selected by a single flag.

---

## 12. Memory, Lifetimes, and the GIL

Mixing a managed runtime with manual C resources requires care on two fronts.

### 12.1 Callback lifetime

`ctypes` synthesizes a C-callable trampoline from a Python function via `CFUNCTYPE`. If the resulting object is garbage-collected while C still holds the pointer, the next log line calls **freed memory** — a crash. The engine therefore appends every callback instance to a module-level list, `_KEPT_ALIVE`, pinning it for the process lifetime. This is a standard, mandatory `ctypes` idiom that is easy to omit and catastrophic to forget.

### 12.2 Native resource teardown

`llama_context`, `llama_model`, and the global backend are C-owned and not visible to Python's garbage collector. `run_inference` releases them in a `finally` block, in strict reverse order of acquisition:

```
llama_free(context) → llama_model_free(model) → llama_backend_free()
```

so that even an exception mid-generation does not leak the multi-hundred-megabyte model mapping.

### 12.3 The GIL during callbacks

When C invokes the Python log callback, `ctypes` re-acquires the Global Interpreter Lock before running Python code, making `sys.stdout.write` inside the callback safe. The callback also wraps its body in a bare `except` so that *no* logging error can ever propagate into — and abort — the native call stack.

---

## 13. Performance Characteristics

- **Threads.** The engine sets `n_threads = n_threads_batch = os.cpu_count()`, parallelizing both prefill and decode across all logical cores.
- **`mmap`.** The model is memory-mapped (`mmap = true`), so the 763 MiB of weights are demand-paged from the OS file cache rather than copied; cold start touches only what attention actually reads.
- **SIMD dispatch.** Backend auto-selection (§10.2) picks the widest instruction set the CPU supports (AVX-512 on the test machine), and the `q4_K_8x8` repack aligns quantized weights to those vector widths.
- **KV cache vs. context.** Memory grows linearly with `--n-ctx` (§4.4). The default of 2048 is far below the model's trained 131 072, trading unused context for a small footprint; raise it only as far as your prompts require.
- **Flash attention.** The build negotiates `flash_attn = auto` and enables it during the reservation phase, reducing attention memory traffic.

The dominant cost per generated token is the quantized matrix-vector products against ~1.24 B weights; on a modern desktop core-count this yields interactive (tens of tokens/second) throughput for this 1B model.

---

## 14. Security Model and Threat Considerations

- **No new process image is created.** The engine respects the *spirit* of an execution-control policy: it adds capability to an *already-sanctioned* interpreter rather than smuggling in a new program. Whether this is *permitted* in a given org is a policy question; this document describes the mechanism, not an endorsement of bypassing controls you are not authorized to bypass.
- **Provenance.** Both the model (Hugging Face) and libraries (official `ggml-org/llama.cpp` GitHub releases) are fetched over HTTPS from named, pinned sources. For a high-assurance deployment you should additionally verify SHA-256 digests / release signatures before loading — DLLs execute with the full privileges of the Python process.
- **Supply-chain surface.** Loading a DLL is equivalent to running its code. The pinned-build approach makes the loaded bytes reproducible; the API fallback widens trust to "whatever is latest" and should be disabled in locked deployments.
- **No telemetry / no network at inference time.** `inferencia.py` performs zero network I/O; it only reads local files and writes the output file.

---

## 15. Reproducibility

| Component | Pinned value |
|---|---|
| Model | `medmekk/Llama-3.2-1B-Q4_K_M-GGUF` → `llama-3.2-1b-q4_k_m.gguf` |
| Model size | 762.81 MiB, 5.18 BPW, GGUF V3 |
| Native libraries | `llama.cpp` release **b9518**, `bin-win-cpu-x64` |
| Selected CPU backend | auto (`ggml-cpu-icelake.dll` on the dev host) |
| Python | 3.12+ (developed/tested on CPython 3.14) |
| Platform | Windows 11 x64 |

Because the build is pinned, the struct layouts that §9 depends on are fixed for the primary path; upgrading the libraries exercises the robustness of the blob/padding strategy rather than requiring code changes.

---

## 16. Usage Reference

```powershell
# 1) Provision (downloads ~800 MiB model + ~16 MiB libraries; idempotent).
python carga.py

# 2) Run with defaults  (model=llama-3.2-1b-q4_k_m.gguf, prompt=prompt.txt, output=output.txt)
python inferencia.py

# 3) Maximum-debug trace to stdout, custom output file and length.
python inferencia.py --prompt prompt.txt --output result.txt --n-predict 128 --verbose

# Introspection
python inferencia.py --help
python inferencia.py --version        # also: --v

# Static syntax check (needs neither model nor DLLs)
python -m py_compile carga.py inferencia.py
```

### Command-line flags (`inferencia.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `llama-3.2-1b-q4_k_m.gguf` | Path to the GGUF model |
| `--prompt` | `prompt.txt` | Path to the prompt text file |
| `--o`, `--output` | `output.txt` | File to write the generated text to |
| `--lib-dir` | `./lib` | Folder containing `llama.dll` + dependencies |
| `--n-ctx` | `2048` | Context window size (scales KV-cache memory) |
| `--n-predict` | `128` | Maximum tokens to generate |
| `--verbose` | off | Stream the full native debug log to stdout |
| `--v`, `--version` | — | Print version and exit |
| `-h`, `--help` | — | Standard argparse help |

All relative paths resolve against the **script directory**, not the current working directory, so the tool behaves identically regardless of where it is launched from.

---

## 17. Troubleshooting

| Symptom | Cause | Resolution |
|---|---|---|
| `no backends are loaded` | Backend plugins not registered | Ensure `ggml.dll` and `ggml-cpu-*.dll` are in `lib/`; `_register_backends` must run before model load (it does, by design). |
| `'llama.dll' was not found` | Provisioning incomplete | Run `python carga.py`; confirm `lib/llama.dll` exists. |
| `Could not load 'llama.dll'` (OSError 193) | 32-bit vs 64-bit mismatch | Use 64-bit Python; `carga.py` fetches the x64 package. |
| Garbled non-ASCII output | UTF-8 fragmentation | Handled by incremental decoding (§11.5); ensure terminal/file encoding is UTF-8. |
| Repetitive output | Greedy decoding on a *base* model | Expected; add sampling (§18) or use an instruct-tuned model + chat template. |
| Pinned URL 404 | Build pruned from Releases | Automatic fallback to `releases/latest` selects a current CPU x64 asset. |

> **Note on output quality.** The shipped model is a *base* completion model, and decoding is *greedy*. Repetition (e.g., looping a sentence) is the mathematically expected behavior of `argmax` decoding on a small base model with no repetition penalty and no instruction tuning — not a bug in the engine.

---

## 18. Limitations and Future Work

- **Sampling strategies.** Only greedy decoding is implemented. Temperature scaling, top-k, top-p (nucleus), and repetition penalties can all be added *in Python* over the existing `logits[:n_vocab]` slice — no new C bindings required.
- **Chat templates.** The engine feeds the raw prompt. For instruct models, applying the GGUF-embedded chat template (BOS, `<|start_header_id|>…`) before tokenization would produce assistant-style replies.
- **Batched / multi-sequence decoding.** The context is created with a single sequence; the `llama_batch` struct already supports multi-sequence batching for higher throughput.
- **Digest verification.** Adding SHA-256 verification of downloads in `carga.py` would close the supply-chain gap noted in §14.
- **Cross-platform.** The DLL-loading and ABI reasoning are Windows-x64-specific; a `.so`/`.dylib` path with the System V AMD64 ABI would generalize the engine to Linux/macOS (the System V rules for large-struct passing differ in detail but the blob/padding strategy still applies).

---

## 19. Glossary

- **ABI** — Application Binary Interface: the machine-level contract (register usage, struct layout, calling convention) two compiled components must share.
- **FFI** — Foreign Function Interface: calling functions written in one language from another (`ctypes` is Python's FFI to C).
- **GGUF** — `llama.cpp`'s self-describing single-file model format.
- **GQA** — Grouped-Query Attention: query heads share a smaller number of key/value heads, shrinking the KV cache.
- **K-quant (Q4_K, Q6_K)** — `llama.cpp`'s block-wise, two-level quantization scheme.
- **KV cache** — stored keys/values of past tokens that make autoregressive decoding linear-time.
- **Logits** — the pre-softmax scores over the vocabulary produced by the model's output layer.
- **RoPE** — Rotary Position Embedding: relative positional encoding via vector rotation.
- **Prefill** — the initial forward pass that encodes the whole prompt at once.

---

## 20. References

- Vaswani et al., *Attention Is All You Need* (2017) — the transformer.
- Touvron et al., *LLaMA / Llama 2 / Llama 3* technical reports — architecture and training.
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021) — RoPE.
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models* (2023) — grouped-query attention.
- `ggml-org/llama.cpp` — source, GGUF spec, K-quant kernels, and the C API (`llama.h`, `ggml-backend.h`).
- Microsoft, *x64 calling convention* and *x64 ABI / aggregate return* documentation.
- CPython documentation, `ctypes` — *A foreign function library for Python*.

---

> **Disclaimer.** This software is provided for legitimate, authorized use. Loading native libraries grants them full process privileges; only run model and library binaries whose provenance you trust, and only employ this technique on systems where you are authorized to do so.
