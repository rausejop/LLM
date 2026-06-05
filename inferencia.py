#!/usr/bin/env python3
"""LLM inference for Llama 3.2 via llama.cpp loaded dynamically with ctypes.

This module runs a GGUF model using the llama.cpp dynamic libraries loaded
into memory through ``ctypes``, without launching any executable and using
only the Python standard library. It targets a hardened Windows environment
where ``.exe`` files are blocked but ``.dll`` files may still be loaded.

Why ctypes:
  ``ctypes.CDLL`` loads ``llama.dll`` into the Python process and exposes its
  C API, which this module binds function by function.

Mapping the llama.cpp C API to Python:
  For each C function we declare its ``restype`` (return type) and ``argtypes``
  (argument types). The llama.cpp objects (model, context, vocabulary) are
  opaque pointers in C and are treated as ``ctypes.c_void_p`` here.

  The delicate part is structs passed *by value* (the ``*_default_params``
  functions and ``llama_batch``). Their layout changes across llama.cpp
  versions, so:
    * ``llama_model_params`` is treated as an opaque blob; defaults are used
      unchanged (a CPU build needs no field edits).
    * ``llama_context_params`` declares only its stable leading fields
      (``n_ctx``, ``n_threads``, ...) followed by generous trailing padding.
      On the Windows x64 ABI these large structs are passed by hidden pointer,
      so extra padding is harmless and guards against future layout changes.
    * ``llama_batch`` has a stable layout and is declared field by field.

Sampling:
  Greedy decoding (argmax over the logits) is implemented by reading the
  pointer returned by ``llama_get_logits_ith``, which avoids the more volatile
  sampler API.

Output:
  Verbose mode streams the full debug log (and the generated text) to stdout.
  The generated text is always written to the output file (``output.txt`` by
  default, configurable with ``--o``/``--output``).
"""

import argparse
import ctypes
import os
import sys
from pathlib import Path

__version__ = '1.0.0'


def _base_dir():
    """Returns the directory that anchors relative default paths.

    When running as a normal script this is the script's own directory. When
    running as a PyInstaller-frozen executable, ``__file__`` points inside the
    temporary extraction folder, so the executable's own directory is used
    instead (where the model, prompt, and lib/ are expected to sit).

    Returns:
      The base ``Path`` for resolving the model, prompt, and library folder.
    """
    if getattr(sys, 'frozen', False):  # Running inside a PyInstaller bundle.
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


# Project root (where the model and the lib/ folder created by carga.py live).
WORK_DIR = _base_dir()
DEFAULT_MODEL = 'llama-3.2-1b-q4_k_m.gguf'
DEFAULT_PROMPT = 'prompt.txt'
DEFAULT_OUTPUT = 'output.txt'
DEFAULT_LIB_DIR = WORK_DIR / 'lib'

# Type of the ggml/llama log callback:
#   void callback(enum ggml_log_level level, const char* text, void* user_data)
LOG_CALLBACK = ctypes.CFUNCTYPE(
    None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p
)

# Keeps installed callbacks alive. If Python garbage-collected them, the native
# library would call freed memory and crash.
_KEPT_ALIVE = []


class InferenceError(Exception):
    """A handled error (missing DLL, load failure, decode failure, ...)."""


# =============================================================================
# 1. ctypes structs that llama.cpp passes or returns by value
# =============================================================================


class LlamaModelParams(ctypes.Structure):
    """``struct llama_model_params`` treated as an opaque blob.

    Its fields are never accessed: defaults from ``llama_model_default_params``
    are passed unchanged to model loading. For a CPU build the defaults are
    correct (``n_gpu_layers == 0``). The generous size guarantees the value
    returned by the C API fits in the buffer.
    """

    _fields_ = [('_raw', ctypes.c_byte * 256)]


class LlamaContextParams(ctypes.Structure):
    """``struct llama_context_params``: stable header plus trailing padding.

    Only the leading fields are declared, since their order has been stable
    across many versions; this lets us safely set ``n_ctx`` and the thread
    counts. The trailing padding absorbs all remaining fields (rope, yarn,
    flags, ...) of any reasonable version. On the x64 ABI the struct is passed
    by pointer, so the extra bytes are harmless.
    """

    _fields_ = [
        ('n_ctx', ctypes.c_uint32),           # Context window size.
        ('n_batch', ctypes.c_uint32),         # Logical batch size.
        ('n_ubatch', ctypes.c_uint32),        # Physical micro-batch size.
        ('n_seq_max', ctypes.c_uint32),       # Max simultaneous sequences.
        ('n_threads', ctypes.c_int32),        # Threads for generation.
        ('n_threads_batch', ctypes.c_int32),  # Threads for prompt processing.
        ('_rest', ctypes.c_byte * 512),       # Padding for remaining fields.
    ]


class LlamaBatch(ctypes.Structure):
    """``struct llama_batch`` (stable layout).

    Returned by ``llama_batch_get_one`` and consumed by ``llama_decode``. The
    internal pointers are managed by llama.cpp.
    """

    _fields_ = [
        ('n_tokens', ctypes.c_int32),
        ('token', ctypes.POINTER(ctypes.c_int32)),
        ('embd', ctypes.POINTER(ctypes.c_float)),
        ('pos', ctypes.c_void_p),
        ('n_seq_id', ctypes.c_void_p),
        ('seq_id', ctypes.c_void_p),
        ('logits', ctypes.c_void_p),
    ]


# =============================================================================
# 2. DLL loading and C API binding
# =============================================================================


def _make_log_callback(verbose):
    """Creates the log callback for the C API based on verbosity.

    Args:
      verbose: If True, log text is written to stdout; otherwise discarded.

    Returns:
      A ``LOG_CALLBACK`` instance, also stored in ``_KEPT_ALIVE``.
    """

    def callback(level, text, user_data):
        del level, user_data  # Unused.
        if verbose and text:
            try:
                sys.stdout.write(text.decode('utf-8', 'replace'))
                sys.stdout.flush()
            except Exception:
                pass  # A logging error must never break inference.

    instance = LOG_CALLBACK(callback)
    _KEPT_ALIVE.append(instance)
    return instance


def _find_dll(lib_dir, name):
    """Finds ``name`` directly in ``lib_dir`` and then recursively.

    Args:
      lib_dir: Directory to search.
      name: DLL file name, e.g. ``'llama.dll'``.

    Returns:
      The path to the DLL, or None if not found.
    """
    direct = lib_dir / name
    if direct.exists():
        return direct
    for candidate in lib_dir.rglob(name):
        return candidate
    return None


def _install_log_callback(handles, log_callback):
    """Installs ``log_callback`` via ``ggml_log_set`` on the first matching DLL.

    Args:
      handles: Iterable of loaded ggml ``CDLL`` handles.
      log_callback: The ``LOG_CALLBACK`` instance to install.
    """
    for handle in handles:
        ggml_log_set = getattr(handle, 'ggml_log_set', None)
        if ggml_log_set is not None:
            ggml_log_set.restype = None
            ggml_log_set.argtypes = [LOG_CALLBACK, ctypes.c_void_p]
            ggml_log_set(log_callback, None)
            return


def _register_backends(handles, lib_dir, verbose):
    """Registers the ggml compute backends (required before model loading).

    In recent llama.cpp versions the backends (``ggml-cpu*.dll``) are plugins
    loaded at runtime; without this, model loading fails with
    "no backends are loaded". ``ggml_backend_load_all_from_path`` scans the
    given directory, loads the ``ggml-cpu-*.dll`` variants and selects the one
    best suited to this CPU. The symbol lives in ``ggml.dll``.

    Args:
      handles: Iterable of loaded ggml ``CDLL`` handles.
      lib_dir: Directory containing the backend DLLs.
      verbose: Whether to print a diagnostic line.

    Raises:
      InferenceError: If no backend-loading symbol is found.
    """
    for handle in handles:
        load_from_path = getattr(
            handle, 'ggml_backend_load_all_from_path', None
        )
        if load_from_path is not None:
            load_from_path.restype = None
            load_from_path.argtypes = [ctypes.c_char_p]
            load_from_path(str(lib_dir).encode('utf-8'))
            break
        load_all = getattr(handle, 'ggml_backend_load_all', None)
        if load_all is not None:
            load_all.restype = None
            load_all.argtypes = []
            load_all()
            break
    else:
        raise InferenceError(
            'Could not register any ggml compute backend '
            "(symbol 'ggml_backend_load_all' not found). The model cannot be "
            f'loaded. Check that ggml.dll is in {lib_dir}'
        )

    if verbose:
        print('[verbose] ggml backends registered')


def load_library(lib_dir, verbose):
    """Loads ``llama.dll`` into memory and wires up logging and backends.

    On Windows ``llama.dll`` depends on ``ggml*.dll``. To let the system
    resolver find them, the directory is registered with
    ``os.add_dll_directory`` and the ggml core DLLs are pre-loaded first.

    Args:
      lib_dir: Directory containing ``llama.dll`` and its dependencies.
      verbose: Whether to print diagnostics and route the native log to stdout.

    Returns:
      The loaded ``ctypes.CDLL`` for ``llama.dll``.

    Raises:
      InferenceError: If the directory, ``llama.dll``, or a backend is missing,
        or the library cannot be loaded.
    """
    if not lib_dir.exists():
        raise InferenceError(
            f'Library directory does not exist: {lib_dir}\n'
            "Run 'python carga.py' first to download the DLLs."
        )

    llama_path = _find_dll(lib_dir, 'llama.dll')
    if llama_path is None:
        raise InferenceError(
            f"'llama.dll' was not found in {lib_dir}.\n"
            "Run 'python carga.py' or inspect the lib/ folder."
        )

    real_dir = llama_path.parent
    if verbose:
        print(f'[verbose] llama.dll -> {llama_path}')

    # 1) Add the directory to the process DLL search path.
    try:
        os.add_dll_directory(str(real_dir))
    except (AttributeError, OSError):
        os.environ['PATH'] = (
            str(real_dir) + os.pathsep + os.environ.get('PATH', '')
        )

    # 2) Pre-load the ggml core. The handles expose the backend registry and
    #    the log setter used below. 'ggml_backend_load_all*' lives in ggml.dll
    #    (not ggml-base.dll), so we load both and search ggml.dll first.
    ggml_handles = []
    for dep in ('ggml-base.dll', 'ggml.dll'):
        dep_path = real_dir / dep
        if dep_path.exists():
            try:
                ggml_handles.append(ctypes.CDLL(str(dep_path)))
                if verbose:
                    print(f'[verbose] pre-loaded dependency: {dep}')
            except OSError as exc:
                if verbose:
                    print(f'[verbose] warning: could not pre-load {dep}: {exc}')

    # ggml.dll is typically the last handle, so search in reverse.
    search_order = list(reversed(ggml_handles))

    # 3) Install the log callback BEFORE registering backends, so the
    #    load_backend/repack messages are routed (or silenced) too.
    log_callback = _make_log_callback(verbose)
    _install_log_callback(search_order, log_callback)

    # 4) Register the compute backends.
    _register_backends(search_order, real_dir, verbose)

    # 5) Load the main library.
    try:
        lib = ctypes.CDLL(str(llama_path))
    except OSError as exc:
        raise InferenceError(
            f"Could not load 'llama.dll': {exc}\n"
            'Common causes: incompatible architecture (must be x64) or '
            'missing ggml*.dll dependencies in the same folder.'
        ) from exc

    # 6) Route llama.dll's own log through the same callback.
    llama_log_set = getattr(lib, 'llama_log_set', None)
    if llama_log_set is not None:
        llama_log_set.restype = None
        llama_log_set.argtypes = [LOG_CALLBACK, ctypes.c_void_p]
        llama_log_set(log_callback, None)

    return lib


def _bind(lib, name, restype, argtypes, *aliases):
    """Binds a C API function and sets its ``restype``/``argtypes``.

    Args:
      lib: The loaded ``ctypes.CDLL``.
      name: Primary exported symbol name.
      restype: ctypes return type.
      argtypes: List of ctypes argument types.
      *aliases: Alternative symbol names, to tolerate API renames across
        llama.cpp versions.

    Returns:
      The bound, ready-to-call function object.

    Raises:
      InferenceError: If neither ``name`` nor any alias is exported.
    """
    function = None
    for candidate in (name, *aliases):
        function = getattr(lib, candidate, None)
        if function is not None:
            break
    if function is None:
        raise InferenceError(
            f"The DLL does not export '{name}' (nor aliases {aliases}). "
            'This llama.cpp version may be incompatible with this script.'
        )
    function.restype = restype
    function.argtypes = argtypes
    return function


def bind_api(lib):
    """Declares every llama.cpp function used, with its C signature.

    Args:
      lib: The loaded ``ctypes.CDLL`` for ``llama.dll``.

    Returns:
      A simple namespace object with each function as an attribute.
    """
    ptr = ctypes.c_void_p  # Short alias for an opaque pointer.
    api = argparse.Namespace()

    # --- Backend lifecycle ---------------------------------------------------
    # void llama_backend_init(void);
    api.backend_init = _bind(lib, 'llama_backend_init', None, [])
    # void llama_backend_free(void);
    api.backend_free = _bind(lib, 'llama_backend_free', None, [])

    # --- Model ---------------------------------------------------------------
    # struct llama_model_params llama_model_default_params(void);
    api.model_default_params = _bind(
        lib, 'llama_model_default_params', LlamaModelParams, []
    )
    # llama_model* llama_model_load_from_file(const char*, llama_model_params);
    api.model_load = _bind(
        lib, 'llama_model_load_from_file', ptr,
        [ctypes.c_char_p, LlamaModelParams], 'llama_load_model_from_file'
    )
    # void llama_model_free(llama_model*);
    api.model_free = _bind(
        lib, 'llama_model_free', None, [ptr], 'llama_free_model'
    )
    # const llama_vocab* llama_model_get_vocab(const llama_model*);
    api.model_get_vocab = _bind(lib, 'llama_model_get_vocab', ptr, [ptr])

    # --- Context -------------------------------------------------------------
    # struct llama_context_params llama_context_default_params(void);
    api.context_default_params = _bind(
        lib, 'llama_context_default_params', LlamaContextParams, []
    )
    # llama_context* llama_init_from_model(llama_model*, llama_context_params);
    api.context_new = _bind(
        lib, 'llama_init_from_model', ptr, [ptr, LlamaContextParams],
        'llama_new_context_with_model'
    )
    # void llama_free(llama_context*);
    api.context_free = _bind(lib, 'llama_free', None, [ptr])
    # uint32_t llama_n_ctx(const llama_context*);
    api.n_ctx = _bind(lib, 'llama_n_ctx', ctypes.c_uint32, [ptr])

    # --- Vocabulary / tokenization ------------------------------------------
    # int32_t llama_vocab_n_tokens(const llama_vocab*);
    api.n_vocab = _bind(
        lib, 'llama_vocab_n_tokens', ctypes.c_int32, [ptr], 'llama_n_vocab'
    )
    # int32_t llama_tokenize(vocab, text, text_len, tokens, n_max,
    #                        add_special, parse_special);
    api.tokenize = _bind(
        lib, 'llama_tokenize', ctypes.c_int32,
        [ptr, ctypes.c_char_p, ctypes.c_int32,
         ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
         ctypes.c_bool, ctypes.c_bool]
    )
    # int32_t llama_token_to_piece(vocab, token, buf, length, lstrip, special);
    api.token_to_piece = _bind(
        lib, 'llama_token_to_piece', ctypes.c_int32,
        [ptr, ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32,
         ctypes.c_int32, ctypes.c_bool]
    )
    # bool llama_vocab_is_eog(const llama_vocab*, llama_token);
    api.is_eog = _bind(
        lib, 'llama_vocab_is_eog', ctypes.c_bool, [ptr, ctypes.c_int32],
        'llama_token_is_eog'
    )

    # --- Decoding (forward pass) --------------------------------------------
    # struct llama_batch llama_batch_get_one(llama_token*, int32_t n_tokens);
    api.batch_get_one = _bind(
        lib, 'llama_batch_get_one', LlamaBatch,
        [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
    )
    # int32_t llama_decode(llama_context*, llama_batch);
    api.decode = _bind(lib, 'llama_decode', ctypes.c_int32, [ptr, LlamaBatch])
    # float* llama_get_logits_ith(llama_context*, int32_t i);
    api.get_logits_ith = _bind(
        lib, 'llama_get_logits_ith', ctypes.POINTER(ctypes.c_float),
        [ptr, ctypes.c_int32]
    )

    return api


# =============================================================================
# 3. Inference logic
# =============================================================================


def tokenize_text(api, vocab, text, add_special):
    """Converts ``text`` into a list of token IDs using ``llama_tokenize``.

    Args:
      api: The bound API namespace.
      vocab: Opaque vocabulary pointer.
      text: Input text to tokenize.
      add_special: Whether to add special tokens (e.g. BOS).

    Returns:
      A list of integer token IDs.

    Raises:
      InferenceError: If tokenization fails.
    """
    data = text.encode('utf-8')
    n_max = len(data) + 16  # Generous upper bound on token count.
    buffer = (ctypes.c_int32 * n_max)()
    count = api.tokenize(vocab, data, len(data), buffer, n_max,
                         add_special, True)  # parse_special=True
    if count < 0:
        # The C API returns -(needed_count) when the buffer was too small.
        n_needed = -count
        buffer = (ctypes.c_int32 * n_needed)()
        count = api.tokenize(vocab, data, len(data), buffer, n_needed,
                             add_special, True)
        if count < 0:
            raise InferenceError('Failed to tokenize the prompt.')
    return list(buffer[:count])


def token_to_bytes(api, vocab, token):
    """Converts a token to its UTF-8 bytes (which may be partial).

    Args:
      api: The bound API namespace.
      vocab: Opaque vocabulary pointer.
      token: Token ID to convert.

    Returns:
      The token's raw UTF-8 bytes.
    """
    buffer = ctypes.create_string_buffer(256)
    count = api.token_to_piece(vocab, token, buffer, len(buffer), 0, False)
    if count < 0:  # Buffer too small: retry with the requested size.
        buffer = ctypes.create_string_buffer(-count)
        count = api.token_to_piece(vocab, token, buffer, len(buffer), 0, False)
    return buffer.raw[:max(count, 0)]


def run_inference(api, model_path, prompt, output_path, n_ctx, n_predict,
                  verbose):
    """Runs the minimal inference sequence and writes the result.

    Sequence: backend -> model -> context -> tokenize -> decode -> generate.
    The generation (prompt followed by the continuation) is written to
    ``output_path``. In verbose mode it is also streamed to stdout alongside
    the debug log.

    Args:
      api: The bound API namespace.
      model_path: Path to the GGUF model file.
      prompt: The prompt text.
      output_path: File to write the generated text to.
      n_ctx: Context window size.
      n_predict: Maximum number of tokens to generate.
      verbose: Whether to stream debug output to stdout.

    Raises:
      InferenceError: On any model, context, or decode failure.
    """
    api.backend_init()
    model = None
    context = None
    try:
        # --- Model loading (default params, opaque blob) --------------------
        model_params = api.model_default_params()
        if verbose:
            print(f'[verbose] loading model: {model_path}')
        model = api.model_load(str(model_path).encode('utf-8'), model_params)
        if not model:
            raise InferenceError(f'llama failed to load the model: {model_path}')

        vocab = api.model_get_vocab(model)
        n_vocab = api.n_vocab(vocab)

        # --- Context configuration and creation -----------------------------
        context_params = api.context_default_params()
        context_params.n_ctx = n_ctx
        context_params.n_threads = os.cpu_count() or 4
        context_params.n_threads_batch = context_params.n_threads
        context = api.context_new(model, context_params)
        if not context:
            raise InferenceError('Could not create the llama context.')

        if verbose:
            print(f'[verbose] n_ctx={api.n_ctx(context)}  n_vocab={n_vocab}  '
                  f'n_threads={context_params.n_threads}')

        # --- Prompt tokenization --------------------------------------------
        tokens = tokenize_text(api, vocab, prompt, add_special=True)
        if verbose:
            print(f'[verbose] prompt -> {len(tokens)} tokens')
        if not tokens:
            raise InferenceError('The prompt produced no tokens.')

        # --- Initial prompt decode ------------------------------------------
        token_array = (ctypes.c_int32 * len(tokens))(*tokens)
        batch = api.batch_get_one(token_array, len(tokens))
        if api.decode(context, batch) != 0:
            raise InferenceError('llama_decode failed (prompt).')

        # --- Generation loop (greedy sampling = argmax) ---------------------
        with open(output_path, 'w', encoding='utf-8') as out_file:
            def emit(text):
                """Writes text to the output file and, if verbose, to stdout."""
                out_file.write(text)
                out_file.flush()
                if verbose:
                    sys.stdout.write(text)
                    sys.stdout.flush()

            if verbose:
                print('\n----- RESPONSE -----')
            emit(prompt)

            pending = bytearray()  # UTF-8 bytes awaiting a complete character.
            n_generated = 0
            single = (ctypes.c_int32 * 1)()  # Reused 1-token buffer.

            for _ in range(n_predict):
                # Logits of the last token: pointer to n_vocab floats.
                logits_ptr = api.get_logits_ith(context, -1)
                if not logits_ptr:
                    raise InferenceError('get_logits_ith returned NULL.')
                logits = logits_ptr[:n_vocab]
                best = max(range(n_vocab), key=logits.__getitem__)

                if api.is_eog(vocab, best):
                    break  # End-of-generation token.

                # Emit text with safe incremental UTF-8 decoding.
                pending += token_to_bytes(api, vocab, best)
                try:
                    emit(pending.decode('utf-8'))
                    pending.clear()
                except UnicodeDecodeError:
                    pass  # Multibyte char split across tokens; wait for more.

                # Feed the generated token back for the next step.
                single[0] = best
                batch = api.batch_get_one(single, 1)
                if api.decode(context, batch) != 0:
                    raise InferenceError('llama_decode failed (generation).')
                n_generated += 1

            if pending:  # Leftover bytes (incomplete trailing character).
                emit(pending.decode('utf-8', 'replace'))

        if verbose:
            print('\n--------------------')
            print(f'[verbose] tokens generated: {n_generated}')
        print(f'Output written to {output_path}')

    finally:
        # Orderly release of native resources.
        if context:
            api.context_free(context)
        if model:
            api.model_free(model)
        api.backend_free()


# =============================================================================
# 4. CLI (argparse) and entry point
# =============================================================================


def build_parser():
    """Builds the command-line argument parser.

    Returns:
      A configured ``argparse.ArgumentParser``.
    """
    parser = argparse.ArgumentParser(
        prog='inferencia.py',
        description='Llama 3.2 inference via llama.cpp loaded with ctypes '
                    '(standard library only).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help='Path to the GGUF model.')
    parser.add_argument('--prompt', default=DEFAULT_PROMPT,
                        help='Path to the prompt text file.')
    parser.add_argument('--o', '--output', dest='output', default=DEFAULT_OUTPUT,
                        help='File to write the generated text to.')
    parser.add_argument('--lib-dir', default=str(DEFAULT_LIB_DIR),
                        help='Folder containing llama.dll and dependencies.')
    parser.add_argument('--n-ctx', type=int, default=2048,
                        help='Context window size.')
    parser.add_argument('--n-predict', type=int, default=128,
                        help='Maximum number of tokens to generate.')
    parser.add_argument('--verbose', action='store_true',
                        help='Stream maximum debug output to stdout.')
    # --v / --version: print the version and exit.
    parser.add_argument('--v', '--version', action='version',
                        version=f'%(prog)s {__version__}',
                        help='Show the script version and exit.')
    return parser


def resolve_path(value):
    """Resolves a path relative to the script directory.

    Args:
      value: A path string, absolute or relative.

    Returns:
      An absolute ``Path``.
    """
    path = Path(value)
    return path if path.is_absolute() else (WORK_DIR / path)


def main(argv=None):
    """Parses arguments, validates inputs, and runs inference.

    Args:
      argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
      Process exit code (0 on success, non-zero on failure).
    """
    args = build_parser().parse_args(argv)

    model_path = resolve_path(args.model)
    prompt_path = resolve_path(args.prompt)
    output_path = resolve_path(args.output)
    lib_dir = resolve_path(args.lib_dir)

    if args.verbose:
        print(f'[verbose] model : {model_path}')
        print(f'[verbose] prompt: {prompt_path}')
        print(f'[verbose] output: {output_path}')
        print(f'[verbose] libdir: {lib_dir}')

    try:
        if not model_path.exists():
            raise InferenceError(
                f'Model not found: {model_path}\n'
                "Run 'python carga.py' to download it."
            )
        if not prompt_path.exists():
            raise InferenceError(f'Prompt file not found: {prompt_path}')

        prompt = prompt_path.read_text(encoding='utf-8').strip()
        if not prompt:
            raise InferenceError('The prompt file is empty.')

        lib = load_library(lib_dir, args.verbose)
        api = bind_api(lib)
        run_inference(api, model_path, prompt, output_path,
                      n_ctx=args.n_ctx, n_predict=args.n_predict,
                      verbose=args.verbose)

    except InferenceError as exc:
        print(f'\n[ERROR] {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('\n[interrupted]', file=sys.stderr)
        return 130
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
