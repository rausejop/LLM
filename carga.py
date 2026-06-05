#!/usr/bin/env python3
"""Provisioning script for the ctypes-based llama.cpp inference system.

This module prepares everything that ``inferencia.py`` needs to run. It uses
only the Python standard library, and it never launches an executable, which
makes it suitable for a hardened Windows environment where ``.exe`` files are
blocked but dynamic libraries (``.dll``) may still be loaded into memory.

Responsibilities:
  1. Download the GGUF model (Llama 3.2 1B, Q4_K_M quantization).
  2. Download the llama.cpp Windows x64 (CPU) dynamic libraries as a ``.zip``
     and extract the ``.dll`` files. The CPU build is self-contained and needs
     no GPU drivers or extra runtimes.
  3. Lay out the working directory so ``inferencia.py`` finds both the model
     and the libraries without extra configuration.

Resulting layout (relative to this script):
    <dir>/
      |- carga.py
      |- inferencia.py
      |- llama-3.2-1b-q4_k_m.gguf     # model
      `- lib/
           |- llama.dll               # main C API
           |- ggml.dll, ggml-base.dll, ggml-cpu*.dll   # compute backend
           `- ...                     # remaining bundled DLLs
"""

import json
import re
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# Model: Llama 3.2 1B quantized to Q4_K_M, in GGUF format.
MODEL_URL = (
    'https://huggingface.co/medmekk/Llama-3.2-1B-Q4_K_M-GGUF/'
    'resolve/main/llama-3.2-1b-q4_k_m.gguf'
)
MODEL_FILENAME = 'llama-3.2-1b-q4_k_m.gguf'

# llama.cpp libraries.
# Primary source: a pinned, verified CPU x64 Windows build. It is downloaded
# directly, without querying any API.
LLAMA_ZIP_URL = (
    'https://github.com/ggml-org/llama.cpp/releases/download/'
    'b9518/llama-b9518-bin-win-cpu-x64.zip'
)
LLAMA_ZIP_FILENAME = 'llama-b9518-bin-win-cpu-x64.zip'

# Fallback: if the pinned URL fails (e.g. the build was removed), resolve the
# latest release through the public GitHub API and pick the CPU x64 package.
GITHUB_LATEST_RELEASE_API = (
    'https://api.github.com/repos/ggml-org/llama.cpp/releases/latest'
)

# Windows x64 ``.zip`` asset selection patterns, in order of preference. The
# generic CPU build is preferred for maximum compatibility.
ASSET_PREFERRED_PATTERNS = (
    re.compile(r'bin-win-cpu-x64\.zip$', re.IGNORECASE),
    re.compile(r'bin-win-avx2-x64\.zip$', re.IGNORECASE),
    re.compile(r'bin-win-avx-x64\.zip$', re.IGNORECASE),
    re.compile(r'bin-win-noavx-x64\.zip$', re.IGNORECASE),
)

# Hardware/toolkit-specific builds are excluded: they would require drivers or
# runtimes that are not guaranteed in the hardened environment.
ASSET_EXCLUDE_PATTERN = re.compile(
    r'(cuda|cu1[12]|vulkan|hip|sycl|kompute|musa|arm64|cpu-arm)',
    re.IGNORECASE,
)

# GitHub rejects API requests that do not send a User-Agent header.
HTTP_HEADERS = {'User-Agent': 'carga.py/1.0 (+llama.cpp bootstrap)'}

DOWNLOAD_CHUNK_BYTES = 1 << 20  # 1 MiB per download block.

WORK_DIR = Path(__file__).resolve().parent
LIB_DIR = WORK_DIR / 'lib'


def _open_url(url):
    """Opens a URL, following redirects (HF and GitHub redirect to a CDN).

    Args:
      url: The absolute URL to open.

    Returns:
      An open ``http.client.HTTPResponse`` object.
    """
    request = urllib.request.Request(url, headers=HTTP_HEADERS)
    return urllib.request.urlopen(request, timeout=60)


def fetch_json(url):
    """Downloads and parses a JSON document (used for the GitHub API).

    Args:
      url: The absolute URL returning a JSON body.

    Returns:
      The parsed JSON as Python objects.
    """
    with _open_url(url) as response:
        return json.loads(response.read().decode('utf-8'))


def _format_size(num_bytes):
    """Formats a byte count as a human-readable string.

    Args:
      num_bytes: A size in bytes.

    Returns:
      A short string such as ``'12.3 MiB'``.
    """
    size = float(num_bytes)
    for unit in ('B', 'KiB', 'MiB', 'GiB'):
        if size < 1024 or unit == 'GiB':
            return f'{size:.1f} {unit}'
        size /= 1024
    return f'{size:.1f} GiB'


def download_file(url, dest, description):
    """Streams ``url`` into ``dest``, printing progress.

    The download is idempotent: if ``dest`` already exists with a non-zero
    size it is skipped, so re-running the script does not re-download large
    files. The data is written to a ``.part`` file and atomically renamed on
    success.

    Args:
      url: The absolute URL to download.
      dest: Destination path for the downloaded file.
      description: Human-readable label shown in progress output.
    """
    if dest.exists() and dest.stat().st_size > 0:
        size = _format_size(dest.stat().st_size)
        print(f'[=] {description}: already present ({size}), skipping.')
        return

    print(f'[>] Downloading {description}\n    {url}')
    tmp = dest.with_suffix(dest.suffix + '.part')
    try:
        with _open_url(url) as response:
            total = int(response.headers.get('Content-Length', 0))
            done = 0
            with open(tmp, 'wb') as out_file:
                while True:
                    block = response.read(DOWNLOAD_CHUNK_BYTES)
                    if not block:
                        break
                    out_file.write(block)
                    done += len(block)
                    if total:
                        pct = done * 100 / total
                        sys.stdout.write(
                            f'\r    {pct:5.1f}%  '
                            f'{_format_size(done)} / {_format_size(total)}'
                        )
                    else:
                        sys.stdout.write(f'\r    {_format_size(done)}')
                    sys.stdout.flush()
        sys.stdout.write('\n')
        tmp.replace(dest)  # Atomic rename once the download completed.
        print(f'[ok] Saved to {dest}')
    except Exception:
        # Remove the incomplete ``.part`` file so it is not mistaken for a
        # finished download on the next run.
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def find_llama_asset():
    """Resolves the best Windows x64 llama.cpp asset from the latest release.

    Returns:
      A ``(name, url)`` tuple for the selected ``.zip`` asset.

    Raises:
      RuntimeError: If no suitable Windows x64 asset is found.
    """
    print('[>] Resolving latest llama.cpp release (GitHub API)...')
    release = fetch_json(GITHUB_LATEST_RELEASE_API)
    tag = release.get('tag_name', 'unknown')
    assets = release.get('assets', [])
    print(f'    Release: {tag}  ({len(assets)} assets)')

    def pick(predicate):
        for asset in assets:
            name = asset['name']
            if ASSET_EXCLUDE_PATTERN.search(name):
                continue
            if predicate(name):
                return name, asset['browser_download_url']
        return None

    # 1) Preferred patterns, in order.
    for pattern in ASSET_PREFERRED_PATTERNS:
        hit = pick(lambda name, p=pattern: bool(p.search(name)))
        if hit:
            return hit
    # 2) Any non-excluded Windows x64 zip.
    hit = pick(lambda name: bool(re.search(r'win.*x64\.zip$', name, re.I)))
    if hit:
        return hit

    raise RuntimeError(
        'No suitable Windows x64 asset found in the latest release.\n'
        f'Available assets: {[a["name"] for a in assets]}'
    )


def extract_dlls(zip_path, lib_dir):
    """Extracts and flattens the ``.dll`` files from a package into ``lib_dir``.

    The archive may contain subdirectories; only ``.dll`` files are kept and
    they are placed directly in ``lib_dir``, which is where ``inferencia.py``
    looks for them.

    Args:
      zip_path: Path to the downloaded ``.zip`` archive.
      lib_dir: Directory into which the DLLs are extracted (created if needed).

    Raises:
      RuntimeError: If the archive contains no DLLs or lacks ``llama.dll``.
    """
    lib_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if member.endswith('/') or not member.lower().endswith('.dll'):
                continue
            target = lib_dir / Path(member).name  # Flatten to basename.
            with archive.open(member) as src, open(target, 'wb') as dst:
                dst.write(src.read())
            count += 1

    if count == 0:
        raise RuntimeError(f'Package {zip_path.name} contained no DLLs.')
    print(f'[ok] Extracted {count} DLLs into {lib_dir}')

    if not (lib_dir / 'llama.dll').exists():
        raise RuntimeError(
            "'llama.dll' was not found after extraction; the package may use "
            f'a different DLL name. Inspect the contents of {lib_dir}'
        )


def main():
    """Downloads the model and libraries, then prepares the working directory.

    Returns:
      Process exit code (0 on success, 1 on a handled failure).
    """
    print('=' * 70)
    print(' Provisioning the ctypes-based llama.cpp inference system')
    print('=' * 70)
    print(f'Working directory: {WORK_DIR}\n')

    try:
        # 1) GGUF model -> project root.
        download_file(
            MODEL_URL, WORK_DIR / MODEL_FILENAME, 'GGUF model (Llama 3.2 1B)'
        )

        # 2) llama.cpp DLLs -> lib/
        if (LIB_DIR / 'llama.dll').exists():
            print("[=] Libraries: 'lib/llama.dll' already present, skipping.")
        else:
            # Primary source: the pinned build (b9518). On HTTP failure, fall
            # back to resolving the latest release via the GitHub API.
            try:
                asset_name, asset_url = LLAMA_ZIP_FILENAME, LLAMA_ZIP_URL
                zip_path = WORK_DIR / asset_name
                download_file(
                    asset_url, zip_path, f'llama.cpp libraries ({asset_name})'
                )
            except urllib.error.HTTPError as exc:
                print(f'[!] Pinned URL failed ({exc.code}); '
                      'resolving the latest release instead...')
                asset_name, asset_url = find_llama_asset()
                zip_path = WORK_DIR / asset_name
                download_file(
                    asset_url, zip_path, f'llama.cpp libraries ({asset_name})'
                )

            extract_dlls(zip_path, LIB_DIR)
            zip_path.unlink(missing_ok=True)  # No longer needed once extracted.

    except urllib.error.HTTPError as exc:
        print(f'\n[HTTP ERROR] {exc.code} {exc.reason} while fetching a URL.',
              file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f'\n[NETWORK ERROR] {exc.reason}', file=sys.stderr)
        return 1
    except Exception as exc:  # Report any other failure clearly to the user.
        print(f'\n[ERROR] {exc}', file=sys.stderr)
        return 1

    print('\n' + '=' * 70)
    print(' Done. You can now run inference:')
    print('   python inferencia.py --prompt prompt.txt --verbose')
    print('=' * 70)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
