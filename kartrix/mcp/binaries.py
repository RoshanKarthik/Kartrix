"""Download, verify and cache pinned binaries (MCP servers) — no Docker, no global installs.

The version and the SHA-256 of every release archive are pinned in Kartrix's own config
(``mcp_servers.json``), never taken from the download source at run time. An archive whose
hash differs is rejected before anything is extracted; only the one expected executable is
read out of it (no ``extractall``, so no path traversal). The verified executable lives in
the per-user cache: ``<cache>/bin/<name>/<version>/<executable>``.
"""

from __future__ import annotations

import hashlib
import io
import os
import platform
import stat
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from kartrix.observability.logger import get_logger
from kartrix.paths import user_cache_dir

logger = get_logger(__name__)

_MAX_ARCHIVE_BYTES = 150 * 1024 * 1024
_MAX_BINARY_BYTES = 300 * 1024 * 1024
_ATTEMPTS = 5  # some networks reset GitHub's download host intermittently
_TIMEOUT = 60


class BinaryError(Exception):
    """A pinned binary couldn't be installed. The message is safe to show the user."""


@dataclass(frozen=True)
class Asset:
    file: str
    sha256: str


@dataclass(frozen=True)
class BinarySpec:
    name: str
    version: str
    url: str  # template with {version} and {file}
    executable: str  # name inside the archive, without .exe
    assets: dict[str, Asset]  # platform key (see platform_key) → archive


def platform_key() -> str:
    system = platform.system().lower()  # windows | darwin | linux
    machine = platform.machine().lower()
    arch = {"amd64": "x86_64", "x86_64": "x86_64", "x64": "x86_64", "arm64": "arm64", "aarch64": "arm64"}.get(machine)
    return f"{system}-{arch or machine}"


def executable_name(spec: BinarySpec) -> str:
    return spec.executable + (".exe" if platform.system() == "Windows" else "")


def install_dir(spec: BinarySpec) -> Path:
    return user_cache_dir() / "bin" / spec.name / spec.version


def _download(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(_ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=_TIMEOUT) as response:  # noqa: S310 — https URL from pinned config
                data = response.read(_MAX_ARCHIVE_BYTES + 1)
            if len(data) > _MAX_ARCHIVE_BYTES:
                raise BinaryError(f"download is larger than {_MAX_ARCHIVE_BYTES // 2**20} MB: {url}")
            return data
        except BinaryError:
            raise
        except Exception as e:  # connection resets, timeouts, HTTP errors
            last = e
            logger.warning("Download failed, retrying", extra={"url": url, "attempt": attempt + 1, "error": str(e)})
            time.sleep(min(2**attempt, 10))
    raise BinaryError(f"could not download {url} after {_ATTEMPTS} attempts ({last})")


def _extract(archive: bytes, file: str, wanted: str) -> bytes:
    def check(size: int) -> None:
        if size > _MAX_BINARY_BYTES:
            raise BinaryError(f"{wanted} in the archive is unexpectedly large")

    if file.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            for info in zf.infolist():
                if not info.is_dir() and Path(info.filename).name == wanted:
                    check(info.file_size)
                    return zf.read(info)
    elif file.endswith((".tar.gz", ".tgz")):
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tf:
            for member in tf.getmembers():
                if member.isfile() and Path(member.name).name == wanted:
                    check(member.size)
                    handle = tf.extractfile(member)
                    if handle is not None:
                        return handle.read()
    else:
        raise BinaryError(f"unsupported archive type: {file}")
    raise BinaryError(f"{wanted} not found in {file}")


def ensure_binary(spec: BinarySpec) -> Path:
    """Path to the verified executable, downloading it on first use."""
    target = install_dir(spec) / executable_name(spec)
    if target.is_file():
        return target

    key = platform_key()
    asset = spec.assets.get(key)
    if asset is None:
        raise BinaryError(f"{spec.name} {spec.version} has no build for this platform ({key})")
    url = spec.url.format(version=spec.version, file=asset.file)
    logger.info("Downloading pinned binary", extra={"binary": spec.name, "version": spec.version, "url": url})
    archive = _download(url)
    digest = hashlib.sha256(archive).hexdigest()
    if digest != asset.sha256.lower():
        raise BinaryError(
            f"{asset.file} failed verification (SHA-256 {digest}, expected {asset.sha256}); not installed. "
            "The download may have been tampered with or corrupted — try again later."
        )
    binary = _extract(archive, asset.file, executable_name(spec))

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".download-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(binary)
        mode = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP
        os.chmod(tmp, mode)
        os.replace(tmp, target)  # atomic: a concurrent install of the same file is harmless
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    logger.info("Installed pinned binary", extra={"binary": spec.name, "version": spec.version, "path": str(target)})
    return target
