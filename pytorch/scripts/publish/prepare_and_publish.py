#!/usr/bin/env python3
"""
Republish local org.bytedeco SNAPSHOT jars as io.github.mullerhai *beta-01*
artifacts to Maven Central (Sonatype Central Portal).

Pipeline:
  stage  -> rewrite GAV + POM metadata from local ~/.m2 SNAPSHOTs
  sign   -> GPG detach-sign + md5/sha1/sha256/sha512 checksums
  bundle -> one ZIP per artifact (or one combined ZIP)
  upload -> Central Portal Publisher API (optional --upload)
  install-local -> also install rewritten GAVs into local m2

Usage:
  python3 prepare_and_publish.py stage
  python3 prepare_and_publish.py sign
  python3 prepare_and_publish.py bundle
  python3 prepare_and_publish.py upload          # needs CENTRAL_USERNAME/PASSWORD
  python3 prepare_and_publish.py all             # stage+sign+bundle
  python3 prepare_and_publish.py all --upload    # full publish
  python3 prepare_and_publish.py install-local   # stage + install to ~/.m2
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET
import ssl
# 创建不校验SSL的上下文
ssl_ctx = ssl._create_unverified_context()
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STAGE = Path(os.environ.get("STAGE_DIR", SCRIPT_DIR / "staging"))
DEFAULT_BUNDLE = Path(os.environ.get("BUNDLE_DIR", SCRIPT_DIR / "bundles"))
M2_BYTEDECO = Path(os.environ.get("M2_BYTEDECO", Path.home() / ".m2/repository/org/bytedeco"))
M2_LOCAL = Path.home() / ".m2/repository"

GROUP_ID = os.environ.get("PUBLISH_GROUP_ID", "io.github.mullerhai")
GROUP_PATH = GROUP_ID.replace(".", "/")
OLD_GROUP = "org.bytedeco"
SUFFIX = os.environ.get("PUBLISH_SUFFIX", "beta-07")
# When re-releasing platform POMs only (property expansion fix), set
#   PUBLISH_SUFFIX=beta-08 NATIVE_SUFFIX=beta-07
# so aggregator GAVs are new while classifier jars keep the already-published native versions.
NATIVE_SUFFIX = os.environ.get("NATIVE_SUFFIX", SUFFIX)
JAVACPP_BASE = os.environ.get("JAVACPP_BASE", "1.5.14")
OPENBLAS_LIB = os.environ.get("OPENBLAS_LIB", "0.3.33")
CUDA_LIB = os.environ.get("CUDA_LIB", "13.3-9.24")
PYTORCH_LIB = os.environ.get("PYTORCH_LIB", "2.13.0")
FFMPEG_LIB = os.environ.get("FFMPEG_LIB", "8.1.1")
OPENCV_LIB = os.environ.get("OPENCV_LIB", "4.13.0")
CPYTHON_LIB = os.environ.get("CPYTHON_LIB", "3.14.6")
NUMPY_LIB = os.environ.get("NUMPY_LIB", "2.5.0")
TENSORRT_LIB = os.environ.get("TENSORRT_LIB", "11.1")
TRITONSERVER_LIB = os.environ.get("TRITONSERVER_LIB", "2.70.0")

JAVACPP_VERSION = os.environ.get("JAVACPP_VERSION", f"{JAVACPP_BASE}-{SUFFIX}")
OPENBLAS_VERSION = os.environ.get("OPENBLAS_VERSION", f"{OPENBLAS_LIB}-{JAVACPP_BASE}-{SUFFIX}")
CUDA_VERSION = os.environ.get("CUDA_VERSION", f"{CUDA_LIB}-{JAVACPP_BASE}-{SUFFIX}")
PYTORCH_VERSION = os.environ.get("PYTORCH_VERSION", f"{PYTORCH_LIB}-{JAVACPP_BASE}-{SUFFIX}")
FFMPEG_VERSION = os.environ.get("FFMPEG_VERSION", f"{FFMPEG_LIB}-{JAVACPP_BASE}-{SUFFIX}")
OPENCV_VERSION = os.environ.get("OPENCV_VERSION", f"{OPENCV_LIB}-{JAVACPP_BASE}-{SUFFIX}")
CPYTHON_VERSION = os.environ.get("CPYTHON_VERSION", f"{CPYTHON_LIB}-{JAVACPP_BASE}-{SUFFIX}")
NUMPY_VERSION = os.environ.get("NUMPY_VERSION", f"{NUMPY_LIB}-{JAVACPP_BASE}-{SUFFIX}")
TENSORRT_VERSION = os.environ.get("TENSORRT_VERSION", f"{TENSORRT_LIB}-{JAVACPP_BASE}-{SUFFIX}")
TRITONSERVER_VERSION = os.environ.get("TRITONSERVER_VERSION", f"{TRITONSERVER_LIB}-{JAVACPP_BASE}-{SUFFIX}")

# Native (non-platform) coordinates — may differ from platform re-release suffix
JAVACPP_NATIVE_VERSION = os.environ.get("JAVACPP_NATIVE_VERSION", f"{JAVACPP_BASE}-{NATIVE_SUFFIX}")
OPENBLAS_NATIVE_VERSION = os.environ.get("OPENBLAS_NATIVE_VERSION", f"{OPENBLAS_LIB}-{JAVACPP_BASE}-{NATIVE_SUFFIX}")
CUDA_NATIVE_VERSION = os.environ.get("CUDA_NATIVE_VERSION", f"{CUDA_LIB}-{JAVACPP_BASE}-{NATIVE_SUFFIX}")
PYTORCH_NATIVE_VERSION = os.environ.get("PYTORCH_NATIVE_VERSION", f"{PYTORCH_LIB}-{JAVACPP_BASE}-{NATIVE_SUFFIX}")
FFMPEG_NATIVE_VERSION = os.environ.get("FFMPEG_NATIVE_VERSION", f"{FFMPEG_LIB}-{JAVACPP_BASE}-{NATIVE_SUFFIX}")
OPENCV_NATIVE_VERSION = os.environ.get("OPENCV_NATIVE_VERSION", f"{OPENCV_LIB}-{JAVACPP_BASE}-{NATIVE_SUFFIX}")
CPYTHON_NATIVE_VERSION = os.environ.get("CPYTHON_NATIVE_VERSION", f"{CPYTHON_LIB}-{JAVACPP_BASE}-{NATIVE_SUFFIX}")
NUMPY_NATIVE_VERSION = os.environ.get("NUMPY_NATIVE_VERSION", f"{NUMPY_LIB}-{JAVACPP_BASE}-{NATIVE_SUFFIX}")
TENSORRT_NATIVE_VERSION = os.environ.get("TENSORRT_NATIVE_VERSION", f"{TENSORRT_LIB}-{JAVACPP_BASE}-{NATIVE_SUFFIX}")
TRITONSERVER_NATIVE_VERSION = os.environ.get("TRITONSERVER_NATIVE_VERSION", f"{TRITONSERVER_LIB}-{JAVACPP_BASE}-{NATIVE_SUFFIX}")

GPG_KEY_ID = os.environ.get("GPG_KEY_ID", "7AD293084072FD9F")

ORG_NAME = os.environ.get("PUBLISH_ORG_NAME", "mullerhai")
DEV_ID = os.environ.get("PUBLISH_DEVELOPER_ID", "mullerhai")
DEV_NAME = os.environ.get("PUBLISH_DEVELOPER_NAME", "muller")
DEV_EMAIL = os.environ.get("PUBLISH_DEVELOPER_EMAIL", "hai710459649@foxmail.com")
DEV_URL = os.environ.get("PUBLISH_DEVELOPER_URL", "https://github.com/mullerhai")
SCM_URL = os.environ.get("PUBLISH_SCM_URL", "https://github.com/mullerhai/javacpp-presets")
SCM_CONN = os.environ.get("PUBLISH_SCM_CONNECTION", "scm:git:git://github.com/mullerhai/javacpp-presets.git")
SCM_DEV = os.environ.get(
    "PUBLISH_SCM_DEV_CONNECTION", "scm:git:ssh://git@github.com/mullerhai/javacpp-presets.git"
)
PROJECT_URL = os.environ.get("PUBLISH_PROJECT_URL", "https://github.com/mullerhai/javacpp-presets")
LICENSE_NAME = os.environ.get("PUBLISH_LICENSE_NAME", "Apache License, Version 2.0")
LICENSE_URL = os.environ.get("PUBLISH_LICENSE_URL", "https://www.apache.org/licenses/LICENSE-2.0")

CENTRAL_UPLOAD = "https://central.sonatype.com/api/v1/publisher/upload"
CENTRAL_STATUS = "https://central.sonatype.com/api/v1/publisher/status"
CENTRAL_PUBLISH = "https://central.sonatype.com/api/v1/publisher/deployment"

# Latest stable bytedeco on Maven Central (we don't publish these yet)
CUDA_STABLE = "12.6-9.5-1.5.11"
VERSION_MAP = {
    f"{JAVACPP_BASE}-SNAPSHOT": JAVACPP_VERSION,
    f"{OPENBLAS_LIB}-{JAVACPP_BASE}-SNAPSHOT": OPENBLAS_VERSION,
    f"{CUDA_LIB}-{JAVACPP_BASE}-SNAPSHOT": CUDA_VERSION,
    f"{PYTORCH_LIB}-{JAVACPP_BASE}-SNAPSHOT": PYTORCH_VERSION,
    f"{FFMPEG_LIB}-{JAVACPP_BASE}-SNAPSHOT": FFMPEG_VERSION,
    f"{OPENCV_LIB}-{JAVACPP_BASE}-SNAPSHOT": OPENCV_VERSION,
    f"{CPYTHON_LIB}-{JAVACPP_BASE}-SNAPSHOT": CPYTHON_VERSION,
    f"{NUMPY_LIB}-{JAVACPP_BASE}-SNAPSHOT": NUMPY_VERSION,
    f"{TENSORRT_LIB}-{JAVACPP_BASE}-SNAPSHOT": TENSORRT_VERSION,
    f"{TRITONSERVER_LIB}-{JAVACPP_BASE}-SNAPSHOT": TRITONSERVER_VERSION,
    # unresolved parent property forms that may appear in raw source poms
    f"{OPENBLAS_LIB}-${{project.parent.version}}": OPENBLAS_VERSION,
    f"{CUDA_LIB}-${{project.parent.version}}": CUDA_VERSION,
    f"{PYTORCH_LIB}-${{project.parent.version}}": PYTORCH_VERSION,
    f"{FFMPEG_LIB}-${{project.parent.version}}": FFMPEG_VERSION,
    f"{OPENCV_LIB}-${{project.parent.version}}": OPENCV_VERSION,
    f"{CPYTHON_LIB}-${{project.parent.version}}": CPYTHON_VERSION,
    f"{NUMPY_LIB}-${{project.parent.version}}": NUMPY_VERSION,
    f"{TENSORRT_LIB}-${{project.parent.version}}": TENSORRT_VERSION,
    f"{TRITONSERVER_LIB}-${{project.parent.version}}": TRITONSERVER_VERSION,
    "${project.parent.version}": JAVACPP_VERSION,
}

PUBLISHED_ARTIFACTS = {
    "javacpp",
    "javacpp-platform",
    "openblas",
    "openblas-platform",
    "cuda",
    "cuda-platform",
    "cuda-redist",
    "cuda-redist-cublas",
    "cuda-redist-cudnn",
    "cuda-redist-cusolver",
    "cuda-redist-cusparse",
    "cuda-redist-npp",
    "cuda-redist-nccl",
    "cuda-redist-nvcomp",
    "cuda-platform-redist",
    "cuda-platform-redist-cublas",
    "cuda-platform-redist-cudnn",
    "cuda-platform-redist-cusolver",
    "cuda-platform-redist-cusparse",
    "cuda-platform-redist-npp",
    "cuda-platform-redist-nccl",
    "cuda-platform-redist-nvcomp",
    "ffmpeg",
    "ffmpeg-platform",
    "opencv",
    "opencv-platform",
    "cpython",
    "cpython-platform",
    "numpy",
    "numpy-platform",
    "pytorch",
    "pytorch-platform",
    "pytorch-platform-gpu",
    "tensorrt",
    "tensorrt-platform",
    "tritonserver",
    "tritonserver-platform",
}

# CUDA redistributable native packages (NVIDIA libs packaged by JavaCPP)
CUDA_REDIST_IDS = [
    "cuda-redist",
    "cuda-redist-cublas",
    "cuda-redist-cudnn",
    "cuda-redist-cusolver",
    "cuda-redist-cusparse",
    "cuda-redist-npp",
    "cuda-redist-nccl",
    "cuda-redist-nvcomp",
]
CUDA_PLATFORM_REDIST_IDS = [
    "cuda-platform-redist",
    "cuda-platform-redist-cublas",
    "cuda-platform-redist-cudnn",
    "cuda-platform-redist-cusolver",
    "cuda-platform-redist-cusparse",
    "cuda-platform-redist-npp",
    "cuda-platform-redist-nccl",
    "cuda-platform-redist-nvcomp",
]
CUDA_REDIST_CLASSIFIERS = [
    "linux-arm64",
    "linux-x86_64",
    "windows-x86_64",
]


@dataclass
class Artifact:
    artifact_id: str
    old_version: str  # SNAPSHOT version in local m2
    new_version: str
    description: str
    # classifiers to include if present (None main jar always included)
    classifiers: list[str] = field(default_factory=list)
    require_javadoc: bool = True
    require_sources: bool = True
    packaging: str = "jar"


ARTIFACTS: list[Artifact] = [
    Artifact(
        artifact_id="javacpp",
        old_version=f"{JAVACPP_BASE}-SNAPSHOT",
        new_version=JAVACPP_VERSION,
        description="The missing bridge between Java and native C++ (mullerhai fork release)",
        classifiers=[
            "android-arm64",
            "android-x86_64",
            "ios-arm64",
            "ios-x86_64",
            "linux-arm64",
            "linux-ppc64le",
            "linux-riscv64",
            "linux-x86_64",
            "macosx-arm64",
            "macosx-x86_64",
            "windows-arm64",
            "windows-x86_64",
        ],
    ),
    Artifact(
        artifact_id="javacpp-platform",
        old_version=f"{JAVACPP_BASE}-SNAPSHOT",
        new_version=JAVACPP_VERSION,
        description="JavaCPP Platform aggregator (mullerhai fork release)",
        classifiers=[],
        require_javadoc=False,
        require_sources=True,
    ),
    Artifact(
        artifact_id="openblas",
        old_version=f"{OPENBLAS_LIB}-{JAVACPP_BASE}-SNAPSHOT",
        new_version=OPENBLAS_VERSION,
        description="JavaCPP Presets for OpenBLAS (mullerhai fork release)",
        classifiers=[
            "android-arm64",
            "android-x86_64",
            "ios-arm64",
            "ios-x86_64",
            "linux-arm64",
            "linux-x86_64",
            "macosx-arm64",
            "macosx-x86_64",
            "windows-x86_64",
        ],
    ),
    Artifact(
        artifact_id="openblas-platform",
        old_version=f"{OPENBLAS_LIB}-{JAVACPP_BASE}-SNAPSHOT",
        new_version=OPENBLAS_VERSION,
        description="JavaCPP Presets Platform for OpenBLAS (mullerhai fork release)",
        classifiers=[],
        require_javadoc=False,
    ),
    Artifact(
        artifact_id="cuda",
        old_version=f"{CUDA_LIB}-{JAVACPP_BASE}-SNAPSHOT",
        new_version=CUDA_VERSION,
        description="JavaCPP Presets for CUDA (mullerhai fork release)",
        classifiers=[
            "linux-arm64",
            "linux-x86_64",
            "windows-x86_64",
        ],
    ),
    Artifact(
        artifact_id="cuda-platform",
        old_version=f"{CUDA_LIB}-{JAVACPP_BASE}-SNAPSHOT",
        new_version=CUDA_VERSION,
        description="JavaCPP Presets Platform for CUDA (mullerhai fork release)",
        classifiers=[],
        require_javadoc=False,
        require_sources=True,
    ),
    Artifact(
        artifact_id="ffmpeg",
        old_version=f"{FFMPEG_LIB}-{JAVACPP_BASE}-SNAPSHOT",
        new_version=FFMPEG_VERSION,
        description="JavaCPP Presets for FFmpeg (mullerhai fork release)",
        classifiers=[
            "android-arm64",
            "android-x86_64",
            "ios-arm64",
            "ios-x86_64",
            "linux-arm64",
            "linux-x86_64",
            "macosx-arm64",
            "macosx-x86_64",
            "windows-x86_64",
        ],
    ),
    Artifact(
        artifact_id="ffmpeg-platform",
        old_version=f"{FFMPEG_LIB}-{JAVACPP_BASE}-SNAPSHOT",
        new_version=FFMPEG_VERSION,
        description="JavaCPP Presets Platform for FFmpeg (mullerhai fork release)",
        classifiers=[],
        require_javadoc=False,
    ),
    Artifact(
        artifact_id="opencv",
        old_version=f"{OPENCV_LIB}-{JAVACPP_BASE}-SNAPSHOT",
        new_version=OPENCV_VERSION,
        description="JavaCPP Presets for OpenCV (mullerhai fork release)",
        classifiers=[
            "android-arm64",
            "android-x86_64",
            "ios-arm64",
            "ios-x86_64",
            "linux-arm64",
            "linux-x86_64",
            "macosx-arm64",
            "macosx-x86_64",
            "windows-x86_64",
        ],
    ),
    Artifact(
        artifact_id="opencv-platform",
        old_version=f"{OPENCV_LIB}-{JAVACPP_BASE}-SNAPSHOT",
        new_version=OPENCV_VERSION,
        description="JavaCPP Presets Platform for OpenCV (mullerhai fork release)",
        classifiers=[],
        require_javadoc=False,
    ),
    Artifact(
        artifact_id="cpython",
        old_version=f"{CPYTHON_LIB}-{JAVACPP_BASE}-SNAPSHOT",
        new_version=CPYTHON_VERSION,
        description="JavaCPP Presets for CPython (mullerhai fork release)",
        classifiers=[
            "android-arm64",
            "android-x86_64",
            "ios-arm64",
            "ios-x86_64",
            "linux-arm64",
            "linux-x86_64",
            "macosx-arm64",
            "macosx-x86_64",
            "windows-x86_64",
        ],
    ),
    Artifact(
        artifact_id="cpython-platform",
        old_version=f"{CPYTHON_LIB}-{JAVACPP_BASE}-SNAPSHOT",
        new_version=CPYTHON_VERSION,
        description="JavaCpp Presets Platform for CPython (mullerhai fork release)",
        classifiers=[],
        require_javadoc=False,
    ),
    Artifact(
        artifact_id="numpy",
        old_version=f"{NUMPY_LIB}-{JAVACPP_BASE}-SNAPSHOT",
        new_version=NUMPY_VERSION,
        description="JavaCpp Presets for NumPy (mullerhai fork release)",
        classifiers=[
            "android-arm64",
            "android-x86_64",
            "ios-arm64",
            "ios-x86_64",
            "linux-arm64",
            "linux-x86_64",
            "macosx-arm64",
            "macosx-x86_64",
            "windows-x86_64",
        ],
    ),
    Artifact(
        artifact_id="numpy-platform",
        old_version=f"{NUMPY_LIB}-{JAVACPP_BASE}-SNAPSHOT",
        new_version=NUMPY_VERSION,
        description="JavaCpp Presets Platform for NumPy (mullerhai fork release)",
        classifiers=[],
        require_javadoc=False,
    ),
    Artifact(
        artifact_id="pytorch",
        old_version=f"{PYTORCH_LIB}-{JAVACPP_BASE}-SNAPSHOT",
        new_version=PYTORCH_VERSION,
        description="JavaCPP Presets for PyTorch (mullerhai fork release)",
        classifiers=[
            "macosx-arm64",
            "macosx-x86_64",
            "linux-x86_64",
            "linux-arm64",
            "windows-x86_64",
            # GPU extension classifiers (built with -Djavacpp.platform.extension=-gpu)
            "linux-x86_64-gpu",
            "linux-arm64-gpu",
            "windows-x86_64-gpu",
        ],
    ),
    Artifact(
        artifact_id="pytorch-platform",
        old_version=f"{PYTORCH_LIB}-{JAVACPP_BASE}-SNAPSHOT",
        new_version=PYTORCH_VERSION,
        description="JavaCPP Presets Platform for PyTorch (mullerhai fork release)",
        classifiers=[],
        require_javadoc=False,
        require_sources=True,
    ),
    Artifact(
        artifact_id="pytorch-platform-gpu",
        old_version=f"{PYTORCH_LIB}-{JAVACPP_BASE}-SNAPSHOT",
        new_version=PYTORCH_VERSION,
        description="JavaCPP Presets Platform GPU for PyTorch (mullerhai fork release)",
        classifiers=[],
        require_javadoc=False,
        require_sources=True,
    ),
    Artifact(
        artifact_id="tensorrt",
        old_version=f"{TENSORRT_LIB}-{JAVACPP_BASE}-SNAPSHOT",
        new_version=TENSORRT_VERSION,
        description="JavaCPP Presets for TensorRT (mullerhai fork release)",
        classifiers=[
            "linux-arm64",
            "linux-x86_64",
            "windows-x86_64",
        ],
    ),
    Artifact(
        artifact_id="tensorrt-platform",
        old_version=f"{TENSORRT_LIB}-{JAVACPP_BASE}-SNAPSHOT",
        new_version=TENSORRT_VERSION,
        description="JavaCPP Presets Platform for TensorRT (mullerhai fork release)",
        classifiers=[],
        require_javadoc=False,
        require_sources=True,
    ),
    Artifact(
        artifact_id="tritonserver",
        old_version=f"{TRITONSERVER_LIB}-{JAVACPP_BASE}-SNAPSHOT",
        new_version=TRITONSERVER_VERSION,
        description="JavaCPP Presets for Triton Inference Server (mullerhai fork release)",
        classifiers=[
            "linux-arm64",
            "linux-x86_64",
            "windows-x86_64",
        ],
    ),
    Artifact(
        artifact_id="tritonserver-platform",
        old_version=f"{TRITONSERVER_LIB}-{JAVACPP_BASE}-SNAPSHOT",
        new_version=TRITONSERVER_VERSION,
        description="JavaCPP Presets Platform for Triton Inference Server (mullerhai fork release)",
        classifiers=[],
        require_javadoc=False,
        require_sources=True,
    ),
]

# CUDA redistributable packages — opt-in only (large natives; skipped for beta-06).
# Enable with: PUBLISH_CUDA_REDIST=1 ./publish.sh all
if os.environ.get("PUBLISH_CUDA_REDIST", "").strip().lower() in ("1", "true", "yes"):
    ARTIFACTS.extend(
        [
            Artifact(
                artifact_id=rid,
                old_version=f"{CUDA_LIB}-{JAVACPP_BASE}-SNAPSHOT",
                new_version=CUDA_VERSION,
                description=f"JavaCPP Presets Redistributable for CUDA ({rid}) (mullerhai fork release)",
                classifiers=list(CUDA_REDIST_CLASSIFIERS),
                require_javadoc=True,
                require_sources=True,
            )
            for rid in CUDA_REDIST_IDS
        ]
    )
    # cuda-platform-redist packages (platform aggregators, no native classifiers needed)
    ARTIFACTS.extend(
        [
            Artifact(
                artifact_id=rid,
                old_version=f"{CUDA_LIB}-{JAVACPP_BASE}-SNAPSHOT",
                new_version=CUDA_VERSION,
                description=f"JavaCPP Presets Platform Redistributable for CUDA ({rid}) (mullerhai fork release)",
                classifiers=[],
                require_javadoc=False,
                require_sources=True,
            )
            for rid in CUDA_PLATFORM_REDIST_IDS
        ]
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def sha_digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_checksums(path: Path) -> None:
    for algo, ext in (("md5", ".md5"), ("sha1", ".sha1"), ("sha256", ".sha256"), ("sha512", ".sha512")):
        (path.parent / (path.name + ext)).write_text(sha_digest(path, algo) + "\n", encoding="ascii")


def gpg_sign(path: Path, key_id: str = GPG_KEY_ID) -> Path:
    sig = path.with_suffix(path.suffix + ".asc")
    if sig.exists():
        sig.unlink()
    cmd = [
        "gpg",
        "--batch",
        "--yes",
        "--local-user",
        key_id,
        "--detach-sign",
        "--armor",
        "--output",
        str(sig),
        str(path),
    ]
    # passphrase via env if set
    env = os.environ.copy()
    if env.get("GPG_PASSPHRASE"):
        cmd = [
            "gpg",
            "--batch",
            "--yes",
            "--pinentry-mode",
            "loopback",
            "--passphrase-fd",
            "0",
            "--local-user",
            key_id,
            "--detach-sign",
            "--armor",
            "--output",
            str(sig),
            str(path),
        ]
        subprocess.run(cmd, input=env["GPG_PASSPHRASE"] + "\n", text=True, check=True)
    else:
        subprocess.run(cmd, check=True)
    return sig


def strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def local_path(artifact_id: str, version: str) -> Path:
    return M2_BYTEDECO / artifact_id / version


def staged_dir(art: Artifact, stage: Path) -> Path:
    return stage / GROUP_PATH / art.artifact_id / art.new_version


def find_file(base: Path, artifact_id: str, version: str, classifier: str | None, ext: str) -> Path | None:
    """Find SNAPSHOT or timestamped equivalent. Prefer newest mtime; skip corrupt jars."""
    if not base.exists():
        return None

    def is_valid_jar(path: Path) -> bool:
        """Check if file is a valid (non-empty, non-corrupt) JAR."""
        if not path.exists() or path.stat().st_size == 0:
            return False
        if path.stat().st_size < 100:
            return False
        if path.suffix == ".jar":
            try:
                with zipfile.ZipFile(path) as zf:
                    # Opening + reading central directory catches truncated/corrupt stubs.
                    # Full CRC scan (testzip) only for small jars; large natives are too slow.
                    _ = zf.namelist()
                    if path.stat().st_size < (32 << 20):
                        if zf.testzip() is not None:
                            return False
            except zipfile.BadZipFile:
                return False
            except Exception:
                return False
        return True

    candidates: list[Path] = []

    # literal SNAPSHOT name
    if classifier:
        snapshot_name = f"{artifact_id}-{version}-{classifier}.{ext}"
    else:
        snapshot_name = f"{artifact_id}-{version}.{ext}"
    snapshot_candidate = base / snapshot_name
    if snapshot_candidate.exists() and is_valid_jar(snapshot_candidate):
        candidates.append(snapshot_candidate)

    # timestamped: artifact-1.5.14-20260714.002819-98-sources.jar
    if classifier:
        pattern = re.compile(
            rf"^{re.escape(artifact_id)}-{re.escape(version.replace('-SNAPSHOT', ''))}-\d{{8}}\.\d{{6}}-\d+-{re.escape(classifier)}\.{re.escape(ext)}$"
        )
    else:
        pattern = re.compile(
            rf"^{re.escape(artifact_id)}-{re.escape(version.replace('-SNAPSHOT', ''))}-\d{{8}}\.\d{{6}}-\d+\.{re.escape(ext)}$"
        )
    for p in base.iterdir():
        if p.is_file() and pattern.match(p.name) and is_valid_jar(p):
            candidates.append(p)

    if not candidates:
        return None
    # Always pick newest by mtime (user requirement; critical for pytorch self-builds)
    return max(candidates, key=lambda p: p.stat().st_mtime)


def minimal_javadoc_jar(artifact_id: str, version: str, out: Path) -> None:
    """Central requires -javadoc.jar; generate a valid jar with Maven metadata if missing."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        readme = (
            f"{artifact_id} {version}\n"
            f"Javadoc not generated for this platform-native / republished artifact.\n"
            f"See {PROJECT_URL}\n"
        )
        zf.writestr("README-javadoc.txt", readme)
        zf.writestr(
            "META-INF/MANIFEST.MF",
            "Manifest-Version: 1.0\nCreated-By: mullerhai-publish\n\n",
        )
        # Add Maven metadata structure like real javadoc jars have
        pom_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>org.bytedeco</groupId>
  <artifactId>{artifact_id}</artifactId>
  <version>{version}</version>
</project>"""
        zf.writestr("META-INF/maven/org.bytedeco/" + artifact_id + "/pom.xml", pom_xml)
        zf.writestr(
            "META-INF/maven/org.bytedeco/" + artifact_id + "/pom.properties",
            f"version={version}\n"
            f"groupId=org.bytedeco\n"
            f"artifactId={artifact_id}\n"
        )
    out.write_bytes(buf.getvalue())


def minimal_sources_jar(artifact_id: str, version: str, out: Path) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "README-sources.txt",
            f"{artifact_id} {version}\nSources not bundled for this republished artifact.\nSee {PROJECT_URL}\n",
        )
        zf.writestr(
            "META-INF/MANIFEST.MF",
            "Manifest-Version: 1.0\nCreated-By: mullerhai-publish\n\n",
        )
    out.write_bytes(buf.getvalue())


def minimal_empty_jar(out: Path, note: str) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("META-INF/MANIFEST.MF", "Manifest-Version: 1.0\nCreated-By: mullerhai-publish\n\n")
        zf.writestr("README.txt", note + "\n")
    out.write_bytes(buf.getvalue())


# ---------------------------------------------------------------------------
# POM rewrite
# ---------------------------------------------------------------------------

POM_NS = "http://maven.apache.org/POM/4.0.0"
NSMAP = {"": POM_NS}


def register_ns() -> None:
    ET.register_namespace("", POM_NS)


def text_child(parent: ET.Element, tag: str, default: str | None = None) -> str | None:
    for c in parent:
        if strip_ns(c.tag) == tag:
            return c.text
    return default


def set_or_create(parent: ET.Element, tag: str, text: str) -> ET.Element:
    for c in parent:
        if strip_ns(c.tag) == tag:
            c.text = text
            return c
    el = ET.SubElement(parent, f"{{{POM_NS}}}{tag}")
    el.text = text
    return el


def remove_children(parent: ET.Element, tag: str) -> None:
    for c in list(parent):
        if strip_ns(c.tag) == tag:
            parent.remove(c)


def rewrite_version_string(v: str | None) -> str | None:
    if v is None:
        return None
    v = v.strip()
    # direct map
    if v in VERSION_MAP:
        return VERSION_MAP[v]
    # generic SNAPSHOT -> beta-01
    if v.endswith("-SNAPSHOT"):
        return v[: -len("-SNAPSHOT")] + f"-{SUFFIX}"
    # ${project.parent.version} already handled
    return v


def rewrite_pom_xml(src_pom: Path, art: Artifact) -> str:
    """Return rewritten POM XML string (standalone, no parent bytedeco)."""
    register_ns()
    raw = src_pom.read_text(encoding="utf-8")
    # Resolve Maven properties to the *old* SNAPSHOT coordinates first so that
    # non-published bytedeco deps (ffmpeg/opencv/...) are not accidentally
    # rewritten into our beta-01 line. Published deps are remapped later.
    # Resolve ${project.parent.version} after library-specific prefixes.
    # cuda is DROPPED later (Mac beta-03 does not ship cuda). Do not map it to
    # a bytedeco stable version — that would re-introduce org.bytedeco in the POM.
    raw = raw.replace(
        f"{OPENBLAS_LIB}-${{project.parent.version}}",
        f"{OPENBLAS_LIB}-{JAVACPP_BASE}-SNAPSHOT",
    )
    raw = raw.replace(
        f"{CUDA_LIB}-${{project.parent.version}}",
        f"{CUDA_LIB}-{JAVACPP_BASE}-SNAPSHOT",
    )
    raw = raw.replace(
        f"{PYTORCH_LIB}-${{project.parent.version}}",
        f"{PYTORCH_LIB}-{JAVACPP_BASE}-SNAPSHOT",
    )
    raw = raw.replace(
        f"{FFMPEG_LIB}-${{project.parent.version}}",
        f"{FFMPEG_LIB}-{JAVACPP_BASE}-SNAPSHOT",
    )
    raw = raw.replace(
        f"{OPENCV_LIB}-${{project.parent.version}}",
        f"{OPENCV_LIB}-{JAVACPP_BASE}-SNAPSHOT",
    )
    raw = raw.replace(
        f"{CPYTHON_LIB}-${{project.parent.version}}",
        f"{CPYTHON_LIB}-{JAVACPP_BASE}-SNAPSHOT",
    )
    raw = raw.replace(
        f"{NUMPY_LIB}-${{project.parent.version}}",
        f"{NUMPY_LIB}-{JAVACPP_BASE}-SNAPSHOT",
    )
    raw = raw.replace(
        f"{TENSORRT_LIB}-${{project.parent.version}}",
        f"{TENSORRT_LIB}-{JAVACPP_BASE}-SNAPSHOT",
    )
    raw = raw.replace(
        f"{TRITONSERVER_LIB}-${{project.parent.version}}",
        f"{TRITONSERVER_LIB}-{JAVACPP_BASE}-SNAPSHOT",
    )
    # generic parent version substitution last
    raw = raw.replace("${project.parent.version}", f"{JAVACPP_BASE}-SNAPSHOT")

    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        # fall back: create minimal pom
        return build_minimal_pom(art)

    # Drop parent (org.bytedeco:javacpp-presets) — Central needs self-contained POM
    for c in list(root):
        if strip_ns(c.tag) == "parent":
            root.remove(c)

    set_or_create(root, "groupId", GROUP_ID)
    set_or_create(root, "artifactId", art.artifact_id)
    set_or_create(root, "version", art.new_version)
    set_or_create(root, "name", text_child(root, "name") or art.artifact_id)
    set_or_create(root, "description", art.description)
    set_or_create(root, "url", PROJECT_URL)

    # licenses
    remove_children(root, "licenses")
    licenses = ET.SubElement(root, f"{{{POM_NS}}}licenses")
    lic = ET.SubElement(licenses, f"{{{POM_NS}}}license")
    set_or_create(lic, "name", LICENSE_NAME)
    set_or_create(lic, "url", LICENSE_URL)
    set_or_create(lic, "distribution", "repo")

    # developers
    remove_children(root, "developers")
    devs = ET.SubElement(root, f"{{{POM_NS}}}developers")
    dev = ET.SubElement(devs, f"{{{POM_NS}}}developer")
    set_or_create(dev, "id", DEV_ID)
    set_or_create(dev, "name", DEV_NAME)
    set_or_create(dev, "email", DEV_EMAIL)
    set_or_create(dev, "url", DEV_URL)
    set_or_create(dev, "organization", ORG_NAME)
    set_or_create(dev, "organizationUrl", DEV_URL)

    # scm
    remove_children(root, "scm")
    scm = ET.SubElement(root, f"{{{POM_NS}}}scm")
    set_or_create(scm, "url", SCM_URL)
    set_or_create(scm, "connection", SCM_CONN)
    set_or_create(scm, "developerConnection", SCM_DEV)

    # Dependency rewrite rules for Central:
    # 1) Never leave org.bytedeco or *-SNAPSHOT in published POMs.
    # 2) Our stack deps → io.github.mullerhai + *-beta-03 explicit versions.
    # 3) cuda / cuda-platform: DROP on this Mac release (not published yet;
    #    Linux would declare io.github.mullerhai:cuda:13.3-9.24-1.5.14-beta-03 later).
    DROP_ARTIFACTS: set[str] = set()  # cuda is published under io.github.mullerhai now
    ver_map = {
        # platform aggregators → current PUBLISH_SUFFIX (may be beta-08 re-release)
        "javacpp-platform": JAVACPP_VERSION,
        "openblas-platform": OPENBLAS_VERSION,
        "cuda-platform": CUDA_VERSION,
        "ffmpeg-platform": FFMPEG_VERSION,
        "opencv-platform": OPENCV_VERSION,
        "cpython-platform": CPYTHON_VERSION,
        "numpy-platform": NUMPY_VERSION,
        "pytorch-platform": PYTORCH_VERSION,
        "pytorch-platform-gpu": PYTORCH_VERSION,
        "tensorrt-platform": TENSORRT_VERSION,
        "tritonserver-platform": TRITONSERVER_VERSION,
        # native modules + classifiers → NATIVE_SUFFIX (already on Central)
        "javacpp": JAVACPP_NATIVE_VERSION,
        "openblas": OPENBLAS_NATIVE_VERSION,
        "cuda": CUDA_NATIVE_VERSION,
        "ffmpeg": FFMPEG_NATIVE_VERSION,
        "opencv": OPENCV_NATIVE_VERSION,
        "cpython": CPYTHON_NATIVE_VERSION,
        "numpy": NUMPY_NATIVE_VERSION,
        "pytorch": PYTORCH_NATIVE_VERSION,
        "tensorrt": TENSORRT_NATIVE_VERSION,
        "tritonserver": TRITONSERVER_NATIVE_VERSION,
    }
    for rid in CUDA_REDIST_IDS:
        ver_map[rid] = CUDA_NATIVE_VERSION
    for rid in CUDA_PLATFORM_REDIST_IDS:
        ver_map[rid] = CUDA_VERSION

    module_id = _module_id_for(art)

    # Collect dependency parents so we can remove nodes safely
    to_remove: list[tuple[ET.Element, ET.Element]] = []
    for parent in root.iter():
        for block in list(parent):
            if strip_ns(block.tag) != "dependency":
                continue
            g = a = v_el = None
            for c in block:
                t = strip_ns(c.tag)
                if t == "groupId":
                    g = c
                elif t == "artifactId":
                    a = c
                elif t == "version":
                    v_el = c
            aid = (a.text or "").strip() if a is not None else ""
            gid = (g.text or "").strip() if g is not None else ""
            # Resolve placeholders before matching (source platform POMs use these)
            if aid in ("${javacpp.moduleId}", "${project.artifactId}"):
                aid = module_id
                if a is not None:
                    a.text = module_id
            if gid in ("${project.groupId}",):
                gid = GROUP_ID
                if g is not None:
                    g.text = GROUP_ID

            # Explicit drop list only (empty = ship everything we rewrite)
            if aid in DROP_ARTIFACTS:
                to_remove.append((parent, block))
                continue

            # Rewrite our published stack to io.github.mullerhai
            if aid in PUBLISHED_ARTIFACTS or (gid in (OLD_GROUP, GROUP_ID) and aid in ver_map):
                if g is not None:
                    g.text = GROUP_ID
                else:
                    g = ET.SubElement(block, f"{{{POM_NS}}}groupId")
                    g.text = GROUP_ID
                target_ver = ver_map.get(aid) or rewrite_version_string(
                    (v_el.text if v_el is not None else None) or ""
                )
                if v_el is None:
                    v_el = ET.SubElement(block, f"{{{POM_NS}}}version")
                v_el.text = target_ver
                continue

            # Any remaining org.bytedeco dep we are not publishing → drop
            # (avoids SNAPSHOT / foreign groupId on Central)
            if gid == OLD_GROUP:
                to_remove.append((parent, block))
                continue

            # Non-bytedeco deps: strip accidental SNAPSHOT versions if any
            if v_el is not None and v_el.text and "SNAPSHOT" in v_el.text:
                v_el.text = v_el.text.replace("-SNAPSHOT", f"-{SUFFIX}")

    for parent, block in to_remove:
        try:
            parent.remove(block)
        except ValueError:
            pass

    for c in list(root):
        if strip_ns(c.tag) in ("build", "profiles", "repositories", "pluginRepositories", "distributionManagement"):
            root.remove(c)

    # Ensure packaging
    if text_child(root, "packaging") is None:
        set_or_create(root, "packaging", art.packaging)

    # Critical for Central consumers (Maven/Coursier/sbt): expand every
    # ${javacpp.platform.*}/${javacpp.moduleId}/${project.*} left after dropping parent.
    expand_javacpp_placeholders(root, art)

    # Pretty-ish serialization
    rough = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return rough.decode("utf-8")


def _module_id_for(art: Artifact) -> str:
    aid = art.artifact_id
    if aid.endswith("-platform-gpu"):
        return aid[: -len("-platform-gpu")]
    if aid.startswith("cuda-platform-redist"):
        # cuda-platform-redist -> cuda-redist ; cuda-platform-redist-cublas -> cuda-redist-cublas
        return "cuda-redist" + aid[len("cuda-platform-redist") :]
    if aid.endswith("-platform"):
        return aid[: -len("-platform")]
    return aid


def _platform_extension_for(art: Artifact, root: ET.Element | None = None) -> str:
    """Return classifier extension (e.g. '-gpu') from POM properties or artifact id."""
    if root is not None:
        for props in root:
            if strip_ns(props.tag) != "properties":
                continue
            for p in props:
                if strip_ns(p.tag) == "javacpp.platform.extension" and p.text:
                    return (p.text or "").strip()
    if art.artifact_id.endswith("-platform-gpu") or art.artifact_id.endswith("-gpu"):
        return "-gpu"
    return ""


# Host platforms that appear as ${javacpp.platform.<name>}
_PLATFORM_HOSTS = [
    "android-arm",
    "android-arm64",
    "android-x86",
    "android-x86_64",
    "ios-arm64",
    "ios-x86_64",
    "linux-armhf",
    "linux-arm64",
    "linux-ppc64le",
    "linux-riscv64",
    "linux-x86",
    "linux-x86_64",
    "macosx-arm64",
    "macosx-x86_64",
    "windows-arm64",
    "windows-x86",
    "windows-x86_64",
]


def expand_javacpp_placeholders(root: ET.Element, art: Artifact) -> None:
    """Expand JavaCPP platform POM placeholders so Coursier/sbt can resolve classifiers.

    bytedeco source POMs keep:
      <classifier>${javacpp.platform.linux-x86_64}</classifier>
    with parent defining javacpp.platform.linux-x86_64 = linux-x86_64${extension}.
    We drop the parent for Central, so these MUST become concrete strings
    (e.g. linux-x86_64-gpu for pytorch-platform-gpu).
    """
    module_id = _module_id_for(art)
    extension = _platform_extension_for(art, root)
    # Prefer explicit property javacpp.moduleId if present
    for props in root:
        if strip_ns(props.tag) != "properties":
            continue
        for p in props:
            if strip_ns(p.tag) == "javacpp.moduleId" and p.text and p.text.strip():
                module_id = p.text.strip()

    platform_map = {h: f"{h}{extension}" for h in _PLATFORM_HOSTS}

    def expand_text(s: str | None) -> str | None:
        if s is None:
            return None
        out = s
        out = out.replace("${project.groupId}", GROUP_ID)
        out = out.replace("${project.version}", art.new_version)
        out = out.replace("${javacpp.moduleId}", module_id)
        out = out.replace("${javacpp.platform.extension}", extension)
        for host, concrete in platform_map.items():
            out = out.replace(f"${{javacpp.platform.{host}}}", concrete)
        # leftover empty platform props (profile-cleared) → empty classifier (drop later)
        out = re.sub(r"\$\{javacpp\.platform\.[^}]+\}", "", out)
        return out

    # Expand all element texts under the POM
    for el in root.iter():
        if el.text and "${" in el.text:
            el.text = expand_text(el.text)
        if el.tail and "${" in el.tail:
            el.tail = expand_text(el.tail)

    # Drop dependency blocks whose classifier became empty (disabled platforms)
    for parent in list(root.iter()):
        for block in list(parent):
            if strip_ns(block.tag) != "dependency":
                continue
            clf = None
            for c in block:
                if strip_ns(c.tag) == "classifier":
                    clf = c
                    break
            if clf is not None and (clf.text is None or not str(clf.text).strip()):
                parent.remove(block)

    # Remove <properties> — everything needed is expanded; keeps POM self-contained
    for c in list(root):
        if strip_ns(c.tag) == "properties":
            root.remove(c)


def _dep(gid: str, aid: str, ver: str, opt: bool = False) -> str:
    opts = f"\n      <optional>true</optional>" if opt else ""
    return f"""    <dependency>
      <groupId>{gid}</groupId>
      <artifactId>{aid}</artifactId>
      <version>{ver}</version>{opts}
    </dependency>"""

def build_minimal_pom(art: Artifact) -> str:
    aid = art.artifact_id

    def dep(a, v, opt=False): return _dep(GROUP_ID, a, v, opt)

    # core javacpp: no javacpp-platform dep
    if aid == "javacpp":
        deps = ""
    elif aid == "javacpp-platform":
        deps = ""  # pure aggregator, no deps
    # openblas depends on javacpp
    elif aid == "openblas":
        deps = "\n" + dep("javacpp", JAVACPP_VERSION)
    elif aid == "openblas-platform":
        deps = "\n" + dep("javacpp", JAVACPP_VERSION) + "\n" + dep("openblas", OPENBLAS_VERSION)
    # ffmpeg depends on javacpp
    elif aid == "ffmpeg":
        deps = "\n" + dep("javacpp", JAVACPP_VERSION)
    elif aid == "ffmpeg-platform":
        deps = "\n" + dep("javacpp", JAVACPP_VERSION) + "\n" + dep("ffmpeg", FFMPEG_VERSION)
    # opencv depends on javacpp
    elif aid == "opencv":
        deps = "\n" + dep("javacpp", JAVACPP_VERSION)
    elif aid == "opencv-platform":
        deps = "\n" + dep("javacpp", JAVACPP_VERSION) + "\n" + dep("opencv", OPENCV_VERSION)
    # cpython depends on javacpp
    elif aid == "cpython":
        deps = "\n" + dep("javacpp", JAVACPP_VERSION)
    elif aid == "cpython-platform":
        deps = "\n" + dep("javacpp", JAVACPP_VERSION) + "\n" + dep("cpython", CPYTHON_VERSION)
    # numpy depends on javacpp + cpython + openblas
    elif aid == "numpy":
        deps = "\n" + dep("javacpp", JAVACPP_VERSION) + "\n" + dep("cpython", CPYTHON_VERSION) + "\n" + dep("openblas", OPENBLAS_VERSION)
    elif aid == "numpy-platform":
        deps = "\n" + dep("javacpp", JAVACPP_VERSION) + "\n" + dep("numpy", NUMPY_VERSION)
    # pytorch depends on javacpp + openblas
    elif aid == "pytorch":
        deps = "\n" + dep("javacpp", JAVACPP_VERSION) + "\n" + dep("openblas", OPENBLAS_VERSION)
    elif aid == "pytorch-platform":
        deps = "\n" + dep("javacpp", JAVACPP_VERSION) + "\n" + dep("pytorch", PYTORCH_VERSION)
    # cuda + redistributables depend on javacpp only
    elif aid == "cuda" or aid in CUDA_REDIST_IDS:
        deps = "\n" + dep("javacpp", JAVACPP_VERSION)
    elif aid == "cuda-platform":
        deps = "\n" + dep("javacpp", JAVACPP_VERSION) + "\n" + dep("cuda", CUDA_VERSION)
    elif aid == "tensorrt":
        deps = "\n" + dep("javacpp", JAVACPP_VERSION) + "\n" + dep("cuda", CUDA_VERSION)
    elif aid == "tensorrt-platform":
        deps = "\n" + dep("javacpp", JAVACPP_VERSION) + "\n" + dep("tensorrt", TENSORRT_VERSION)
    elif aid == "tritonserver":
        deps = "\n" + dep("javacpp", JAVACPP_VERSION)
    elif aid == "tritonserver-platform":
        deps = "\n" + dep("javacpp", JAVACPP_VERSION) + "\n" + dep("tritonserver", TRITONSERVER_VERSION)
    else:
        deps = ""

    deps_xml = f"\n  <dependencies>{deps}\n  </dependencies>" if deps else ""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <groupId>{GROUP_ID}</groupId>
  <artifactId>{aid}</artifactId>
  <version>{art.new_version}</version>
  <packaging>{art.packaging}</packaging>
  <name>{aid}</name>
  <description>{art.description}</description>
  <url>{PROJECT_URL}</url>
  <licenses>
    <license>
      <name>{LICENSE_NAME}</name>
      <url>{LICENSE_URL}</url>
      <distribution>repo</distribution>
    </license>
  </licenses>
  <developers>
    <developer>
      <id>{DEV_ID}</id>
      <name>{DEV_NAME}</name>
      <email>{DEV_EMAIL}</email>
      <url>{DEV_URL}</url>
      <organization>{ORG_NAME}</organization>
      <organizationUrl>{DEV_URL}</organizationUrl>
    </developer>
  </developers>
  <scm>
    <url>{SCM_URL}</url>
    <connection>{SCM_CONN}</connection>
    <developerConnection>{SCM_DEV}</developerConnection>
  </scm>{deps_xml}
</project>
"""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <groupId>{GROUP_ID}</groupId>
  <artifactId>{art.artifact_id}</artifactId>
  <version>{art.new_version}</version>
  <packaging>{art.packaging}</packaging>
  <name>{art.artifact_id}</name>
  <description>{art.description}</description>
  <url>{PROJECT_URL}</url>
  <licenses>
    <license>
      <name>{LICENSE_NAME}</name>
      <url>{LICENSE_URL}</url>
      <distribution>repo</distribution>
    </license>
  </licenses>
  <developers>
    <developer>
      <id>{DEV_ID}</id>
      <name>{DEV_NAME}</name>
      <email>{DEV_EMAIL}</email>
      <url>{DEV_URL}</url>
      <organization>{ORG_NAME}</organization>
      <organizationUrl>{DEV_URL}</organizationUrl>
    </developer>
  </developers>
  <scm>
    <url>{SCM_URL}</url>
    <connection>{SCM_CONN}</connection>
    <developerConnection>{SCM_DEV}</developerConnection>
  </scm>{deps}
</project>
"""


def synthetic_platform_pom(art: Artifact) -> str:
    """Build platform aggregator POM if not present in local m2."""
    base = art.artifact_id.replace("-platform", "")
    base_ver = art.new_version
    # list known classifiers for platform deps
    class_map = {
        "javacpp": [
            "linux-x86_64",
            "linux-arm64",
            "macosx-x86_64",
            "macosx-arm64",
            "windows-x86_64",
            "windows-arm64",
            "android-arm64",
            "android-x86_64",
        ],
        "openblas": [
            "linux-x86_64",
            "linux-arm64",
            "macosx-x86_64",
            "macosx-arm64",
            "windows-x86_64",
        ],
        "cuda": ["linux-x86_64", "linux-arm64", "windows-x86_64"],
        "pytorch": ["macosx-arm64", "macosx-x86_64", "linux-x86_64", "linux-arm64", "windows-x86_64"],
        "pytorch-platform-gpu": [],  # aggregator only
        "tensorrt": ["linux-x86_64", "linux-arm64", "windows-x86_64"],
        "tritonserver": ["linux-x86_64", "linux-arm64", "windows-x86_64"],
    }
    classifiers = class_map.get(base, [])
    deps = [
        f"""    <dependency>
      <groupId>{GROUP_ID}</groupId>
      <artifactId>{base}</artifactId>
      <version>{base_ver}</version>
    </dependency>"""
    ]
    if base != "javacpp":
        # platform modules also depend on javacpp-platform
        deps.insert(
            0,
            f"""    <dependency>
      <groupId>{GROUP_ID}</groupId>
      <artifactId>javacpp-platform</artifactId>
      <version>{JAVACPP_VERSION}</version>
    </dependency>""",
        )
    for c in classifiers:
        deps.append(
            f"""    <dependency>
      <groupId>{GROUP_ID}</groupId>
      <artifactId>{base}</artifactId>
      <version>{base_ver}</version>
      <classifier>{c}</classifier>
    </dependency>"""
        )
    deps_xml = "\n".join(deps)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <groupId>{GROUP_ID}</groupId>
  <artifactId>{art.artifact_id}</artifactId>
  <version>{art.new_version}</version>
  <packaging>jar</packaging>
  <name>{art.artifact_id}</name>
  <description>{art.description}</description>
  <url>{PROJECT_URL}</url>
  <licenses>
    <license>
      <name>{LICENSE_NAME}</name>
      <url>{LICENSE_URL}</url>
      <distribution>repo</distribution>
    </license>
  </licenses>
  <developers>
    <developer>
      <id>{DEV_ID}</id>
      <name>{DEV_NAME}</name>
      <email>{DEV_EMAIL}</email>
      <url>{DEV_URL}</url>
      <organization>{ORG_NAME}</organization>
      <organizationUrl>{DEV_URL}</organizationUrl>
    </developer>
  </developers>
  <scm>
    <url>{SCM_URL}</url>
    <connection>{SCM_CONN}</connection>
    <developerConnection>{SCM_DEV}</developerConnection>
  </scm>
  <dependencies>
{deps_xml}
  </dependencies>
</project>
"""


# ---------------------------------------------------------------------------
# Stage / Sign / Bundle / Upload
# ---------------------------------------------------------------------------

def stage_artifact(art: Artifact, stage: Path) -> list[Path]:
    out_dir = staged_dir(art, stage)
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    src_base = local_path(art.artifact_id, art.old_version)

    # POM
    pom_out = out_dir / f"{art.artifact_id}-{art.new_version}.pom"
    if src_base.exists():
        src_pom = find_file(src_base, art.artifact_id, art.old_version, None, "pom")
    else:
        src_pom = None

    if src_pom and src_pom.exists():
        pom_xml = rewrite_pom_xml(src_pom, art)
    elif art.artifact_id.endswith("-platform"):
        log(f"  ! no local POM for {art.artifact_id}; synthesizing platform POM")
        pom_xml = synthetic_platform_pom(art)
    else:
        log(f"  ! no local POM for {art.artifact_id}; synthesizing minimal POM")
        pom_xml = build_minimal_pom(art)
    pom_out.write_text(pom_xml, encoding="utf-8")
    produced.append(pom_out)

    # Main jar
    main_out = out_dir / f"{art.artifact_id}-{art.new_version}.jar"
    src_main = find_file(src_base, art.artifact_id, art.old_version, None, "jar") if src_base.exists() else None
    if src_main and src_main.exists():
        shutil.copy2(src_main, main_out)
    else:
        log(f"  ! missing main jar for {art.artifact_id}; creating empty placeholder jar")
        minimal_empty_jar(main_out, f"{art.artifact_id} {art.new_version} placeholder (platform aggregator or missing binary)")
    produced.append(main_out)

    # sources
    sources_out = out_dir / f"{art.artifact_id}-{art.new_version}-sources.jar"
    src_sources = (
        find_file(src_base, art.artifact_id, art.old_version, "sources", "jar") if src_base.exists() else None
    )
    # Only copy if file exists AND is not empty
    if src_sources and src_sources.exists() and src_sources.stat().st_size > 0:
        shutil.copy2(src_sources, sources_out)
    else:
        # Central Portal requires -sources.jar for every component
        log(f"  ! missing sources for {art.artifact_id}; creating minimal sources jar")
        minimal_sources_jar(art.artifact_id, art.new_version, sources_out)
    if sources_out.exists():
        produced.append(sources_out)

    # javadoc
    javadoc_out = out_dir / f"{art.artifact_id}-{art.new_version}-javadoc.jar"
    src_javadoc = (
        find_file(src_base, art.artifact_id, art.old_version, "javadoc", "jar") if src_base.exists() else None
    )
    # Only copy if file exists AND is not empty (0 bytes = corrupted/incomplete)
    if src_javadoc and src_javadoc.exists() and src_javadoc.stat().st_size > 0:
        shutil.copy2(src_javadoc, javadoc_out)
    elif art.require_javadoc or True:
        # Central Portal requires javadoc for all jars in most validations
        log(f"  ! missing javadoc for {art.artifact_id}; creating minimal javadoc jar")
        minimal_javadoc_jar(art.artifact_id, art.new_version, javadoc_out)
    if javadoc_out.exists():
        produced.append(javadoc_out)

    # classified native jars
    for clf in art.classifiers:
        src_clf = find_file(src_base, art.artifact_id, art.old_version, clf, "jar") if src_base.exists() else None
        if not src_clf or not src_clf.exists():
            log(f"  - skip missing classifier {art.artifact_id}:{clf}")
            continue
        out_clf = out_dir / f"{art.artifact_id}-{art.new_version}-{clf}.jar"
        shutil.copy2(src_clf, out_clf)
        produced.append(out_clf)

    log(f"  staged {art.artifact_id}:{art.new_version} -> {out_dir} ({len(produced)} files)")
    return produced


def selected_artifacts(only: list[str] | None = None) -> list[Artifact]:
    """Filter ARTIFACTS by --only list.

    Matching rules (intentionally narrow — one package at a time):
      - exact artifactId
      - trailing ``*`` prefix: ``cuda-redist*`` selects all cuda-redist*
      - auto companion: ``foo`` also selects ``foo-platform``
        (``pytorch`` → ``pytorch-platform``; ``cuda-redist`` → ``cuda-platform-redist``)
      - alias: ``pytorch-gpu`` → ``pytorch-platform-gpu``
      - ``cuda`` does NOT select ``cuda-redist*``
    """
    if not only:
        return list(ARTIFACTS)
    selected: list[Artifact] = []
    seen: set[str] = set()
    for art in ARTIFACTS:
        for pat in only:
            pat = pat.strip()
            if not pat:
                continue
            aid = art.artifact_id
            match = False
            if aid == pat:
                match = True
            elif pat.endswith("*") and aid.startswith(pat[:-1]):
                match = True
            elif aid == f"{pat}-platform":
                # --only pytorch → also pytorch-platform
                match = True
            elif (
                pat.startswith("cuda-")
                and "redist" in pat
                and aid == "cuda-platform-" + pat[len("cuda-") :]
            ):
                # --only cuda-redist → cuda-platform-redist
                # --only cuda-redist-cublas → cuda-platform-redist-cublas
                match = True
            elif pat in ("pytorch-gpu", "pytorch-platform-gpu") and aid == "pytorch-platform-gpu":
                match = True
            if match and aid not in seen:
                selected.append(art)
                seen.add(aid)
                break
    if not selected:
        raise SystemExit(f"No artifacts matched --only={only}. Known: {[a.artifact_id for a in ARTIFACTS]}")
    return selected


def stage_all(stage: Path, only: list[str] | None = None) -> None:
    arts = selected_artifacts(only)
    if stage.exists():
        # Always fully wipe the stage dir used for this run.
        # Selective wipe by new_version only left leftover beta-XX trees that
        # sign_all would then re-sign and re-bundle (seen with beta-06 mixed in).
        if only:
            for art in arts:
                # wipe ALL versions under this artifact id in this stage
                art_root = stage / GROUP_PATH / art.artifact_id
                if art_root.exists():
                    shutil.rmtree(art_root)
            # also drop any other leftover artifact trees in an --only isolated stage
            # (isolated dirs are named staging-only-*; safe to fully clear)
            if stage.name.startswith("staging-only-"):
                shutil.rmtree(stage)
        else:
            shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)
    log(f"Staging into {stage}")
    log(f"Group: {GROUP_ID}")
    log(f"Versions: javacpp={JAVACPP_VERSION} openblas={OPENBLAS_VERSION} cuda={CUDA_VERSION} pytorch={PYTORCH_VERSION}")
    log(f"Selected artifacts ({len(arts)}): {[a.artifact_id for a in arts]}")
    for art in arts:
        log(f"== {art.artifact_id} ==")
        stage_artifact(art, stage)
    log("Stage complete.")


def sign_all(stage: Path) -> None:
    log(f"Signing artifacts under {stage} with key {GPG_KEY_ID}")
    files = [p for p in stage.rglob("*") if p.is_file() and not p.name.endswith(".asc") and not any(p.name.endswith(e) for e in (".md5", ".sha1", ".sha256", ".sha512"))]
    for p in sorted(files):
        log(f"  sign {p.relative_to(stage)}")
        gpg_sign(p)
        write_checksums(p)
        # also checksum the signature
        sig = p.with_suffix(p.suffix + ".asc")
        if sig.exists():
            write_checksums(sig)
    log("Sign complete.")


def bundle_all(stage: Path, bundle_dir: Path) -> Path:
    """Create a single deployment bundle ZIP for Central Portal upload.

    Central Portal expects a zip whose internal paths are Maven repository layout:
      io/github/mullerhai/<artifact>/<version>/<files>
    """
    bundle_dir.mkdir(parents=True, exist_ok=True)
    # Include pid + fractional seconds so parallel --only publishes never clobber
    # each other's zip (second-resolution stamps collided under concurrent runs).
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}-{int(time.time() * 1000) % 1000:03d}"
    # Derive a short label from stage dir name when isolated (staging-only-foo)
    label = ""
    if stage.name.startswith("staging-only-"):
        label = "-" + stage.name[len("staging-only-") :][:48]
    zip_path = bundle_dir / f"mullerhai-javacpp-stack-{SUFFIX}{label}-{stamp}.zip"
    if zip_path.exists():
        zip_path.unlink()

    count = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(stage.rglob("*")):
            if not p.is_file():
                continue
            # include signed files + checksums
            arc = p.relative_to(stage).as_posix()
            zf.write(p, arcname=arc)
            count += 1
    log(f"Bundle: {zip_path} ({count} files, {zip_path.stat().st_size / (1<<20):.1f} MiB)")
    return zip_path


def central_auth_header() -> str:
    user = os.environ.get("CENTRAL_USERNAME") or os.environ.get("SONATYPE_USERNAME")
    pwd = os.environ.get("CENTRAL_PASSWORD") or os.environ.get("SONATYPE_PASSWORD")
    if not user or not pwd:
        # try parse from settings.xml
        settings = Path.home() / ".m2" / "settings.xml"
        if settings.exists():
            try:
                tree = ET.parse(settings)
                for server in tree.getroot().iter():
                    if strip_ns(server.tag) != "server":
                        continue
                    sid = text_child(server, "id")
                    if sid == "central":
                        user = text_child(server, "username")
                        pwd = text_child(server, "password")
                        # expand ${env.X}
                        if user and user.startswith("${env.") and user.endswith("}"):
                            user = os.environ.get(user[6:-1])
                        if pwd and pwd.startswith("${env.") and pwd.endswith("}"):
                            pwd = os.environ.get(pwd[6:-1])
            except Exception as e:
                log(f"warn: could not parse settings.xml: {e}")
    if not user or not pwd:
        raise SystemExit(
            "Missing Central credentials. Set CENTRAL_USERNAME and CENTRAL_PASSWORD "
            "(Sonatype Central Portal user token) or put them in ~/.m2/settings.xml "
            "under <server><id>central</id>."
        )
    # Central Portal Publisher API expects:
    #   Authorization: Bearer base64(username:password)
    # (user token name + token password from https://central.sonatype.com/account)
    token = base64.b64encode(f"{user}:{pwd}".encode("utf-8")).decode("ascii")
    return f"Bearer {token}"


def upload_bundle(zip_path: Path, publishing_type: str = "USER_MANAGED") -> str:
    """Upload deployment bundle via curl subprocess. Returns deploymentId.

    publishing_type:
      USER_MANAGED - upload only; publish manually in UI or via API
      AUTOMATIC    - validate and publish automatically
    """
    import subprocess

    user, pwd = None, None
    settings = Path.home() / ".m2" / "settings.xml"
    if settings.exists():
        try:
            tree = ET.parse(settings)
            for server in tree.getroot().iter():
                sid = text_child(server, "id")
                if sid == "central":
                    user = text_child(server, "username")
                    pwd = text_child(server, "password")
        except Exception:
            pass

    if not user or not pwd:
        user = os.environ.get("CENTRAL_USERNAME") or os.environ.get("SONATYPE_USERNAME")
        pwd = os.environ.get("CENTRAL_PASSWORD") or os.environ.get("SONATYPE_PASSWORD")

    if not user or not pwd:
        raise SystemExit("Missing Central credentials")

    file_size = zip_path.stat().st_size
    log(f"Uploading {zip_path.name} ({file_size/(1<<20):.1f} MiB) to Central Portal ...")
    log(f"  (using curl subprocess for reliable large file transfer)")

    # Build curl command with streaming upload (-T flag)
    # First create a .netrc file for auth (more reliable than -u for large files)
    netrc_path = Path(tempfile.gettempdir()) / "muller_netrc"
    netrc_path.write_text(f"machine central.sonatype.com login {user} password {pwd}\n", encoding="utf-8")

    # Large multipart uploads over HTTP/2 often return empty body on Central Portal.
    # Force HTTP/1.1; write body to file and capture HTTP status via -w.
    body_path = Path(tempfile.gettempdir()) / f"central_upload_body_{os.getpid()}.txt"
    cmd = [
        "curl",
        "-sS",
        "-X",
        "POST",
        "--http1.1",
        "-n",
        "--netrc-file",
        str(netrc_path),
        "-F",
        f"bundle=@{zip_path};type=application/zip",
        f"https://central.sonatype.com/api/v1/publisher/upload?publishingType={publishing_type}&name={zip_path.stem}",
        "--connect-timeout",
        "120",
        "--max-time",
        "7200",
        "-o",
        str(body_path),
        "-w",
        "HTTP_CODE=%{http_code} SIZE_UPLOAD=%{size_upload} TIME=%{time_total}\n",
    ]

    log(f"  running: curl --http1.1 -F bundle=@{zip_path.name} ...")

    start_time = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    netrc_path.unlink(missing_ok=True)

    log(f"  upload completed in {elapsed:.1f}s")
    if proc.stdout:
        log(f"  curl meta: {proc.stdout.strip()}")
    if proc.returncode != 0:
        log(f"  curl stderr: {(proc.stderr or '')[-1000:]}")
        raise SystemExit(f"Upload failed with return code {proc.returncode}")

    deployment_id = body_path.read_text(encoding="utf-8", errors="replace").strip() if body_path.exists() else ""
    body_path.unlink(missing_ok=True)

    http_code = None
    if proc.stdout and "HTTP_CODE=" in proc.stdout:
        try:
            http_code = proc.stdout.split("HTTP_CODE=")[1].split()[0]
        except Exception:
            http_code = None

    if http_code and http_code not in ("200", "201", "202"):
        log(f"  unexpected HTTP {http_code}: {deployment_id[:500]}")
        raise SystemExit(f"Upload failed HTTP {http_code}: {deployment_id[:200]}")

    if not deployment_id or "{" in deployment_id:
        log(f"  unexpected response: {deployment_id[:500]}")
        raise SystemExit(f"Upload failed: {deployment_id[:200]}")

    log(f"Upload OK. deploymentId = {deployment_id}")
    return deployment_id


def poll_status(deployment_id: str, timeout_s: int = 1800) -> dict:
    auth = central_auth_header()
    url = f"{CENTRAL_STATUS}?id={deployment_id}"
    start = time.time()
    while True:
        req = urllib.request.Request(url, method="POST")
        req.add_header("Authorization", auth)
        try:
            with urllib.request.urlopen(req, timeout=800, context=ssl_ctx) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            raise SystemExit(f"Status check failed HTTP {e.code}: {err}") from e
        state = data.get("deploymentState") or data.get("state") or "?"
        log(f"  deployment {deployment_id}: {state}")
        if state in ("PUBLISHED", "FAILED", "VALIDATED"):
            return data
        if time.time() - start > timeout_s:
            log("Timeout waiting for deployment; check https://central.sonatype.com/publishing")
            return data
        time.sleep(15)


def publish_deployment(deployment_id: str) -> None:
    auth = central_auth_header()
    url = f"{CENTRAL_PUBLISH}/{deployment_id}"
    req = urllib.request.Request(url, method="POST", data=b"")
    req.add_header("Authorization", auth)
    try:
        with urllib.request.urlopen(req, timeout=800, context=ssl_ctx) as resp:
            log(f"Publish requested: HTTP {resp.status}")
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Publish failed HTTP {e.code}: {err}") from e


def install_local(stage: Path, only: list[str] | None = None) -> None:
    """Copy staged (unsigned) artifacts into local m2 under new GAV."""
    dest_root = M2_LOCAL / GROUP_PATH
    arts = selected_artifacts(only)
    log(f"Installing staged artifacts into {dest_root}")
    for art in arts:
        src = staged_dir(art, stage)
        if not src.exists():
            log(f"  skip missing staged {art.artifact_id}")
            continue
        dest = dest_root / art.artifact_id / art.new_version
        dest.mkdir(parents=True, exist_ok=True)
        for p in src.iterdir():
            if p.suffix in (".asc", ".md5", ".sha1", ".sha256", ".sha512"):
                continue
            if p.is_file():
                shutil.copy2(p, dest / p.name)
        log(f"  installed {GROUP_ID}:{art.artifact_id}:{art.new_version}")
    log("Local install done. Consumers can depend on io.github.mullerhai:*")


def print_summary() -> None:
    print(
        f"""
============================================================
  mullerhai JavaCPP stack  ->  Maven Central
============================================================
  groupId : {GROUP_ID}
  versions (multi-platform + cuda):
    javacpp / javacpp-platform : {JAVACPP_VERSION}
    openblas / openblas-platform : {OPENBLAS_VERSION}
    ffmpeg / ffmpeg-platform : {FFMPEG_VERSION}
    opencv / opencv-platform : {OPENCV_VERSION}
    pytorch / pytorch-platform : {PYTORCH_VERSION}
  GPG key : {GPG_KEY_ID}
  SCM     : {SCM_URL}

  Prerequisites before --upload:
    1. Account at https://central.sonatype.com/
    2. Namespace claim for '{GROUP_ID}' (verify github.com/mullerhai)
    3. User token -> export CENTRAL_USERNAME / CENTRAL_PASSWORD
       or fill ~/.m2/settings.xml from settings.xml.template
    4. Upload public key (already sent to keys.openpgp.org):
       gpg --keyserver hkps://keys.openpgp.org --send-keys {GPG_KEY_ID}
       Then verify email at keys.openpgp.org for identity binding.

  Consumer dependency example:
    <dependency>
      <groupId>io.github.mullerhai</groupId>
      <artifactId>pytorch</artifactId>
      <version>{PYTORCH_VERSION}</version>
    </dependency>
    <dependency>
      <groupId>io.github.mullerhai</groupId>
      <artifactId>pytorch</artifactId>
      <version>{PYTORCH_VERSION}</version>
      <classifier>macosx-arm64</classifier>
    </dependency>
============================================================
"""
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish mullerhai JavaCPP stack to Maven Central")
    parser.add_argument(
        "command",
        choices=["stage", "sign", "bundle", "upload", "all", "install-local", "status", "summary"],
        help="Pipeline step",
    )
    parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--upload", action="store_true", help="With 'all': also upload after bundling")
    parser.add_argument(
        "--publishing-type",
        choices=["USER_MANAGED", "AUTOMATIC"],
        default="USER_MANAGED",
        help="Central Portal publishing type (default USER_MANAGED = review in UI)",
    )
    parser.add_argument("--deployment-id", help="For status/publish commands")
    parser.add_argument("--publish", action="store_true", help="After upload validation, call publish API")
    parser.add_argument("--bundle-file", type=Path, help="Existing bundle zip for upload")
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Upload and return immediately without polling Central deployment status",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help=(
            "Limit stage/sign/bundle/install to matching artifactIds "
            "(repeatable; also accepts comma-separated). "
            "Prefix match: --only cuda-redist selects all cuda-redist*."
        ),
    )
    args = parser.parse_args(argv)

    only: list[str] | None = None
    if args.only:
        only = []
        for item in args.only:
            only.extend(x.strip() for x in item.split(",") if x.strip())

    print_summary()

    if args.command == "summary":
        return 0

    if args.command == "stage":
        stage_all(args.stage_dir, only=only)
        return 0

    if args.command == "sign":
        if not args.stage_dir.exists():
            stage_all(args.stage_dir, only=only)
        sign_all(args.stage_dir)
        return 0

    if args.command == "bundle":
        if not args.stage_dir.exists():
            raise SystemExit("No staging dir; run stage+sign first")
        bundle_all(args.stage_dir, args.bundle_dir)
        return 0

    if args.command == "install-local":
        if not args.stage_dir.exists():
            stage_all(args.stage_dir, only=only)
        install_local(args.stage_dir, only=only)
        return 0

    if args.command == "upload":
        z = args.bundle_file
        if z is None:
            # pick newest bundle
            bundles = sorted(args.bundle_dir.glob("mullerhai-javacpp-stack-*.zip"), key=lambda p: p.stat().st_mtime)
            if not bundles:
                raise SystemExit("No bundle zip found; run bundle first")
            z = bundles[-1]
        dep_id = upload_bundle(z, publishing_type=args.publishing_type)
        if args.no_wait:
            log(f"Upload submitted (no-wait). deploymentId={dep_id}")
            log("Review later: https://central.sonatype.com/publishing/deployments")
            return 0
        data = poll_status(dep_id)
        if args.publish and (data.get("deploymentState") == "VALIDATED"):
            publish_deployment(dep_id)
            poll_status(dep_id)
        log(f"Done. Review: https://central.sonatype.com/publishing/deployments")
        log(f"deploymentId={dep_id}")
        return 0

    if args.command == "status":
        if not args.deployment_id:
            raise SystemExit("--deployment-id required")
        data = poll_status(args.deployment_id, timeout_s=30)
        print(json.dumps(data, indent=2))
        return 0

    if args.command == "all":
        # For selective publishes (e.g. cuda-redist only), use a dedicated stage dir
        # so we never re-upload already-published GAVs from a leftover full stage.
        stage_dir = args.stage_dir
        if only:
            stage_dir = args.stage_dir.parent / f"staging-only-{'-'.join(only)[:40]}"
            log(f"--only set -> using isolated stage dir: {stage_dir}")
        stage_all(stage_dir, only=only)
        sign_all(stage_dir)
        z = bundle_all(stage_dir, args.bundle_dir)
        install_local(stage_dir, only=only)
        if args.upload:
            dep_id = upload_bundle(z, publishing_type=args.publishing_type)
            if args.no_wait:
                log(f"Upload submitted (no-wait). deploymentId={dep_id}")
                log(f"Bundle: {z}")
                log("Review later: https://central.sonatype.com/publishing/deployments")
                return 0
            data = poll_status(dep_id)
            if args.publish and (data.get("deploymentState") == "VALIDATED"):
                publish_deployment(dep_id)
                poll_status(dep_id)
            log(f"deploymentId={dep_id}")
            log("Review: https://central.sonatype.com/publishing/deployments")
        else:
            log(f"Bundle ready: {z}")
            log("Set CENTRAL_USERNAME/CENTRAL_PASSWORD then re-run with --upload")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
