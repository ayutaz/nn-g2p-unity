#!/usr/bin/env python3
"""
Download nn-g2p model artifacts from Hugging Face into Unity StreamingAssets.

Usage:
  uv run python tools/download_hf_nn_g2p.py --repo ayousanz/nn-g2p-jp

Auth:
  Set HF_TOKEN for private repositories.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List
from urllib import error, parse, request

PROXY_ENV_KEYS = (
    "ALL_PROXY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "GIT_HTTP_PROXY",
    "GIT_HTTPS_PROXY",
)

EXPECTED_ONNX = ("encoder.onnx", "ctc_heads.onnx", "decoder_step.onnx")
EXPECTED_VOCAB = (
    "ja_grapheme_m4.txt",
    "ja_phones_m8.txt",
    "ja_prosody_or_stress_m8.txt",
)


def clear_proxy_env() -> None:
    for key in PROXY_ENV_KEYS:
        os.environ.pop(key, None)


def opener_without_proxy() -> request.OpenerDirector:
    return request.build_opener(request.ProxyHandler({}))


def auth_headers(token: str | None) -> Dict[str, str]:
    headers = {"User-Agent": "nn-g2p-unity-downloader/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_json(url: str, token: str | None) -> Dict:
    req = request.Request(url, headers=auth_headers(token))
    try:
        with opener_without_proxy().open(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as ex:
        body = ex.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {ex.code} for {url}: {body}") from ex


def download_file(repo_id: str, rfilename: str, out_path: Path, token: str | None) -> None:
    encoded_path = parse.quote(rfilename, safe="/")
    url = f"https://huggingface.co/{repo_id}/resolve/main/{encoded_path}"
    req = request.Request(url, headers=auth_headers(token))
    try:
        with opener_without_proxy().open(req, timeout=120) as resp:
            data = resp.read()
    except error.HTTPError as ex:
        body = ex.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {ex.code} when downloading {rfilename}: {body}") from ex

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)


def choose_file_by_basename(siblings: Iterable[str], basename: str) -> str | None:
    candidates: List[str] = [rf for rf in siblings if Path(rf).name == basename]
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x.count("/"), len(x)))
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download nn-g2p Hugging Face model assets for Unity.")
    parser.add_argument("--repo", default="ayousanz/nn-g2p-jp", help="Hugging Face repo id")
    parser.add_argument(
        "--output-root",
        default="Assets/StreamingAssets/nn-g2p",
        help="Target root in Unity project",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN", ""),
        help="Hugging Face token (default: HF_TOKEN env var)",
    )
    args = parser.parse_args()

    clear_proxy_env()

    repo_id = args.repo.strip()
    token = args.token.strip() or None
    output_root = Path(args.output_root).resolve()

    model_api_url = f"https://huggingface.co/api/models/{repo_id}"
    try:
        model_info = fetch_json(model_api_url, token)
    except RuntimeError as ex:
        print(f"[error] Failed to query model metadata: {ex}")
        print("[hint] If the repo is private, set HF_TOKEN and rerun.")
        return 1

    siblings_raw = model_info.get("siblings", [])
    siblings = [entry.get("rfilename", "") for entry in siblings_raw if entry.get("rfilename")]
    if not siblings:
        print("[error] No files were listed in model metadata.")
        return 1

    selected_onnx: Dict[str, str] = {}
    selected_vocab: Dict[str, str] = {}

    for name in EXPECTED_ONNX:
        found = choose_file_by_basename(siblings, name)
        if not found:
            print(f"[error] Missing expected ONNX file in repo: {name}")
            return 1
        selected_onnx[name] = found

    for name in EXPECTED_VOCAB:
        found = choose_file_by_basename(siblings, name)
        if not found:
            print(f"[error] Missing expected vocab file in repo: {name}")
            return 1
        selected_vocab[name] = found

    downloaded = {"onnx": {}, "vocab": {}}

    for local_name, remote_name in selected_onnx.items():
        out_path = output_root / "onnx" / local_name
        print(f"[download] {remote_name} -> {out_path}")
        download_file(repo_id, remote_name, out_path, token)
        downloaded["onnx"][local_name] = remote_name

    for local_name, remote_name in selected_vocab.items():
        out_path = output_root / "vocab" / local_name
        print(f"[download] {remote_name} -> {out_path}")
        download_file(repo_id, remote_name, out_path, token)
        downloaded["vocab"][local_name] = remote_name

    meta = {
        "repo": repo_id,
        "sha": model_info.get("sha", ""),
        "downloaded": downloaded,
    }
    meta_path = output_root / "download_manifest.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] Wrote download manifest: {meta_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
