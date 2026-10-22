"""T3.3：manifest model_files 下载源逐一核对（临时脚本）。

本引擎无 marker/key_files（全部单文件条目），核对项为：
    1. 每个 source.url（ModelScope 主源）可访问（GET + Range 0-0 → 206/200）
    2. Content-Range / Content-Length 与 manifest 声明的 size_bytes 一致（1% 容差）
    3. 有 mirror_url 的同时核对镜像可达性（不校验大小，镜像侧容差更宽）
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.yaml"
TOLERANCE = 0.01  # 1% 大小容差


def probe(url: str) -> tuple[int | None, int]:
    """探测 URL，返回 (总大小 or None, http_status)。"""
    try:
        with httpx.Client(
            timeout=httpx.Timeout(20, read=20),
            follow_redirects=True,
        ) as client:
            resp = client.get(url, headers={"Range": "bytes=0-0"})
        status = resp.status_code
        if status in (200, 206):
            cr = resp.headers.get("content-range")
            if cr and "/" in cr:
                total = int(cr.split("/")[-1])
                return total, status
            cl = resp.headers.get("content-length")
            if cl:
                return int(cl), status  # 200 时 content-length 即全量
            return None, status
        return None, status
    except Exception as exc:  # noqa: BLE001
        print(f"    [EXC] {exc!r}")
        return None, -1


def main() -> int:
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    files = data["models"]["model_files"]
    ok_all = True
    for f in files:
        fid = f["id"]
        src = f.get("source") or {}
        url = src.get("url", "")
        declared = f.get("size_bytes") or 0
        mirror = src.get("mirror_url", "")
        print(f"[{fid}] {f.get('filename')}")
        total, status = probe(url)
        if total is None:
            print(f"    [FAIL] 主源不可达 status={status} url={url}")
            ok_all = False
        else:
            size_ok = declared == 0 or abs(total - declared) <= max(1, declared * TOLERANCE)
            print(f"    主源: status={status} 总大小={total:,} 声明={declared:,} "
                  f"{'✅' if size_ok else '❌ 大小不符'}")
            if not size_ok:
                ok_all = False
        if mirror:
            mtotal, mstatus = probe(mirror)
            if mtotal is None:
                print(f"    [WARN] 镜像不可达 status={mstatus}（主源可用，不阻断）")
            else:
                m_ok = abs(mtotal - (total or declared)) <= max(1, max(mtotal, total or declared) * 0.05)
                print(f"    镜像: status={mstatus} 总大小={mtotal:,} {'✅' if m_ok else '[WARN] 大小偏差>5%'}")
    print()
    print("=== T3.3 下载源核对:", "PASS" if ok_all else "FAIL", "===")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
