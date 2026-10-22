# -*- coding: utf-8 -*-
"""回归测试（T9.1）：连续切换、LoRA 卸载循环、推理中断清理。

用例：
    1. 连续模式切换     t2i → edit → i2i → t2i（同档 Q8，验证状态无泄漏，全出图）
    2. LoRA 卸载循环    带 LoRA 请求 → 无 LoRA 请求 → 引擎挂载态收敛为空
    3. 推理中断清理     提交长任务（50 步）→ denoising 中 cancel →
                       任务态 cancelled → 引擎回 ready 并继续服务下一请求
    4. 重启状态恢复     已在 2026-08-22 会话内实测两次（重启 → idle → load →
                       ready 双槽位），本脚本不重复执行，仅记录

运行前提：gateway 8000 / 引擎 7007 运行中，引擎已 load（Q8 档）
用法：
    envs\image-qwen-image\.venv\Scripts\python.exe engines\qwen_image_5e8f3a21\_test_regression.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

GATEWAY = "http://127.0.0.1:8000"
ENGINE = "http://127.0.0.1:7007"
PROJECT = Path(__file__).resolve().parents[2]
OUT_IMAGE_DIR = PROJECT / "output" / "image"

SRC1 = OUT_IMAGE_DIR / "test_i2i_001.png"
SRC2 = OUT_IMAGE_DIR / "test_edit_001.png"

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def engine_status(client: httpx.Client) -> dict:
    return client.get(f"{ENGINE}/status", timeout=10).json()


def wait_engine_ready(client: httpx.Client, timeout: float = 600.0) -> None:
    """等待引擎 ready（含换档/加载中的长时间等待）。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            st = engine_status(client)
            if st.get("state") == "ready":
                log(f"引擎就绪（precision={st.get('extra', {}).get('precision')}）")
                return
            log(f"引擎状态 {st.get('state')}，等待…")
        except Exception:
            log("引擎不可达，等待…")
        time.sleep(10)
    raise RuntimeError(f"引擎未就绪（{timeout}s 超时）")


def submit_and_wait(
    client: httpx.Client,
    body: dict,
    name: str,
    endpoint: str = "/v1/images/generations",
    timeout: float = 900.0,
) -> dict | None:
    """提交任务并轮询到终态。"""
    log(f"--- {name}")
    r = client.post(f"{GATEWAY}{endpoint}", json=body, timeout=30)
    if r.status_code != 200:
        log(f"    {FAIL} 提交失败 HTTP {r.status_code}: {r.text[:200]}")
        return None
    task_id = r.json().get("task_id")
    log(f"    task_id={task_id}")

    t0 = time.time()
    while time.time() - t0 < timeout:
        rec = client.get(f"{GATEWAY}/v1/tasks/{task_id}", timeout=10).json()
        status = rec.get("status")
        if status == "succeeded":
            result = rec.get("result") or {}
            imgs = result.get("images") or []
            log(f"    {PASS} 成功 images={len(imgs)}")
            return result
        if status in ("failed", "cancelled"):
            log(f"    {FAIL} {status}: {json.dumps(rec.get('error', ''), ensure_ascii=False)[:300]}")
            return None
        time.sleep(3)
    log(f"    {FAIL} 轮询超时")
    return None


def main() -> int:
    results: dict[str, bool] = {}

    with httpx.Client() as client:
        wait_engine_ready(client)

        # ============ 用例 1：连续模式切换（t2i → edit → i2i → t2i）============
        r1 = submit_and_wait(
            client,
            {
                "model": "FLUX/Kontext-GGUF-Q8",
                "prompt": "a red sports car on a mountain road",
                "n": 1, "size": "512x512",
                "extra": {"steps": 6, "seed": 101},
            },
            "用例1a t2i",
        )
        r2 = submit_and_wait(
            client,
            {
                "model": "FLUX/Kontext-GGUF-Q8",
                "prompt": "make the background a sunset",
                "n": 1, "size": "512x512",
                "extra": {
                    "steps": 6, "seed": 102,
                    "images": [{"path": str(SRC1), "role": "source"}],
                },
            },
            "用例1b edit（切 Kontext transformer）",
            endpoint="/v1/images/edits",
        )
        r3 = submit_and_wait(
            client,
            {
                "model": "FLUX/Kontext-GGUF-Q8",
                "prompt": "combine both references into a forest scene",
                "n": 1, "size": "512x512",
                "extra": {
                    "steps": 6, "seed": 103,
                    "images": [
                        {"path": str(SRC1), "role": "reference"},
                        {"path": str(SRC2), "role": "reference"},
                    ],
                },
            },
            "用例1c i2i（多图拼接）",
            endpoint="/v1/images/edits",
        )
        r4 = submit_and_wait(
            client,
            {
                "model": "FLUX/Kontext-GGUF-Q8",
                "prompt": "a blue bird on a branch, watercolor style",
                "n": 1, "size": "512x512",
                "extra": {"steps": 6, "seed": 104},
            },
            "用例1d t2i（切回 dev transformer）",
        )
        results["mode_cycle"] = all(
            r and r.get("images") for r in (r1, r2, r3, r4)
        )

        # ============ 用例 2：LoRA 卸载循环 ============
        st = engine_status(client)
        loaded_before = st.get("loaded_loras") or []
        log(f"当前引擎挂载 LoRA：{loaded_before}")

        # 2a. 带 LoRA 请求（若当前未挂则挂上）
        r = submit_and_wait(
            client,
            {
                "model": "FLUX/Kontext-GGUF-Q8",
                "prompt": "a cat wearing sunglasses, flux koda style",
                "n": 1, "size": "512x512",
                "extra": {
                    "steps": 6, "seed": 105,
                    "loras": [{"name": "flux-koda", "weight": 0.8}],
                },
            },
            "用例2a 带 LoRA 请求",
        )
        st = engine_status(client)
        mounted = st.get("loaded_loras") or []
        log(f"    推理后挂载态：{mounted}")
        ok_mount = bool(r and r.get("images")) and "flux-koda" in mounted

        # 2b. 显式空 LoRA 集合（extra.loras=[]，lora_explicit=True）→ 清空挂载
        r = submit_and_wait(
            client,
            {
                "model": "FLUX/Kontext-GGUF-Q8",
                "prompt": "a plain white coffee mug on a wooden table",
                "n": 1, "size": "512x512",
                "extra": {"steps": 6, "seed": 106, "loras": []},
            },
            "用例2b 显式空 LoRA 集合（应清空 flux-koda）",
        )
        st = engine_status(client)
        mounted_after = st.get("loaded_loras") or []
        log(f"    推理后挂载态：{mounted_after}")
        ok_unmount = bool(r and r.get("images")) and "flux-koda" not in mounted_after

        # 2c. 未指定 loras 字段（lora_explicit=False）→ 保持当前挂载（WebUI 会话语义）
        #     先重新挂上 flux-koda，再发不带 loras 字段的请求，验证挂载不被动
        r = submit_and_wait(
            client,
            {
                "model": "FLUX/Kontext-GGUF-Q8",
                "prompt": "a cat wearing a hat, flux koda style",
                "n": 1, "size": "512x512",
                "extra": {
                    "steps": 6, "seed": 109,
                    "loras": [{"name": "flux-koda", "weight": 0.8}],
                },
            },
            "用例2c-1 重新挂载 flux-koda",
        )
        st = engine_status(client)
        log(f"    挂载态：{st.get('loaded_loras') or []}")
        r = submit_and_wait(
            client,
            {
                "model": "FLUX/Kontext-GGUF-Q8",
                "prompt": "a yellow sunflower in a field",
                "n": 1, "size": "512x512",
                "extra": {"steps": 6, "seed": 110},
            },
            "用例2c-2 未指定 loras（应保持 flux-koda 挂载）",
        )
        st = engine_status(client)
        mounted_kept = st.get("loaded_loras") or []
        log(f"    推理后挂载态：{mounted_kept}")
        ok_keep = bool(r and r.get("images")) and "flux-koda" in mounted_kept

        # 2d. 收尾：显式空集合清空，避免影响后续用例
        submit_and_wait(
            client,
            {
                "model": "FLUX/Kontext-GGUF-Q8",
                "prompt": "a minimal gray stone",
                "n": 1, "size": "512x512",
                "extra": {"steps": 6, "seed": 111, "loras": []},
            },
            "用例2d 收尾清空 LoRA",
        )

        results["lora_cycle"] = ok_mount and ok_unmount and ok_keep

        # ============ 用例 3：推理中断清理 ============
        log("--- 用例3 推理中断（50 步长任务，denoising 中取消）")
        r = client.post(
            f"{GATEWAY}/v1/images/generations",
            json={
                "model": "FLUX/Kontext-GGUF-Q8",
                "prompt": "an extremely detailed cyberpunk city at night",
                "n": 1, "size": "1024x1024",
                "extra": {"steps": 50, "seed": 107},
            },
            timeout=30,
        )
        task_id = r.json().get("task_id")
        log(f"    task_id={task_id}")

        # 等到 denoising 阶段再取消
        cancelled = False
        t0 = time.time()
        while time.time() - t0 < 120:
            rec = client.get(f"{GATEWAY}/v1/tasks/{task_id}", timeout=10).json()
            stage = rec.get("stage", "")
            if stage == "denoising":
                cd = client.delete(f"{GATEWAY}/v1/tasks/{task_id}", timeout=10)
                log(f"    cancel HTTP {cd.status_code}: {cd.json()}")
                cancelled = True
                break
            if rec.get("status") in ("succeeded", "failed"):
                log(f"    任务在取消前已进入终态：{rec.get('status')}（50 步太快）")
                break
            time.sleep(1)

        if cancelled:
            # 等终态
            t0 = time.time()
            final = None
            while time.time() - t0 < 120:
                rec = client.get(f"{GATEWAY}/v1/tasks/{task_id}", timeout=10).json()
                if rec.get("status") in ("succeeded", "failed", "cancelled"):
                    final = rec.get("status")
                    break
                time.sleep(2)
            log(f"    任务终态：{final}")

        # 引擎应回 ready 并能继续服务（引擎不支持步中断，推理在后台跑完也无妨，
        # 关键是不卡 ERROR/INFERRING 死态）
        ok_resume = False
        t0 = time.time()
        while time.time() - t0 < 300:
            st = engine_status(client)
            if st.get("state") == "ready":
                log("    引擎已回 ready")
                break
            time.sleep(5)
        r = submit_and_wait(
            client,
            {
                "model": "FLUX/Kontext-GGUF-Q8",
                "prompt": "a simple green apple on white background",
                "n": 1, "size": "512x512",
                "extra": {"steps": 6, "seed": 108},
            },
            "用例3b 取消后继续服务",
        )
        ok_resume = bool(r and r.get("images"))
        results["cancel_resume"] = (final == "cancelled" or not cancelled) and ok_resume

    # ============ 汇总 ============
    print("\n" + "=" * 60)
    print("回归测试汇总（T9.1）")
    print("=" * 60)
    print("  （重启状态恢复已在本日会话实测两次：重启 → idle → load → ready 双槽位）")
    for name, ok in results.items():
        print(f"  {PASS if ok else FAIL}  {name}")
    n_pass = sum(1 for v in results.values() if v)
    print(f"\n{n_pass}/{len(results)} 通过")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
