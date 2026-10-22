# -*- coding: utf-8 -*-
"""端到端测试：经 gateway 的四类请求（T8.3）。

验证链路：gateway OpenAI 端点 → router.resolve → SmartTaskScheduler →
引擎 /infer → SSE 进度 → 任务中心结果。

用例：
    1. generations-t2i      POST /v1/images/generations（无图 → t2i）
    2. edits-edit           POST /v1/images/edits（1 图 → edit）
    3. edits-多图×2         POST /v1/images/edits（2 图 → i2i 拼接）
    4. 带 LoRA              generations + extra.loras
    5. 档位切换             model=FLUX/Kontext-FP8（gateway variant → 引擎 precision）

运行前提：
    - gateway 8000 / core 8001 / 引擎 7007 均在运行
    - 引擎已 load（默认 Q8 档）
    - output/image/ 下有历史测试图可作为 edits 输入

用法：
    cd D:\\myProInstallDir\\AIReliable
    .venv\\Scripts\\python.exe engines\\qwen_image_5e8f3a21\\_test_e2e.py
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

# 复用历史测试图作为 edits 输入（引擎同机，path 直读）
EDIT_SRC = OUT_IMAGE_DIR / "test_i2i_001.png"
I2I_SRC1 = OUT_IMAGE_DIR / "test_i2i_001.png"
I2I_SRC2 = OUT_IMAGE_DIR / "test_edit_001.png"

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_engine_ready(client: httpx.Client, timeout: float = 60.0) -> None:
    """等待引擎就绪（load 完成）。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = client.get(f"{ENGINE}/status", timeout=5)
            st = r.json()
            if st.get("state") in ("ready", "inferring"):
                log(f"引擎就绪：{st['state']}")
                return
            log(f"引擎状态 {st.get('state')}，等待…")
        except Exception:
            log("引擎不可达，等待…")
        time.sleep(5)
    raise RuntimeError("引擎未就绪（60s 超时）")


def submit_and_wait(
    client: httpx.Client,
    url: str,
    body: dict,
    name: str,
    timeout: float = 900.0,
) -> dict | None:
    """提交 gateway 任务并轮询到终态，返回 result dict（失败返回 None）。"""
    log(f"--- {name}：POST {url}")
    log(f"    body keys: {list(body.keys())}, model={body.get('model')}")
    r = client.post(url, json=body, timeout=30)
    if r.status_code != 200:
        log(f"    {FAIL} 提交失败 HTTP {r.status_code}: {r.text[:300]}")
        return None
    task = r.json()
    task_id = task.get("task_id")
    log(f"    task_id={task_id} status={task.get('status')}")

    t0 = time.time()
    last_stage = ""
    while time.time() - t0 < timeout:
        tr = client.get(f"{GATEWAY}/v1/tasks/{task_id}", timeout=10)
        rec = tr.json()
        status = rec.get("status")
        stage = rec.get("stage", "")
        progress = rec.get("progress", 0.0)
        if stage != last_stage:
            log(f"    [{status}] {stage} {progress:.0%}")
            last_stage = stage
        if status == "succeeded":
            result = rec.get("result") or {}
            imgs = result.get("images") or []
            log(f"    {PASS} 成功，images={len(imgs)}")
            for i, im in enumerate(imgs):
                log(f"      [{i}] url={im.get('url', '')[:120]}")
            # 完整 result 帧也拉一次
            rr = client.get(f"{GATEWAY}/v1/tasks/{task_id}/result", timeout=10)
            log(f"    /result HTTP {rr.status_code}")
            return result
        if status in ("failed", "cancelled"):
            log(f"    {FAIL} {status}: {json.dumps(rec.get('error', ''), ensure_ascii=False)[:500]}")
            return None
        time.sleep(3)
    log(f"    {FAIL} 轮询超时（{timeout}s）")
    return None


def main() -> int:
    results: dict[str, bool] = {}

    with httpx.Client() as client:
        # 前置检查
        wait_engine_ready(client)

        # ============ 用例 1：generations-t2i ============
        r = submit_and_wait(
            client,
            f"{GATEWAY}/v1/images/generations",
            {
                "model": "FLUX/Kontext-GGUF-Q8",
                "prompt": "a cute orange cat wearing a tiny wizard hat, studio lighting, 4k",
                "n": 1,
                "size": "512x512",
                "extra": {
                    "steps": 8,
                    "guidance_scale": 2.5,
                    "seed": 42,
                },
            },
            "用例1 generations-t2i",
        )
        results["t2i"] = bool(r and r.get("images"))

        # ============ 用例 2：edits-edit（1 图）============
        if EDIT_SRC.exists():
            r = submit_and_wait(
                client,
                f"{GATEWAY}/v1/images/edits",
                {
                    "model": "FLUX/Kontext-GGUF-Q8",
                    "prompt": "make it snow in the background, winter atmosphere",
                    "n": 1,
                    "size": "512x512",
                    "extra": {
                        "steps": 8,
                        "seed": 43,
                        "images": [{"path": str(EDIT_SRC), "role": "source"}],
                    },
                },
                "用例2 edits-edit（单图指令编辑）",
            )
            results["edit"] = bool(r and r.get("images"))
        else:
            log(f"跳过用例2：输入图不存在 {EDIT_SRC}")
            results["edit"] = False

        # ============ 用例 3：edits-多图×2（i2i 拼接）============
        if I2I_SRC1.exists() and I2I_SRC2.exists():
            r = submit_and_wait(
                client,
                f"{GATEWAY}/v1/images/edits",
                {
                    "model": "FLUX/Kontext-GGUF-Q8",
                    "prompt": "combine both references into one cinematic scene",
                    "n": 1,
                    "size": "512x512",
                    "extra": {
                        "steps": 8,
                        "seed": 44,
                        "images": [
                            {"path": str(I2I_SRC1), "role": "reference"},
                            {"path": str(I2I_SRC2), "role": "reference"},
                        ],
                    },
                },
                "用例3 edits-多图参考（2 图拼接 i2i）",
            )
            results["i2i"] = bool(r and r.get("images"))
        else:
            log(f"跳过用例3：输入图不存在 {I2I_SRC1} / {I2I_SRC2}")
            results["i2i"] = False

        # ============ 用例 4：带 LoRA ============
        lora_status = client.get(f"{ENGINE}/status", timeout=5).json()
        loaded = (lora_status.get("extra") or {}).get("loaded_loras") or lora_status.get("loaded_loras") or []
        avail = client.get(f"{ENGINE}/capabilities", timeout=5).json().get("loras", [])
        if avail:
            lora_name = avail[0]
            log(f"引擎可用 LoRA：{avail}（已挂载：{loaded}）")
            r = submit_and_wait(
                client,
                f"{GATEWAY}/v1/images/generations",
                {
                    "model": "FLUX/Kontext-GGUF-Q8",
                    "prompt": "a golden retriever puppy playing in autumn leaves",
                    "n": 1,
                    "size": "512x512",
                    "extra": {
                        "steps": 8,
                        "seed": 45,
                        "loras": [{"name": lora_name, "weight": 0.8}],
                    },
                },
                f"用例4 带 LoRA（{lora_name}@0.8）",
            )
            # 验证 LoRA 挂载状态
            st = client.get(f"{ENGINE}/status", timeout=5).json()
            loaded_after = (st.get("extra") or {}).get("loaded_loras") or st.get("loaded_loras") or []
            log(f"    推理后引擎已挂载 LoRA：{loaded_after}")
            results["lora"] = bool(r and r.get("images")) and lora_name in loaded_after
        else:
            log("跳过用例4：引擎无可用 LoRA")
            results["lora"] = False

        # ============ 用例 5：gateway variant 档位切换（FP8）============
        # model=FLUX/Kontext-FP8 → 引擎切换到 fp8 档（SDK auto-load 完整换档）
        # 用 edits（edit 模式）：只用 kontext 侧 FP8 transformer（在盘）；
        # t2i 侧 dev-fp8（11.9GB）未下载，会正确报缺失指引下载（已在首轮实测验证）
        r = submit_and_wait(
            client,
            f"{GATEWAY}/v1/images/edits",
            {
                "model": "FLUX/Kontext-FP8",
                "prompt": "make it a snowy winter scene",
                "n": 1,
                "size": "512x512",
                "extra": {
                    "steps": 6,
                    "seed": 46,
                    "images": [{"path": str(EDIT_SRC), "role": "source"}],
                },
            },
            "用例5 档位热切换（model=FLUX/Kontext-FP8，edit 模式）",
            timeout=1200.0,  # 完整换档（unload+load FP8）约 220s + 推理
        )
        # 验证引擎当前精度
        st = client.get(f"{ENGINE}/status", timeout=5).json()
        precision = (st.get("extra") or {}).get("precision") or st.get("precision")
        log(f"    推理后引擎精度档：{precision}")
        results["variant_swap"] = bool(r and r.get("images")) and precision == "fp8"

        # 切回默认档（后续使用）
        log("切回默认档 Q8（load 请求后台执行）…")
        try:
            client.post(f"{ENGINE}/load", json={"variant": "FLUX/Kontext-GGUF-Q8"}, timeout=5)
        except Exception:
            pass  # 客户端超时无妨，服务器继续

    # ============ 汇总 ============
    print("\n" + "=" * 60)
    print("端到端测试汇总（T8.3）")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {PASS if ok else FAIL}  {name}")
    n_pass = sum(1 for v in results.values() if v)
    print(f"\n{n_pass}/{len(results)} 通过")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
