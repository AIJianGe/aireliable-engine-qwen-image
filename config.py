"""Qwen-Image 图像引擎配置（v0.2.1+：引擎 ID qwen_image_5e8f3a21，由 FLUX 时代注册完整改名）。

从环境变量 + 命令行参数加载，提供引擎运行所需的全部配置。

v0.2.1 双源真相消除策略：
    - MODEL_SUBDIR 默认读环境变量 AIRELIABLE_MODEL_LAYOUT_BASE
      （由调度层 EngineManager 启动子进程时同步 manifest.models.model_layout.base）
    - 优先级：环境变量 > 硬编码 fallback
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ===== 常量 =====

ENGINE_ID = "qwen_image_5e8f3a21"
ENGINE_NAME = "QwenImage"
CAPABILITY = "text-to-image"
ENGINE_TYPE = "image"

# 默认端口（与 config/engines/qwen_image_5e8f3a21.yaml 一致）
DEFAULT_PORT = 7007

# 模型目录布局（相对 MODELS_ROOT）
# v0.2.1：优先读环境变量 AIRELIABLE_MODEL_LAYOUT_BASE
# （Qwen 迁移后默认 image/qwen-image/，见 manifest.yaml model_layout.base）
_DEFAULT_MODEL_SUBDIR = "image/qwen-image/"

# 设备策略合法取值（v0.3.5+ 设备策略，见 DEVICE_POLICY_GPU_CPU_OFFLOAD_ANALYSIS.md）
#   auto    ：引擎自行决定（现状行为：GPU 优先全常驻 + T5 异构 CPU）
#   gpu     ：强制全 GPU，显存不足直接报错
#   gpu+cpu ：GPU 优先，显存不足时组件溢出 CPU 内存（槽位换入换出）
#   cpu     ：强制 CPU（调试 / 无卡环境）
DEVICE_POLICIES = ("auto", "gpu", "gpu+cpu", "cpu")


def _resolve_model_subdir() -> str:
    """解析模型子目录（v0.2.1：优先读环境变量）。"""
    env_val = os.environ.get("AIRELIABLE_MODEL_LAYOUT_BASE")
    if env_val:
        return env_val.rstrip("/") + "/"
    return _DEFAULT_MODEL_SUBDIR


# 运行时常量
MODEL_SUBDIR = _resolve_model_subdir()


@dataclass
class EngineConfig:
    """Qwen-Image 图像引擎运行配置。

    通过环境变量 + 命令行参数构造。
    """

    # ===== 基础 =====
    engine_id: str = ENGINE_ID
    engine_name: str = ENGINE_NAME
    capability: str = CAPABILITY
    engine_type: str = ENGINE_TYPE
    port: int = DEFAULT_PORT
    host: str = "127.0.0.1"

    # ===== 路径 =====
    models_root: Path = Path("models")
    data_root: Path = Path("data")
    project_root: Path = Path(".")
    # v0.2.5+ 输出文件根目录统一为 output（项目顶层目录）
    # 规范：{project_root}/output/{category}/{filename}
    output_root: Path = Path("output")

    # ===== 推导路径 =====
    model_dir: Path = field(init=False)
    lora_dir: Path = field(init=False)
    log_dir: Path = field(init=False)
    input_dir: Path = field(init=False)

    # ===== 设备 =====
    # auto: 自动检测 GPU / cpu: 强制 CPU / cuda:0: 指定 GPU
    device: str = "auto"

    # ===== 设备策略（v0.3.5+）=====
    # auto / gpu / gpu+cpu / cpu，见 DEVICE_POLICIES 注释。
    # 平台通过 AIRELIABLE_DEVICE_POLICY 环境变量注入（UI 设置 > yaml > 默认 auto）。
    device_policy: str = "auto"

    # ===== 日志 =====
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """推导子路径。"""
        # 模型相关
        self.model_dir = self.models_root / MODEL_SUBDIR
        # LoRA 目录（AIRELIABLE_LORA_DIR 环境变量优先）
        # 平台契约（v0.2.1，与 llama_cpp 一致）：注入值是相对 models_root 的子目录
        # （如 "image/qwen-image/loras"）；绝对路径（手动调试场景）则直接使用
        lora_env = os.environ.get("AIRELIABLE_LORA_DIR")
        if lora_env:
            p = Path(lora_env)
            self.lora_dir = p if p.is_absolute() else self.models_root / p
        else:
            self.lora_dir = self.model_dir / "loras"

        # 日志目录（v0.2.15+ 统一到 logs/engines/）
        self.log_dir = self.project_root / "logs" / "engines"

        # 输入 staging 目录（多图参考 / 编辑源图落盘）
        self.input_dir = self.data_root / "inputs" / self.engine_id

        # 确保运行时目录存在
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)

    def resolve_device(self) -> str:
        """解析推理设备（device="auto" 在 load() 入口统一解析）。

        v0.3.5+：device_policy=cpu 时强制 CPU（策略优先于 auto 检测）；
        其余策略（auto/gpu/gpu+cpu）不影响设备解析，由 TransformerManager
        按 gpu+cpu 策略执行槽位放置。

        Returns:
            "cuda" / "cpu"（cuda 时返回 "cuda"，具体卡由 torch 默认）
        """
        if self.device_policy == "cpu":
            return "cpu"
        if self.device in ("cpu",):
            return "cpu"
        if self.device.startswith("cuda"):
            return self.device
        # auto
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def to_dict(self) -> dict:
        """转 dict（用于日志/调试）。"""
        return {
            "engine_id": self.engine_id,
            "engine_name": self.engine_name,
            "capability": self.capability,
            "engine_type": self.engine_type,
            "port": self.port,
            "host": self.host,
            "models_root": str(self.models_root),
            "model_dir": str(self.model_dir),
            "lora_dir": str(self.lora_dir),
            "output_root": str(self.output_root),
            "log_dir": str(self.log_dir),
            "input_dir": str(self.input_dir),
            "device": self.device,
            "device_policy": self.device_policy,
            "log_level": self.log_level,
        }


def parse_args() -> EngineConfig:
    """解析命令行参数 + 环境变量，构造 EngineConfig。

    优先级：命令行 > 环境变量 > 默认值
    """
    parser = argparse.ArgumentParser(description="Qwen-Image 图像生成引擎（t2i/edit/i2i）")

    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"HTTP 端口（默认 {DEFAULT_PORT}）")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="绑定地址（默认 127.0.0.1）")
    parser.add_argument("--device", type=str, default="auto",
                        help="推理设备：auto / cpu / cuda:0（默认 auto）")
    parser.add_argument("--device-policy", type=str, default=None,
                        help="设备策略：auto / gpu / gpu+cpu / cpu（默认 auto；"
                             "未传时读环境变量 AIRELIABLE_DEVICE_POLICY）")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="日志级别（默认 INFO）")
    parser.add_argument("--models-root", type=str, default=None,
                        help="模型根目录（默认 models/）")

    args = parser.parse_args()

    # 环境变量 fallback
    models_root = Path(args.models_root or os.environ.get("AIRELIABLE_MODELS_ROOT", "models"))
    project_root = Path(os.environ.get("AIRELIABLE_PROJECT_ROOT", ".")).resolve()
    output_root_env = os.environ.get("AIRELIABLE_OUTPUT_ROOT")
    output_root = Path(output_root_env) if output_root_env else project_root / "output"
    data_root = Path(os.environ.get("AIRELIABLE_DATA_ROOT", project_root / "data"))

    # 设备策略：命令行 > AIRELIABLE_DEVICE_POLICY 环境变量（平台注入）> 默认 auto
    device_policy = (
        args.device_policy
        or os.environ.get("AIRELIABLE_DEVICE_POLICY")
        or "auto"
    ).strip()
    if device_policy not in DEVICE_POLICIES:
        raise ValueError(
            f"非法设备策略：{device_policy!r}（合法值：{DEVICE_POLICIES}）。"
            f"来源：命令行 --device-policy 或环境变量 AIRELIABLE_DEVICE_POLICY。"
        )

    config = EngineConfig(
        port=args.port,
        host=args.host,
        device=args.device,
        device_policy=device_policy,
        log_level=args.log_level,
        models_root=models_root,
        project_root=project_root,
        output_root=output_root,
        data_root=data_root,
    )

    return config
