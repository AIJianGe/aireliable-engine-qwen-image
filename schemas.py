"""Qwen-Image 图像引擎输入/输出 schema（FLUX Kontext → Qwen 迁移，2026-08）。

请求输入（ImageInputs，经 InferRequest.inputs 传入）：
    - prompt / negative_prompt（true_cfg_scale > 1 时负向提示词生效）
    - mode: t2i / i2i / edit（省略时按 images 数量推断）
    - width / height: 输出尺寸（16 对齐；edit 模式默认跟随输入图）
    - num_inference_steps / true_cfg_scale / guidance_scale / seed
    - true_cfg_scale: 真 CFG 强度（Qwen 语义，默认 4.0；与 FLUX 的
      embedded guidance 完全不同）
    - guidance_scale: norm rescaling（Qwen 语义，默认 1.0，一般不动）
    - max_sequence_length: VL 编码长度（默认 512，上限 1024）
    - view: edit 模式视角控制（{horizontal, vertical, distance}，
      全默认值 = 不拼视角描述，纯指令编辑；分析文档 §4.2.1）

模式判定：
    inputs.mode 显式 > images 推断（0 图→t2i；1 图→edit；≥2 图→i2i）> 兜底 t2i
    （i2i 上限 5 张：官方训练分布 1-3 张，4-5 张管线可跑但指令遵循度下降）
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

Mode = Literal["t2i", "i2i", "edit"]
Precision = Literal["gguf_q8"]

# edit 模式视角控制取值（WebUI 控件 ↔ prompt 翻译，分析文档 §4.2.1 翻译表）
Horizontal = Literal["front", "left45", "left90", "right45", "right90"]
Vertical = Literal["level", "up30", "down30", "down60"]
Distance = Literal["medium", "close", "extreme", "far"]


class ViewSpec(BaseModel):
    """视角控制参数（edit 模式专属；全默认值 = 无视角控制）。

    依据官方 multiple-angles 能力（WaveSpeed 参数化规范）：视角枚举 →
    prompt 英文前缀拼接，零额外模型文件。
    """

    horizontal: Horizontal = "front"
    """水平角度：正面 0°（默认）/ 左 45° / 左 90° / 右 45° / 右 90°"""

    vertical: Vertical = "level"
    """俯仰角度：平视 0°（默认）/ 仰视 -30° / 俯视 30° / 高角俯视 60°"""

    distance: Distance = "medium"
    """镜头距离：中景（默认）/ 近景 / 特写（超近）/ 远景"""

    @property
    def has_any(self) -> bool:
        """是否任一维度偏离默认值（决定是否拼接视角前缀）。"""
        return not (self.horizontal == "front" and self.vertical == "level" and self.distance == "medium")


class ImageInputs(BaseModel):
    """推理输入参数（/infer 的 inputs 字段）。"""

    prompt: str = ""
    # gateway /v1/images/* 构造 inputs 时 negative_prompt 取 extra.get()，
    # 未提供时为 None（str 类型校验会失败）→ Optional 兼容
    # （Qwen 管线在 true_cfg_scale > 1 时消费 negative_prompt）
    negative_prompt: Optional[str] = None

    mode: Optional[Mode] = None
    """生成模式：t2i / i2i / edit。None 时按 images 数量推断。"""

    # 输出尺寸（None 时按模式默认：t2i 1328x1328；edit 跟随输入；i2i 1328x1328）
    width: Optional[int] = None
    height: Optional[int] = None

    num_inference_steps: int = 28
    true_cfg_scale: float = 4.0
    """真 CFG 强度（Qwen 语义；官方推荐 4.0；1.0 = 关闭负向引导）"""

    guidance_scale: float = 1.0
    """norm rescaling（Qwen 语义，默认 1.0；与 FLUX guidance 不同，一般不动）"""

    seed: Optional[int] = None

    # 运行时精度覆盖（None 用 variant 档位；Qwen 迁移后仅 gguf_q8）
    precision: Optional[Precision] = None

    max_sequence_length: int = 512

    # edit 模式视角控制（t2i/i2i 忽略；见 ViewSpec）
    view: Optional[ViewSpec] = None

    @field_validator("num_inference_steps")
    @classmethod
    def _check_steps(cls, v: int) -> int:
        if v < 1 or v > 100:
            raise ValueError(f"num_inference_steps 越界（1-100）：{v}")
        return v

    @field_validator("true_cfg_scale")
    @classmethod
    def _check_true_cfg(cls, v: float) -> float:
        if v < 0 or v > 20:
            raise ValueError(f"true_cfg_scale 越界（0-20）：{v}")
        return v

    @field_validator("guidance_scale")
    @classmethod
    def _check_guidance(cls, v: float) -> float:
        if v < 0 or v > 20:
            raise ValueError(f"guidance_scale 越界（0-20）：{v}")
        return v

    @field_validator("width", "height")
    @classmethod
    def _check_size(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < 256 or v > 2048):
            raise ValueError(f"宽高越界（256-2048）：{v}")
        return v

    @field_validator("max_sequence_length")
    @classmethod
    def _check_seq_len(cls, v: int) -> int:
        if v < 64 or v > 1024:
            raise ValueError(f"max_sequence_length 越界（64-1024，Qwen 管线上限 1024）：{v}")
        return v


def resolve_mode(inputs: ImageInputs, image_count: int) -> Mode:
    """模式判定：显式 mode > images 数量推断 > 兜底 t2i。

    0 图 → t2i（QwenImagePipeline + Qwen-Image-2512）
    1 图 → edit（EditPlus 单图指令编辑 + 视角控制）
    ≥2 图 → i2i（EditPlus 原生多图参考，上限 5 张）
    """
    if inputs.mode is not None:
        return inputs.mode
    if image_count == 0:
        return "t2i"
    if image_count == 1:
        return "edit"
    return "i2i"


class HistoryItem(BaseModel):
    """生成历史条目（/history 端点返回，供 WebUI 结果区使用）。"""

    task_id: str
    mode: Mode
    prompt: str
    image_url: str
    thumbnail_url: Optional[str] = None
    width: int
    height: int
    seed: Optional[int] = None
    precision: Optional[Precision] = None
    loras: list[str] = Field(default_factory=list)
    created_at: float
    duration_seconds: float = 0.0


class HistoryResponse(BaseModel):
    """/history 响应。"""

    items: list[HistoryItem]
    total: int
    page: int
    page_size: int
