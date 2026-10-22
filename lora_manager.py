"""LoRA 管理器：多 LoRA 扫描、叠加加载、运行时切换、transformer 切换重挂。

设计（分析文档 §2 决策 6）：
    - 扫描 lora_dir 下 *.safetensors（manifest loras.format: safetensors）
    - 多 LoRA 叠加：diffusers/PEFT adapter 机制（adapter_name = LoRA 文件名）
      pipe.load_lora_weights(path, adapter_name=...) + pipe.set_adapters(...)
    - 运行时切换：调整 adapter 权重（set_adapters 不重载文件）或动态增删
    - transformer 热切换后：LoRA 绑定在新 transformer 对象上，需要全部重挂
      （engine 的 on_swap 回调 → reapply()）

Qwen 迁移注意（T5.6 / R7）：
    - PEFT 加载路径对 Qwen transformer 通用（diffusers QwenImageLoraLoaderMixin，
      adapter 注入 transformer 的 nn.Linear 层，GGUF Q8 dequant 权重同样可挂）
    - 旧 FLUX LoRA（dev/Kontext 版，键名 transformer.* 但结构不匹配 Qwen）
      加载时 PEFT 抛键名不匹配错误 → try/except 记 warning 不阻断；
      scan 时对疑似 FLUX 命名（flux/kontext）的文件主动提示失效（R7）
    - t2i/edit 两个 pipeline 的 transformer 是不同对象，各自挂各自的
      （重挂时只挂到当前活跃 transformer）
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LoraManager:
    """多 LoRA 管理器。"""

    def __init__(self, lora_dir: Path):
        self.lora_dir = lora_dir
        # LoRA 名 → 权重（运行时状态，transformer 切换后按此重挂）
        self._loaded: dict[str, float] = {}

    # ===== 状态查询 =====

    @property
    def loaded_loras(self) -> dict[str, float]:
        return dict(self._loaded)

    def scan_loras(self) -> list[str]:
        """扫描 LoRA 目录，返回可用 LoRA 名列表（文件名去后缀）。"""
        if not self.lora_dir.exists():
            return []
        names = sorted(p.stem for p in self.lora_dir.glob("*.safetensors"))
        # R7：旧 FLUX LoRA 与 Qwen transformer 键名不兼容，主动提示失效
        legacy = [n for n in names if "flux" in n.lower() or "kontext" in n.lower()]
        if legacy:
            logger.warning(
                "检测到疑似旧 FLUX LoRA（%s）：与 Qwen-Image transformer 键名不兼容，"
                "挂载会失败；请改用 Qwen-Image 生态 LoRA", legacy,
            )
        logger.info("LoRA 扫描：%s（%d 个）", self.lora_dir, len(names))
        return names

    def lora_path(self, name: str) -> Path:
        return self.lora_dir / f"{name}.safetensors"

    # ===== 加载 / 卸载 =====

    def load_lora(self, pipe: Any, name: str, weight: float = 1.0) -> dict[str, float]:
        """加载单个 LoRA 到当前 pipeline 的 transformer（叠加，不卸载已有的）。

        Args:
            pipe: 当前活跃的 diffusers pipeline（load_lora_weights 入口）
            name: LoRA 名（文件名去后缀）
            weight: 叠加权重
        """
        path = self.lora_path(name)
        if not path.exists():
            available = self.scan_loras()
            raise FileNotFoundError(
                f"LoRA 文件不存在: {path}（可用: {available}）"
            )
        if pipe is None or pipe.transformer is None:
            raise RuntimeError("transformer 未加载，无法挂 LoRA")

        logger.info("加载 LoRA：%s（weight=%.2f）", name, weight)
        try:
            if name in self._loaded:
                # 已加载 → 只调权重（不重读文件）
                self._loaded[name] = weight
                self._apply_weights(pipe)
            else:
                pipe.load_lora_weights(str(path), adapter_name=name)
                self._loaded[name] = weight
                self._apply_weights(pipe)
        except Exception as e:
            logger.warning("LoRA %s 加载失败（键名与当前 transformer 不匹配？）: %s", name, e)
            raise
        return self.loaded_loras

    def unload_lora(self, pipe: Any, name: str) -> dict[str, float]:
        """卸载单个 LoRA。"""
        if name not in self._loaded:
            logger.warning("LoRA %s 未加载（当前: %s）", name, list(self._loaded))
            return self.loaded_loras
        logger.info("卸载 LoRA：%s", name)
        del self._loaded[name]
        if pipe is not None:
            try:
                pipe.delete_adapters(name)
            except Exception as e:  # noqa: BLE001
                # adapter 不在当前 transformer 上（如切换后），忽略
                logger.debug("delete_adapters(%s) 跳过: %s", name, e)
            self._apply_weights(pipe)
        return self.loaded_loras

    def clear(self) -> None:
        """清空运行时记录（不操作 pipeline——transformer 已释放的场景，如引擎 unload）。

        与 unload_all 的区别：unload_all 需要活着的 pipe 做 delete_adapters；
        clear 只重置记录，供 engine._unload_impl 在释放 transformer 后调用。
        """
        if self._loaded:
            logger.info("LoRA 运行时记录清空：%s", list(self._loaded))
        self._loaded.clear()

    def unload_all(self, pipe: Any) -> None:
        """卸载全部 LoRA。"""
        names = list(self._loaded.keys())
        self._loaded.clear()
        if pipe is not None and names:
            try:
                pipe.delete_adapters(names)
            except Exception as e:  # noqa: BLE001
                logger.debug("delete_adapters 跳过: %s", e)
        logger.info("LoRA 全部卸载：%s", names)

    def set_weight(self, pipe: Any, name: str, weight: float) -> dict[str, float]:
        """运行时调整单个 LoRA 权重（不重载文件）。"""
        if name not in self._loaded:
            raise KeyError(f"LoRA {name} 未加载")
        self._loaded[name] = weight
        self._apply_weights(pipe)
        return self.loaded_loras

    # ===== 请求级集合同步 =====

    def sync(self, pipe: Any, desired: dict[str, float]) -> dict[str, float]:
        """把挂载集合收敛到目标集合（请求级 LoRA diff 同步）。

        供 gateway 请求（extra.loras → InferRequest.loras）在推理前调用：
            - 卸载多余：不在目标集合中的已挂 LoRA
            - 加载缺失：目标集合中的新 LoRA（pipe 就绪时直接挂载）
            - 调整权重：已在挂但权重变化的 LoRA（不重读文件）

        pipe 为 None（引擎刚 load、尚未推理过）时仅更新期望记录，
        generate() 组装管线后按 transformer 身份变化 reapply 统一挂载。

        Raises:
            FileNotFoundError: 目标 LoRA 文件不在盘（明确指引可用列表）
        """
        # 盘上不存在 → 直接报错（请求参数问题，不静默忽略）
        available = set(self.scan_loras())
        missing = sorted(n for n in desired if n not in available)
        if missing:
            raise FileNotFoundError(
                f"请求的 LoRA 不存在: {missing}（可用: {sorted(available)}）"
            )

        current = dict(self._loaded)

        # 1) 卸载多余（pipe=None 时 unload_lora 只清记录）
        for name in [n for n in current if n not in desired]:
            self.unload_lora(pipe, name)

        # 2) 加载缺失 / 调整权重
        for name, weight in desired.items():
            if name not in current:
                if pipe is not None and getattr(pipe, "transformer", None) is not None:
                    self.load_lora(pipe, name, weight)
                else:
                    logger.info(
                        "LoRA %s 记录期望状态（%.2f，管线未组装，generate 时统一挂载）",
                        name, weight,
                    )
                    self._loaded[name] = weight
            elif current[name] != weight:
                self._loaded[name] = weight
                if pipe is not None:
                    self._apply_weights(pipe)

        logger.info(
            "LoRA 集合同步完成：%s → %s",
            current, dict(self._loaded),
        )
        return self.loaded_loras

    # ===== transformer 切换重挂 =====

    def reapply(self, pipe: Any) -> dict[str, float]:
        """transformer 热切换后重挂全部 LoRA（新 transformer 对象上）。

        - adapter 已在目标 transformer 上（如 t2i→edit→t2i 往返回到曾挂过的
          transformer）：跳过重载（重载会抛 "Adapter name already in use"），
          保留运行时记录并刷新权重
        - LoRA 与 transformer 槽位不匹配（如 Qwen-2512 LoRA 挂 Edit-2509
          transformer，或残留 FLUX LoRA）时单个失败仅记 warning，不阻断
          （分析文档决策 6 的容错策略）
        """
        if not self._loaded:
            return {}
        if pipe is None or pipe.transformer is None:
            logger.warning("reapply 跳过：pipeline/transformer 未就绪")
            return {}

        # 目标 transformer 上已有的 adapter 名（PEFT 注入后 peft_config 按名索引）
        existing = set(getattr(pipe.transformer, "peft_config", {}) or {})

        reloaded: dict[str, float] = {}
        for name, weight in self._loaded.items():
            if name in existing:
                logger.info("LoRA %s 已在当前 transformer 上，跳过重载（仅刷新权重）", name)
                reloaded[name] = weight
                continue
            path = self.lora_path(name)
            try:
                pipe.load_lora_weights(str(path), adapter_name=name)
                reloaded[name] = weight
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "LoRA %s 重挂失败（与当前 transformer 档位不匹配？跳过）: %s", name, e
                )
        self._loaded = reloaded
        if reloaded:
            self._apply_weights(pipe)
        logger.info("LoRA 重挂完成：%s", reloaded)
        return self.loaded_loras

    # ===== 内部 =====

    def _apply_weights(self, pipe: Any) -> None:
        """把当前 LoRA 集合与权重应用到 pipeline（set_adapters）。"""
        if not self._loaded:
            return
        names = list(self._loaded.keys())
        weights = [self._loaded[n] for n in names]
        try:
            pipe.set_adapters(names, adapter_weights=weights)
            logger.info("LoRA 权重生效：%s", dict(zip(names, weights)))
        except Exception as e:  # noqa: BLE001
            logger.warning("set_adapters 失败: %s", e)
