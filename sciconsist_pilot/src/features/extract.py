"""
VLM 特征提取模块 — 从 InternVL2.5 / Qwen2.5-VL 提取 multi-layer fused hidden states。

作为 FEH 的上游，负责将 (figure, claim_text) 对转化为固定维度的特征向量。
支持 cross-model (InternVL) 和 same-model (Qwen) 两种模式用于 A8 消融。

典型用法:
    extractor = FeatureExtractor(model_name="OpenGVLab/InternVL2_5-8B")
    h_visual, h_text = extractor.extract_pair(figure_image, claim_text)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """特征提取配置。

    Attributes:
        model_name: HuggingFace 模型名称
        fuse_layer_indices: multi-layer fusion 使用的层号 (0-indexed)
        batch_size: 批处理大小
        max_length: 文本最大 token 数
        device: 计算设备
        cache_dir: 特征缓存目录
        use_full_figure: True=全图输入, False=裁剪区域 (用于 P4 实验)
    """

    model_name: str = "OpenGVLab/InternVL2_5-8B"
    fuse_layer_indices: list[int] | None = None
    batch_size: int = 8
    max_length: int = 2048
    device: str = "cuda"
    cache_dir: str = "data/processed/features"
    use_full_figure: bool = True
    fail_on_placeholder: bool = False


class FeatureExtractor:
    """VLM 特征提取器。

    封装了模型加载、multi-layer fusion、mean pooling 的完整流程。
    支持批量提取并缓存到磁盘。

    Args:
        config: 提取配置
    """

    def __init__(self, config: ExtractionConfig | None = None) -> None:
        self.config = config or ExtractionConfig()
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self._loaded = False
        self.vision_backend = "unknown"
        self._warned_vision_model = False

    def load_model(self) -> None:
        """延迟加载模型到 GPU。仅在首次调用 extract 时触发。"""
        if self._loaded:
            return

        logger.info(f"Loading {self.config.model_name}...")

        # InternVL2.5-8B 使用 transformers AutoModel
        try:
            from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

            self.model = AutoModel.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                output_hidden_states=True,
            ).to(self.config.device)
            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
            )
            try:
                self.image_processor = AutoImageProcessor.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=True,
                )
            except Exception:
                self.image_processor = None
        except Exception as e:
            logger.warning(f"Failed to load {self.config.model_name}: {e}")
            self.model = None
            self.tokenizer = None
            self.image_processor = None
            self.vision_backend = "image_stats_fallback"
            logger.info("Will use image-stats fallback features.")

        self._loaded = True

        # 自动推断 fuse_layer_indices
        if self.config.fuse_layer_indices is None and self.model is not None:
            n_layers = self._infer_num_hidden_layers()
            self.config.fuse_layer_indices = [
                n_layers // 4,
                n_layers // 2,
                n_layers * 3 // 4,
                n_layers - 1,
            ]
            logger.info(f"Auto fuse layers: {self.config.fuse_layer_indices} (total {n_layers} layers)")

    def _infer_num_hidden_layers(self) -> int:
        """兼容不同模型配置结构，推断 hidden layers 数量。"""
        if self.model is None:
            return 32
        cfg = getattr(self.model, "config", None)
        if cfg is None:
            return 32

        candidates: list[Any] = [cfg]
        for name in ["llm_config", "text_config", "language_config", "vision_config"]:
            sub = getattr(cfg, name, None)
            if sub is not None:
                candidates.append(sub)

        for c in candidates:
            n = getattr(c, "num_hidden_layers", None)
            if isinstance(n, int) and n > 0:
                return n
            n = getattr(c, "n_layer", None)
            if isinstance(n, int) and n > 0:
                return n
        return 32

    def extract_text_features(self, texts: list[str]) -> np.ndarray:
        """批量提取文本特征。

        Args:
            texts: 文本列表

        Returns:
            features: (N, hidden_dim)
        """
        self.load_model()

        if self.model is None:
            if self.config.fail_on_placeholder:
                raise RuntimeError("Text model unavailable and fail_on_placeholder=True")
            return self._text_hash_features(texts)

        all_features = []
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i : i + self.config.batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            ).to(self.config.device)

            with torch.no_grad():
                outputs = self._forward_text(inputs)

            # Multi-layer fusion + mean pooling
            hidden_states = outputs.hidden_states
            fused = self._fuse_and_pool(hidden_states, inputs.get("attention_mask"))
            all_features.append(fused.cpu().numpy())

        return np.concatenate(all_features, axis=0)

    def _forward_text(self, inputs: dict[str, torch.Tensor]) -> Any:
        """优先走全模型前向，失败时回退到 language 模型前向。"""
        assert self.model is not None
        try:
            return self.model(**inputs, output_hidden_states=True)
        except Exception:
            pass

        for name in ["language_model", "llm", "llm_model", "text_model", "model"]:
            sub = getattr(self.model, name, None)
            if sub is None:
                continue
            try:
                return sub(**inputs, output_hidden_states=True)
            except Exception:
                continue

        raise RuntimeError("No usable text forward path for current model")

    def extract_visual_features(self, images: list[str | Image.Image]) -> np.ndarray:
        """批量提取视觉特征。

        Args:
            images: 图像路径或 PIL.Image 列表

        Returns:
            features: (N, hidden_dim)
        """
        self.load_model()

        if self.model is None:
            return self._placeholder_features(len(images))

        loaded_images = []
        for img in images:
            if isinstance(img, str):
                if Path(img).exists():
                    loaded_images.append(Image.open(img).convert("RGB"))
                else:
                    loaded_images.append(Image.new("RGB", (224, 224), color=(128, 128, 128)))
            else:
                loaded_images.append(img)

        all_features = []
        for i in range(0, len(loaded_images), self.config.batch_size):
            batch_images = loaded_images[i : i + self.config.batch_size]
            # InternVL2.5 使用 chat 接口时需要处理图像
            # 这里用更底层的方式：利用 vision encoder 直接提取
            features = self._extract_vision_batch(batch_images)
            all_features.append(features)

        return np.concatenate(all_features, axis=0)

    def _extract_vision_batch(self, images: list[Image.Image]) -> np.ndarray:
        """内部方法：对一批图像提取视觉特征。

        由于 InternVL 的具体接口可能因版本而异，
        这里提供基本框架，实际运行时需根据模型版本调整。
        """
        if self.model is None:
            if self.config.fail_on_placeholder:
                raise RuntimeError("Vision model unavailable and fail_on_placeholder=True")
            self.vision_backend = "image_stats_fallback"
            return self._image_stats_features(images)

        # Try generic image feature APIs first.
        try:
            if self.image_processor is not None:
                proc = self.image_processor(images=images, return_tensors="pt")
                model_dtype = next(self.model.parameters()).dtype if self.model is not None else torch.float32
                proc_cast = {}
                for k, v in proc.items():
                    if not torch.is_tensor(v):
                        continue
                    t = v.to(self.config.device)
                    if t.is_floating_point():
                        t = t.to(model_dtype)
                    proc_cast[k] = t
                proc = proc_cast

                # InternVL family: prefer extract_feature path; vision_model often has incompatible signatures.
                prefer_extract = "internvl" in self.config.model_name.lower()

                if prefer_extract and hasattr(self.model, "extract_feature") and "pixel_values" in proc:
                    try:
                        pv = proc["pixel_values"]
                        with torch.no_grad():
                            try:
                                out = self.model.extract_feature(pv)
                            except Exception:
                                if pv.ndim == 4:
                                    out = self.model.extract_feature(pv.unsqueeze(1))
                                else:
                                    raise
                        self.vision_backend = "model.extract_feature"
                        if isinstance(out, torch.Tensor):
                            if out.ndim == 3:
                                out = out.mean(dim=1)
                            return out.detach().float().cpu().numpy()
                        if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
                            t = out[0]
                            if t.ndim == 3:
                                t = t.mean(dim=1)
                            return t.detach().float().cpu().numpy()
                    except Exception as e:
                        logger.warning(f"extract_feature failed: {e}")

                if hasattr(self.model, "get_image_features"):
                    try:
                        with torch.no_grad():
                            out = self.model.get_image_features(**proc)
                        self.vision_backend = "model.get_image_features"
                        if isinstance(out, torch.Tensor):
                            return out.detach().float().cpu().numpy()
                        if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
                            t = out[0]
                            if t.ndim == 3:
                                t = t.mean(dim=1)
                            return t.detach().float().cpu().numpy()
                    except Exception as e:
                        logger.warning(f"get_image_features failed: {e}")

                if hasattr(self.model, "vision_model") and "pixel_values" in proc:
                    try:
                        with torch.no_grad():
                            out = self.model.vision_model(pixel_values=proc["pixel_values"], output_hidden_states=True)
                        if hasattr(out, "hidden_states") and out.hidden_states is not None:
                            fused = self._fuse_and_pool(out.hidden_states, None)
                            self.vision_backend = "model.vision_model.hidden_states"
                            return fused.detach().float().cpu().numpy()
                    except Exception as e:
                        if not self._warned_vision_model:
                            logger.warning(f"vision_model forward failed: {e}")
                            self._warned_vision_model = True

                if hasattr(self.model, "extract_feature") and "pixel_values" in proc:
                    try:
                        pv = proc["pixel_values"]
                        with torch.no_grad():
                            try:
                                out = self.model.extract_feature(pv)
                            except Exception:
                                # Some InternVL implementations expect (B, N, C, H, W).
                                if pv.ndim == 4:
                                    out = self.model.extract_feature(pv.unsqueeze(1))
                                else:
                                    raise
                        self.vision_backend = "model.extract_feature"
                        if isinstance(out, torch.Tensor):
                            if out.ndim == 3:
                                out = out.mean(dim=1)
                            return out.detach().float().cpu().numpy()
                        if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
                            t = out[0]
                            if t.ndim == 3:
                                t = t.mean(dim=1)
                            return t.detach().float().cpu().numpy()
                    except Exception as e:
                        logger.warning(f"extract_feature failed: {e}")
        except Exception as e:
            logger.warning(f"Vision backend failed ({self.config.model_name}): {e}")

        if self.config.fail_on_placeholder:
            raise RuntimeError("No usable vision backend and fail_on_placeholder=True")

        self.vision_backend = "image_stats_fallback"
        return self._image_stats_features(images)

    def _fuse_and_pool(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Multi-layer fusion + mean pooling。

        Args:
            hidden_states: 各层 hidden states
            attention_mask: attention mask

        Returns:
            pooled: (B, hidden_dim)
        """
        from src.models.feh import FactualEntailmentHead

        fused = FactualEntailmentHead.fuse_layers(hidden_states, self.config.fuse_layer_indices)
        pooled = FactualEntailmentHead.pool_sequence(fused, attention_mask)
        return pooled

    def _placeholder_features(self, n: int, dim: int = 4096) -> np.ndarray:
        """生成占位特征向量，用于离线开发。"""
        rng = np.random.RandomState(42)
        return rng.randn(n, dim).astype(np.float32)

    def _text_hash_features(self, texts: list[str], dim: int = 4096) -> np.ndarray:
        """文本哈希特征：当无法加载文本模型时的可复现兜底。"""
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            h = abs(hash(text))
            rng = np.random.RandomState(h % (2**32 - 1))
            out[i] = rng.randn(dim).astype(np.float32)
        return out

    def _image_stats_features(self, images: list[Image.Image], dim: int = 4096) -> np.ndarray:
        """图像统计特征：基于真实像素，避免随机占位。"""
        feats = []
        side = int(np.sqrt(dim))
        if side * side != dim:
            side = 64
            dim = 4096
        for img in images:
            arr = np.asarray(img.convert("L").resize((side, side)), dtype=np.float32) / 255.0
            vec = arr.reshape(-1)
            feats.append(vec)
        return np.stack(feats).astype(np.float32)

    def extract_and_cache(
        self,
        texts: list[str],
        images: list[str | Image.Image],
        labels: list[int],
        output_dir: str | Path,
        split: str = "train",
        perturbation_ratios: list[float] | None = None,
    ) -> Path:
        """批量提取特征并缓存到磁盘。

        Args:
            texts: 文本列表
            images: 图像路径或 PIL.Image 列表
            labels: 标签列表
            output_dir: 输出目录
            split: 数据集划分 ("train"/"val"/"test")
            perturbation_ratios: 篡改幅度列表

        Returns:
            输出目录路径
        """
        output_path = Path(output_dir) / split
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Extracting text features for {len(texts)} samples...")
        text_feats = self.extract_text_features(texts)

        logger.info(f"Extracting visual features for {len(images)} samples...")
        visual_feats = self.extract_visual_features(images)

        np.save(output_path / "text_features.npy", text_feats)
        np.save(output_path / "visual_features.npy", visual_feats)
        np.save(output_path / "labels.npy", np.array(labels))
        if perturbation_ratios is not None:
            np.save(output_path / "perturbation_ratios.npy", np.array(perturbation_ratios))

        logger.info(f"Features cached to {output_path}")
        return output_path
