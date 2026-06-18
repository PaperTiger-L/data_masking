import os
import sys
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch


class TextBlurrer:
    def __init__(
        self,
        warmup: bool = True,
        dbnet_root: Optional[str] = None,
        weights_path: Optional[str] = None,
        input_size: int = 960,
        prob_threshold: float = 0.10,
        box_threshold: float = 0.30,
        max_candidates: int = 800,
        min_size: int = 2,
        dilate_px: int = 16,
        pad_ratio: float = 0.08,
        use_padded_rect: bool = False,
    ):
        """
        基于本地 DBNet 仓库的街牌/文本检测与模糊，流程与 FaceBlurrer 保持一致。

        参数:
          dbnet_root: DBNet 仓库根目录（默认 ../DBNet）
          weights_path: DBNet 权重 .pt（默认使用 DBNet/weights/best.pt）
          input_size: 推理高度，宽按比例缩放并对齐到 32
        """
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        self.device = torch.device("cpu")

        # 解析 DBNet 路径
        if dbnet_root is None:
            here = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            direct_dbnet = os.path.normpath(os.path.join(here, "DBNet"))
            nested_dbnet = os.path.normpath(os.path.join(here, "src", "DBNet"))
            dbnet_root = direct_dbnet if os.path.isdir(direct_dbnet) else nested_dbnet
        self.dbnet_root = dbnet_root
        self.weights_path = (
            weights_path
            if weights_path is not None
            else os.path.join(self.dbnet_root, "weights", "best.pt")
        )
        self.input_size = int(input_size)
        # 降低阈值提升召回；膨胀覆盖更多区域
        self.prob_threshold = float(prob_threshold)
        self.box_threshold = float(box_threshold)
        self.max_candidates = int(max_candidates)
        self.min_size = int(min_size)
        self.dilate_px = int(dilate_px)
        self.pad_ratio = float(pad_ratio)
        self.use_padded_rect = bool(use_padded_rect)

        # 导入 DBNet 模块
        if self.dbnet_root not in sys.path:
            sys.path.append(self.dbnet_root)
        try:
            from nets import nn as dbnn  # type: ignore
            from utils import util as dutil  # type: ignore
        except Exception as e:
            raise ImportError(
                f"无法从 {self.dbnet_root} 导入 DBNet 依赖，请确认仓库存在且可用"
            ) from e
        self.dbnn = dbnn
        self.dutil = dutil

        # 加载模型
        self.model = self._load_model()

        # 预热
        if warmup:
            try:
                dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
                _ = self.detect_text_polygons(dummy)
            except Exception:
                pass

    def _load_model(self):
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"未找到 DBNet 权重: {self.weights_path}，请将权重放置于该路径。"
            )
        # 两种权重形态：训练保存的 {'model': ...} 或 直接模型
        model_obj = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        if isinstance(model_obj, dict) and "model" in model_obj:
            model = model_obj["model"]
        else:
            model = model_obj
        model = model.float().to(self.device)
        model.eval()
        return model

    def _preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        # 与 DBNet demo 对齐：固定高度 input_size，宽按比例并对齐 32
        h, w = image.shape[:2]
        width = int(np.ceil(((w * self.input_size / h) / 32.0)) * 32)
        x = cv2.resize(image, dsize=(width, self.input_size), interpolation=cv2.INTER_LINEAR)
        x = x.astype("float32") / 255.0
        mean = np.array([0.406, 0.456, 0.485], dtype=np.float32).reshape((1, 1, 3))
        std = np.array([0.225, 0.224, 0.229], dtype=np.float32).reshape((1, 1, 3))
        x = (x - mean) / std
        # BGR -> RGB and HWC -> CHW
        x = x[:, :, ::-1].transpose(2, 0, 1)
        x = np.ascontiguousarray(x)
        t = torch.from_numpy(x).unsqueeze(0).to(self.device)
        return t, (h, w)

    def detect_text_polygons(self, image: np.ndarray) -> List[np.ndarray]:
        """返回文本多边形列表，每个为 Nx2 的 numpy 数组。"""
        x, orig_shape = self._preprocess(image)
        with torch.no_grad():
            outputs = self.model(x)
        outputs_cpu = outputs.cpu()
        # 使用 DBNet 内置 mask_to_box，传入更低概率阈值以提升召回
        res = self.dutil.mask_to_box(
            targets={"shape": [orig_shape]}, outputs=outputs_cpu,
            threshold=self.prob_threshold, is_polygon=True
        )
        boxes, scores = res[0][0], res[1][0]
        polygons: List[np.ndarray] = []
        for poly in boxes:
            arr = np.array(poly, dtype=np.int32).reshape((-1, 2))
            polygons.append(arr)
        # 对检测到的区域进行统一膨胀，覆盖边缘与细节
        if self.dilate_px > 0 and polygons:
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            for poly in polygons:
                cv2.fillPoly(mask, [poly.reshape((-1, 1, 2))], 255)
            # 自适应膨胀：叠加固定像素 + 相对尺寸
            short = min(h, w)
            k = max(1, int(self.dilate_px + 0.01 * short))
            if k % 2 == 0:
                k += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            dilated = cv2.dilate(mask, kernel, iterations=1)
            # 从膨胀后的掩码重新提取多边形
            cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            new_polys: List[np.ndarray] = []
            for c in cnts:
                if cv2.contourArea(c) < 20:  # 过滤极小区域
                    continue
                epsilon = 0.002 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, epsilon, True).reshape((-1, 2))
                new_polys.append(approx.astype(np.int32))
            if new_polys:
                polygons = new_polys

        # 可选：将多边形转为带 padding 的矩形（轴对齐）。斜体会放大覆盖，默认关闭。
        if self.use_padded_rect and polygons and self.pad_ratio > 0:
            h, w = image.shape[:2]
            padded_polys: List[np.ndarray] = []
            for poly in polygons:
                x, y, bw, bh = cv2.boundingRect(poly.reshape((-1, 1, 2)))
                pad_x = int(bw * self.pad_ratio)
                pad_y = int(bh * self.pad_ratio)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w, x + bw + pad_x)
                y2 = min(h, y + bh + pad_y)
                rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                padded_polys.append(rect)
            polygons = padded_polys
        return polygons

    def blur_polygons(self, image: np.ndarray, polygons: List[np.ndarray], method: str = "gaussian") -> np.ndarray:
        if not polygons:
            return image
        result = image.copy()
        for poly in polygons:
            # 以多边形外接矩形为 ROI，参照人脸模糊逻辑
            x, y, w, h = cv2.boundingRect(poly.reshape((-1, 1, 2)))
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(image.shape[1], x + w), min(image.shape[0], y + h)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = result[y1:y2, x1:x2]

            # 多边形在 ROI 内的掩码
            shifted = poly - np.array([x1, y1])
            mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
            cv2.fillPoly(mask, [shifted.reshape((-1, 1, 2))], 255)

            if method == "pixelate":
                pixel_size = max(1, min(roi.shape[0] // 10, roi.shape[1] // 10))
                small = cv2.resize(
                    roi,
                    (max(1, roi.shape[1] // pixel_size), max(1, roi.shape[0] // pixel_size)),
                    interpolation=cv2.INTER_NEAREST,
                )
                blurred = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
            elif method == "solid":
                blurred = np.zeros_like(roi)
            else:
                # 高斯模糊（参照 FaceBlurrer，核随 ROI 尺寸）
                ksize = max(3, min(roi.shape[0] // 4, roi.shape[1] // 4) * 2 + 1)
                blurred = cv2.GaussianBlur(roi, (ksize, ksize), 0)

            # 仅覆盖多边形区域
            roi[mask == 255] = blurred[mask == 255]
            result[y1:y2, x1:x2] = roi
        return result

    def process_image(self, input_path: str, output_path: str, method: str = "gaussian", max_size: int = 1000) -> str:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        image = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图像: {input_path}")

        # 可选下采样至较小尺寸以提速（不强制，与 FaceBlurrer 风格一致）
        orig = image.copy()
        oh, ow = image.shape[:2]
        scale = min(max_size / max(oh, ow), 1.0)
        if scale < 1.0:
            image = cv2.resize(image, (int(ow * scale), int(oh * scale)))

        polygons = self.detect_text_polygons(image)
        print(f"检测到 {len(polygons)} 个文本区域")
        if polygons:
            image = self.blur_polygons(image, polygons, method)

        # 还原到原始尺寸
        if scale < 1.0:
            image = cv2.resize(image, (ow, oh))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not cv2.imwrite(output_path, image):
            raise IOError(f"无法写入输出文件: {output_path}")
        print(f"处理完成，结果保存至: {output_path}")
        return output_path
