import cv2
import torch
import yaml
from ultralytics import YOLO
from pathlib import Path
from loguru import logger

class UnifiedBlurrer:
    """
    一个统一的模糊处理器，使用单个YOLOv8模型检测并模糊化图像中的人脸和车牌。
    该逻辑改编自 dashcam_anonymizer 项目，并为实时单图像处理进行了优化。
    """
    def __init__(self, config_path='config/unified_blur_config.yaml', warmup=True):
        """
        初始化 UnifiedBlurrer。

        参数:
            config_path (str): YAML 配置文件的路径。
            warmup (bool): 是否在初始化时预热模型。
        """
        # 读取配置文件
        with open(config_path, 'r') as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                logger.error(f"读取配置文件失败: {exc}")
                config = {}

        self.conf_thresh = config.get('detection_conf_thresh', 0.1)
        self.blur_radius = config.get('blur_radius', 11)
        use_gpu = config.get('gpu_avail', True)
        model_path = config.get('model_path', 'models/best.pt')

        self.device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'
        self.model = YOLO(model_path)
        self.model.to(self.device)

        if self.blur_radius % 2 == 0:
            self.blur_radius += 1

        if warmup:
            dummy_image_path = Path('data/carpai.jpeg')
            if dummy_image_path.exists():
                self.model(str(dummy_image_path), verbose=False, device=self.device)

    def _blur_regions(self, image, bboxes):
        """
        对图像中的指定区域进行高斯模糊。
        """
        for box in bboxes:
            x1, y1, x2, y2 = map(int, box)
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            blurred_roi = cv2.GaussianBlur(roi, (self.blur_radius, self.blur_radius), 0)
            image[y1:y2, x1:x2] = blurred_roi
        return image

    def process_image(self, input_path: str, output_path: str, method: str = 'gaussian'):
        """
        处理单张图像：检测对象并应用模糊。
        """
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"无法读取图像: {input_path}")

        results = self.model(image, conf=self.conf_thresh, verbose=False, device=self.device)

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            # 关键修复：直接使用模型返回的绝对像素坐标 (xyxy)
            bboxes = results[0].boxes.xyxy.cpu().numpy()
            
            logger.info(f"在 {Path(input_path).name} 中检测到 {len(bboxes)} 个目标，准备进行模糊处理。")
            
            if method == 'gaussian':
                image = self._blur_regions(image, bboxes)
            else:
                image = self._blur_regions(image, bboxes)

        cv2.imwrite(output_path, image)
