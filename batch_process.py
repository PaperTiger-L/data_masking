#!/usr/bin/env python3
"""
批量处理脚本
用于命令行批量处理图像文件
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

from loguru import logger

# 添加src到路径
sys.path.append(str(Path(__file__).parent / "src"))

from src.pipeline.faces import FaceBlurrer
from src.pipeline.plates import PlateBlurrer
from src.pipeline.texts import TextBlurrer


def process_directory(
    input_dir: str,
    output_dir: str,
    blur_faces: bool = True,
    blur_plates: bool = True,
    blur_texts: bool = True,
    blur_method: str = "gaussian"
):
    """批量处理目录中的图像文件"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        return False
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # 查找所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f"**/*{ext}"))
        image_files.extend(input_path.glob(f"**/*{ext.upper()}"))
    
    if not image_files:
        logger.warning(f"在 {input_dir} 中未找到图像文件")
        return True
    
    logger.info(f"找到 {len(image_files)} 个图像文件")
    
    # 初始化脱敏器
    blurrers = {}
    if blur_faces:
        try:
            blurrers['face'] = FaceBlurrer(warmup=True)
            logger.info("人脸脱敏器初始化完成")
        except Exception as e:
            logger.error(f"人脸脱敏器初始化失败: {e}")
            return False
    
    if blur_plates:
        try:
            blurrers['plate'] = PlateBlurrer(warmup=True)
            logger.info("车牌脱敏器初始化完成")
        except Exception as e:
            logger.error(f"车牌脱敏器初始化失败: {e}")
            return False
    
    if blur_texts:
        try:
            blurrers['text'] = TextBlurrer(warmup=True)
            logger.info("文本脱敏器初始化完成")
        except Exception as e:
            logger.error(f"文本脱敏器初始化失败: {e}")
            return False
    
    # 处理每个文件
    success_count = 0
    for i, image_file in enumerate(image_files, 1):
        try:
            # 计算相对路径以保持目录结构
            rel_path = image_file.relative_to(input_path)
            output_file = output_path / rel_path
            
            # 创建输出目录
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"[{i}/{len(image_files)}] 处理: {rel_path}")
            
            # 依次应用各种脱敏
            current_input = str(image_file)
            temp_files = []
            
            if blur_faces and 'face' in blurrers:
                temp_output = str(output_path / f"temp_face_{rel_path.name}")
                blurrers['face'].process_image(current_input, temp_output, method=blur_method)
                temp_files.append(temp_output)
                current_input = temp_output
            
            if blur_plates and 'plate' in blurrers:
                temp_output = str(output_path / f"temp_plate_{rel_path.name}")
                blurrers['plate'].process_image(current_input, temp_output, method=blur_method)
                temp_files.append(temp_output)
                current_input = temp_output
            
            if blur_texts and 'text' in blurrers:
                temp_output = str(output_path / f"temp_text_{rel_path.name}")
                blurrers['text'].process_image(current_input, temp_output, method=blur_method)
                temp_files.append(temp_output)
                current_input = temp_output
            
            # 复制最终结果
            if temp_files:
                import shutil
                shutil.copy2(current_input, output_file)
                # 清理临时文件
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            else:
                # 如果没有启用任何脱敏，直接复制
                import shutil
                shutil.copy2(image_file, output_file)
            
            success_count += 1
            logger.success(f"完成: {rel_path}")
            
        except Exception as e:
            logger.error(f"处理 {rel_path} 失败: {e}")
            continue
    
    logger.info(f"批量处理完成: {success_count}/{len(image_files)} 个文件成功处理")
    return success_count == len(image_files)


def main():
    parser = argparse.ArgumentParser(description="数据脱敏批量处理工具")
    parser.add_argument("input_dir", help="输入目录路径")
    parser.add_argument("output_dir", help="输出目录路径")
    parser.add_argument("--no-faces", action="store_true", help="禁用人脸脱敏")
    parser.add_argument("--no-plates", action="store_true", help="禁用车牌脱敏")
    parser.add_argument("--no-texts", action="store_true", help="禁用文本脱敏")
    parser.add_argument("--method", choices=["gaussian", "pixelate"], default="gaussian",
                       help="模糊方法 (默认: gaussian)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="日志级别")
    
    args = parser.parse_args()
    
    # 配置日志
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    
    # 执行批量处理
    success = process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        blur_faces=not args.no_faces,
        blur_plates=not args.no_plates,
        blur_texts=not args.no_texts,
        blur_method=args.method
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
