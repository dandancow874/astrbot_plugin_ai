import base64
import mimetypes
from datetime import datetime
from pathlib import Path

from astrbot.api import logger


def get_key_index(current_index: int, item_len: int) -> int:
    """获取key索引"""
    return (current_index + 1) % item_len


def save_images(image_result, path_dir: Path, save_json: bool = False, prompt: str = "", model_name: str = "") -> list[tuple[str, Path]]:
    """保存图片到本地文件系统，返回 元组(文件名, 文件路径) 列表
    
    参数:
        image_result: 图片列表 [(mime, b64), ...]
        path_dir: 保存目录
        save_json: 是否同时保存JSON元数据
        prompt: 提示词 (用于JSON元数据)
        model_name: 模型名称 (用于JSON元数据)
    """
    import json
    
    saved_paths: list[tuple[str, Path]] = []
    for mime, b64 in image_result:
        if not b64:
            continue
        # 构建文件名
        now = datetime.now()
        current_time_str = (
            now.strftime("%Y%m%d%H%M%S") + f"{int(now.microsecond / 1000):03d}"
        )
        ext = mimetypes.guess_extension(mime) or ".jpg"
        file_name = f"banana_{current_time_str}{ext}"
        # 构建文件保存路径
        save_path = path_dir / file_name
        # 转换成bytes
        image_bytes = base64.b64decode(b64)
        # 保存到文件系统
        with open(save_path, "wb") as f:
            f.write(image_bytes)
        saved_paths.append((file_name, save_path))
        logger.info(f"[BIG BANANA] 图片已保存到 {save_path}")
        
        # 保存JSON元数据
        if save_json:
            json_data = {
                "tags": [model_name] if model_name else [],
                "annotation": prompt or ""
            }
            json_path = path_dir / f"banana_{current_time_str}.json"
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                logger.info(f"[BIG BANANA] JSON已保存到 {json_path}")
            except Exception as e:
                logger.error(f"[BIG BANANA] 保存JSON失败: {e}")
    
    return saved_paths


def read_file(path) -> tuple[str | None, str | None]:
    try:
        with open(path, "rb") as f:
            file_data = f.read()
            mime_type, _ = mimetypes.guess_type(path)
            b64_data = base64.b64encode(file_data).decode("utf-8")
            return mime_type, b64_data
    except Exception as e:
        logger.error(f"[BIG BANANA] 读取参考图片 {path} 失败: {e}")
        return None, None


def clear_cache(temp_dir: Path):
    """清理缓存文件，应当在图片发送完成后调用"""
    if not temp_dir.exists():
        logger.warning(f"[BIG BANANA] 缓存目录 {temp_dir} 不存在")
        return
    for file in temp_dir.iterdir():
        try:
            if file.is_file():
                file.unlink()
                logger.debug(f"[BIG BANANA] 已删除缓存文件: {file}")
        except Exception as e:
            logger.error(f"[BIG BANANA] 删除缓存文件 {file} 失败: {e}")


def random_string(length: int) -> str:
    import random
    import string

    letters = string.ascii_letters + string.digits
    return "".join(random.choice(letters) for _ in range(length))
