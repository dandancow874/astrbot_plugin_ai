
from .openai_chat import OpenAIImagesProvider

class ZImageProvider(OpenAIImagesProvider):
    api_type: str = "ZImage_Provider"

    @staticmethod
    def _map_image_size(image_size: object, aspect_ratio: str | None = None) -> str | None:
        """
        Gitee AI / Z-Image-Turbo 专用分辨率映射
        """
        if not isinstance(image_size, str):
            return None
        size = image_size.strip().upper()
        
        # 基础分辨率映射 (1:1)
        base_w, base_h = 1024, 1024
        if size == "1K":
            base_w, base_h = 1024, 1024
        elif size == "2K":
            base_w, base_h = 2048, 2048
        elif size == "4K":
            base_w, base_h = 4096, 4096
        elif "X" in size and all(p.isdigit() for p in size.split("X", 1)):
            parts = size.split("X", 1)
            base_w, base_h = int(parts[0]), int(parts[1])
        else:
            base_w, base_h = 1024, 1024
            
        if not aspect_ratio or aspect_ratio.lower() == "default":
            return f"{base_w}x{base_h}"

        # 解析宽高比
        try:
            ar_w, ar_h = map(float, aspect_ratio.replace(":", " ").split())
            target_ratio = ar_w / ar_h
        except Exception:
            return f"{base_w}x{base_h}"

        # Gitee AI / Z-Image-Turbo 专用常用分辨率
        STANDARD_SIZES = [
            # 1:1
            (1024, 1024), (512, 512), (2048, 2048),
            # 16:9 / 9:16
            (1280, 720), (720, 1280),   # 720P (通用)
            (1920, 1080), (1080, 1920), # FHD (部分模型支持)
            # 4:3 / 3:4
            (1024, 768), (768, 1024),   # 通用 4:3
            (1152, 896), (896, 1152),   # SDXL 4:3
            # 3:2 / 2:3
            (1216, 832), (832, 1216),   # SDXL 3:2
        ]

        target_area = base_w * base_h
        best_size = None
        min_score = float("inf")

        for w, h in STANDARD_SIZES:
            ratio = w / h
            # 允许 5% 的宽高比误差
            if abs(ratio - target_ratio) / target_ratio > 0.05:
                continue
            
            # 评分标准：面积差异 + 宽高比差异 (加权)
            area_diff = abs((w * h) - target_area) / target_area
            ratio_diff = abs(ratio - target_ratio)
            
            score = area_diff + ratio_diff * 2
            if score < min_score:
                min_score = score
                best_size = (w, h)

        if best_size:
            return f"{best_size[0]}x{best_size[1]}"

        # 兜底计算
        import math
        new_h = math.sqrt(target_area / target_ratio)
        new_w = new_h * target_ratio
        
        final_w = int(round(new_w / 64) * 64)
        final_h = int(round(new_h / 64) * 64)
        
        return f"{final_w}x{final_h}"
