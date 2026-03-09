#!/usr/bin/env python3
"""
Midjourney 敏感词处理工具
用于清理和优化提示词，避免触发Midjourney API的敏感词检测
"""

class MidjourneyPromptCleaner:
    """Midjourney提示词清理器"""
    
    # 常见敏感词替换映射
    SENSITIVE_REPLACEMENTS = {
        # 身体相关
        "屁股": "臀部",
        "胸部": "上身",
        "胸": "上身", 
        "奶": "饮品",
        "裸": "不着",
        
        # 暴力相关
        "杀": "击败",
        "死": "倒下",
        "血": "红",
        "暴力": "激烈",
        "战斗": "对决",
        
        # 成人内容
        "色": "色彩",
        "性感": "魅力",
        "撩": "互动",
        "诱惑": "吸引",
        
        # 其他可能敏感词
        "赌": "游戏",
        "毒": "药水",
        "枪": "工具",
        "武器": "装备",
    }
    
    @staticmethod
    def clean_prompt(prompt: str) -> str:
        """清理提示词中的敏感词"""
        cleaned = prompt
        
        # 替换常见敏感词
        for sensitive, replacement in MidjourneyPromptCleaner.SENSITIVE_REPLACEMENTS.items():
            cleaned = cleaned.replace(sensitive, replacement)
        
        return cleaned
    
    @staticmethod
    def get_safe_alternatives(original_prompt: str) -> list[str]:
        """获取安全的提示词变体"""
        base_prompt = MidjourneyPromptCleaner.clean_prompt(original_prompt)
        
        alternatives = []
        
        # 方案1：简化描述
        simple_prompt = base_prompt
        alternatives.append(simple_prompt)
        
        # 方案2：更艺术化的表达
        artistic_prompt = base_prompt.replace("花朵", "花卉").replace("叶子", "叶片")
        alternatives.append(artistic_prompt)
        
        # 方案3：英文混合（有时中文更容易触发）
        english_keywords = {
            "花朵": "flowers",
            "紫色": "purple", 
            "绿色": "green",
            "白色": "white",
            "手绘": "hand drawn",
            "插画": "illustration",
            "清新": "fresh",
            "淡雅": "elegant"
        }
        
        english_prompt = base_prompt
        for chinese, english in english_keywords.items():
            english_prompt = english_prompt.replace(chinese, f"{chinese}({english})")
        
        alternatives.append(english_prompt)
        
        return alternatives

def main():
    print("=== Midjourney 敏感词处理工具 ===")
    print()
    
    # 原始提示词
    original_prompt = "彩风格无缝图案，淡紫色花朵灰绿色叶子，花朵圆形绽放，花瓣边缘模糊柔和，细长叶片穿插，纯白色背景，手绘插画，清新淡雅，花朵间距大，留白多，呼吸感，疏朗布局 --tile --s 100 --sw 30"
    
    print("原始提示词：")
    print(original_prompt)
    print()
    
    # 清理提示词
    cleaner = MidjourneyPromptCleaner()
    cleaned_prompt = cleaner.clean_prompt(original_prompt)
    
    print("清理后提示词：")
    print(cleaned_prompt)
    print()
    
    # 获取安全替代方案
    alternatives = cleaner.get_safe_alternatives(cleaned_prompt)
    
    print("=== 安全替代方案 ===")
    for i, alt in enumerate(alternatives, 1):
        print(f"方案{i}：{alt}")
        print()

if __name__ == "__main__":
    main()