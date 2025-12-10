import requests
import json
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import os
import base64
import time

# 检查openai库是否可用
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# 导入必要的函数
from .modelscope_api_node import load_api_token, save_api_token, tensor_to_base64_url

# 支持的图片反推模型列表
SUPPORTED_CAPTION_MODELS = [
    ("Qwen/Qwen3-VL-235B-A22B-Instruct", "Qwen3-VL 235B A22B Instruct"),
    ("Qwen/Qwen3-VL-8B-Instruct", "Qwen3-VL 8B Instruct"),
    ("Qwen/Qwen3-VL-2B-Instruct", "Qwen3-VL 2B Instruct"),
]

class ModelscopeApiImageCaptionNode:
    """魔搭API图片反推节点 - 用于从图像生成描述文本"""
    def __init__(self):
        pass
    
    def parse_api_token(self, token_input):
        """解析输入的单个API Token"""
        if not token_input or token_input.strip() == "":
            # 尝试加载保存的token
            saved_token = load_api_token()
            if saved_token:
                return saved_token
            return ""
        
        # 返回单个Token
        return token_input.strip()
    
    @classmethod
    def INPUT_TYPES(cls):
        if not OPENAI_AVAILABLE:
            return {
                "required": {
                    "error_message": ("STRING", {
                        "default": "请先安装openai库: pip install openai",
                        "multiline": True
                    }),
                }
            }
        saved_token = load_api_token()
        return {
            "required": {
                "image": ("IMAGE",),
                "api_token": ("STRING", {
                    "default": saved_token,
                    "label": "API Token",
                    "description": "modelscope API 令牌",
                    "placeholder": "请输入您的 modelscope API Token",
                    "multiline": False
                }),
                "model_name": ("STRING", {
                    "default": "Qwen/Qwen3-VL-235B-A22B-Instruct",
                    "options": [model[0] for model in SUPPORTED_CAPTION_MODELS],
                    "labels": {model[0]: model[1] for model in SUPPORTED_CAPTION_MODELS},
                    "label": "模型名称"
                }),
            },
            "optional": {
                "prompt": ("STRING", {
                    "default": "详细描述这张图片的内容，包括主体、背景、颜色、风格等信息",
                    "label": "提示词",
                    "description": "用于图片描述的提示词",
                    "multiline": True
                }),
                "max_tokens": ("INT", {
                    "default": 1000,
                    "min": 100,
                    "max": 4000,
                    "label": "最大令牌数",
                    "description": "生成描述文本的最大长度"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "label": "温度系数",
                    "description": "控制生成文本的随机性"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("图片描述",)
    FUNCTION = "generate_caption"
    CATEGORY = "modelscope-api"
    
    def generate_caption(self, image, api_token, model_name, prompt="详细描述这张图片的内容，包括主体、背景、颜色、风格等信息", max_tokens=1000, temperature=0.7):
        if not OPENAI_AVAILABLE:
            return ("请先安装openai库: pip install openai",)
        
        # 解析单个API Token
        token = self.parse_api_token(api_token)
        if not token:
            raise Exception("请输入有效的API Token")
        
        # 保存新Token（如果有变化）
        saved_token = load_api_token()
        if api_token.strip() != saved_token:
            if save_api_token(token):
                print("✅ API Token已自动保存")
            else:
                print("⚠️ API Token保存失败，但不影响当前使用")
        
        try:
            print(f"🔍 开始生成图像描述...")
            print(f"📝 提示词: {prompt}")
            print(f"🤖 模型: {model_name}")
            print(f"🔑 使用API Token: {token[:10]}...")
            
            # 转换图像为base64格式
            image_url = tensor_to_base64_url(image)
            print(f"🖼️ 图像已转换为base64格式")
            
            # 构建消息体
            messages = [{
                'role': 'user',
                'content': [{
                    'type': 'text',
                    'text': prompt,
                }, {
                    'type': 'image_url',
                    'image_url': {
                        'url': image_url,
                    },
                }],
            }]
            
            try:
                print(f"🔄 使用API Token进行调用...")
                
                # 初始化OpenAI客户端
                client = OpenAI(
                    base_url='https://api-inference.modelscope.cn/v1',
                    api_key=token
                )
                
                # 调用API（使用选中的模型）
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=False
                )
                
                # 成功获取结果
                description = response.choices[0].message.content
                print(f"✅ API调用成功!")
                print(f"📄 结果预览: {description[:100]}...")
                return (description,)
                
            except Exception as e:
                error_msg = f"API调用失败: {str(e)}"
                print(f"❌ {error_msg}")
                return (error_msg,)
            
        except Exception as e:
            error_msg = f"图像描述生成失败: {str(e)}"
            print(f"❌ {error_msg}")
            return (error_msg,)

# 节点映射和显示名称映射
NODE_CLASS_MAPPINGS = {
    "ModelscopeApiImageCaptionNode": ModelscopeApiImageCaptionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelscopeApiImageCaptionNode": "魔搭API-图片反推",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']