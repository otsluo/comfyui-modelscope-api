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

# 支持的视频反推模型列表
SUPPORTED_VIDEO_CAPTION_MODELS = [
    ("Qwen/Qwen3-VL-235B-A22B-Instruct", "Qwen3-VL 235B A22B Instruct"),
    ("Qwen/Qwen2-VL-72B-Instruct", "Qwen2-VL 72B Instruct"),
    ("Qwen/Qwen2-VL-7B-Instruct", "Qwen2-VL 7B Instruct"),
    ("Qwen/Qwen-VL-Chat", "Qwen-VL Chat"),
]

class ModelscopeApiVideoCaptionNode:
    """魔搭API视频反推节点 - 用于从视频生成描述文本"""
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
                "video_frames": ("IMAGE",),
                "api_token": ("STRING", {
                    "default": saved_token,
                    "label": "API Token",
                    "description": "modelscope API 令牌",
                    "placeholder": "请输入您的 modelscope API Token",
                    "multiline": False
                }),
                "model_name": ("STRING", {
                    "default": "Qwen/Qwen3-VL-235B-A22B-Instruct",
                    "options": [model[0] for model in SUPPORTED_VIDEO_CAPTION_MODELS],
                    "labels": {model[0]: model[1] for model in SUPPORTED_VIDEO_CAPTION_MODELS},
                    "label": "模型名称"
                }),
            },
            "optional": {
                "prompt": ("STRING", {
                    "default": "请详细描述这个视频的内容，包括场景、动作、主体、背景等信息",
                    "label": "提示词",
                    "description": "用于视频描述的提示词",
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
    RETURN_NAMES = ("视频描述",)
    FUNCTION = "generate_caption"
    CATEGORY = "modelscope-api"
    
    def generate_caption(self, video_frames, api_token, model_name, prompt="请详细描述这个视频的内容，包括场景、动作、主体、背景等信息", max_tokens=1000, temperature=0.7):
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
            print(f"🔍 开始生成视频描述...")
            print(f"📝 提示词: {prompt}")
            print(f"🤖 模型: {model_name}")
            print(f"🔑 使用API Token: {token[:10]}...")
            
            # 转换视频帧为base64格式列表
            if isinstance(video_frames, torch.Tensor):
                # 处理视频帧张量 (batch of images)
                frame_count = video_frames.shape[0]
                print(f"🎞️ 视频帧数量: {frame_count}")
                
                # 构建消息体，包含所有视频帧
                content = [{
                    'type': 'text',
                    'text': prompt,
                }]
                
                # 添加所有视频帧
                for i in range(frame_count):
                    frame_tensor = video_frames[i:i+1]  # 取单帧
                    frame_url = tensor_to_base64_url(frame_tensor)
                    content.append({
                        'type': 'image_url',
                        'image_url': {
                            'url': frame_url,
                        },
                    })
                
                messages = [{
                    'role': 'user',
                    'content': content,
                }]
            else:
                # 如果不是张量，尝试直接处理
                messages = [{
                    'role': 'user',
                    'content': [{
                        'type': 'text',
                        'text': prompt,
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
            error_msg = f"视频描述生成失败: {str(e)}"
            print(f"❌ {error_msg}")
            return (error_msg,)

# 节点映射和显示名称映射
NODE_CLASS_MAPPINGS = {
    "ModelscopeApiVideoCaptionNode": ModelscopeApiVideoCaptionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelscopeApiVideoCaptionNode": "魔搭API-视频反推",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']