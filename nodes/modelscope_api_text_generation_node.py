import requests
import json
import os
import time

# 检查openai库是否可用
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# 导入必要的函数
from .modelscope_api_node import load_api_token, save_api_token

# 支持的文本生成模型列表
SUPPORTED_TEXT_GENERATION_MODELS = [
    ("Qwen/Qwen3-VL-235B-A22B-Instruct", "Qwen3-VL 235B A22B Instruct"),
]

class ModelscopeApiTextGenerationNode:
    """魔搭API文本生成节点 - 用于生成文本内容"""
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
                "prompt": ("STRING", {
                    "default": "请生成一段关于人工智能的文本",
                    "label": "提示词",
                    "description": "用于文本生成的提示词",
                    "multiline": True
                }),
                "api_token": ("STRING", {
                    "default": saved_token,
                    "label": "API Token",
                    "description": "modelscope API 令牌",
                    "placeholder": "请输入您的 modelscope API Token",
                    "multiline": False
                }),
                "model_name": ("STRING", {
                    "default": "Qwen/Qwen3-VL-235B-A22B-Instruct",
                    "options": [model[0] for model in SUPPORTED_TEXT_GENERATION_MODELS],
                    "labels": {model[0]: model[1] for model in SUPPORTED_TEXT_GENERATION_MODELS},
                    "label": "模型名称"
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "default": "你是一个有帮助的AI助手",
                    "label": "系统提示词",
                    "description": "系统级别的提示词，用于设定AI的行为",
                    "multiline": True
                }),
                "max_tokens": ("INT", {
                    "default": 1000,
                    "min": 100,
                    "max": 4000,
                    "label": "最大令牌数",
                    "description": "生成文本的最大长度"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "label": "温度系数",
                    "description": "控制生成文本的随机性"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "label": "Top P",
                    "description": "控制生成文本的多样性"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("生成文本",)
    FUNCTION = "generate_text"
    CATEGORY = "modelscope-api"
    
    def generate_text(self, prompt, api_token, model_name, system_prompt="你是一个有帮助的AI助手", max_tokens=1000, temperature=0.7, top_p=0.9):
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
            print(f"📝 开始生成文本...")
            print(f"🔤 提示词: {prompt}")
            print(f"🤖 模型: {model_name}")
            print(f"🔑 使用API Token: {token[:10]}...")
            
            try:
                print(f"🔄 使用API Token进行调用...")
                
                # 初始化OpenAI客户端
                client = OpenAI(
                    base_url='https://api-inference.modelscope.cn/v1',
                    api_key=token
                )
                
                # 构建消息体
                messages = []
                if system_prompt.strip():
                    messages.append({
                        'role': 'system',
                        'content': system_prompt,
                    })
                
                messages.append({
                    'role': 'user',
                    'content': prompt,
                })
                
                # 调用API（使用选中的模型）
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=False
                )
                
                # 成功获取结果
                generated_text = response.choices[0].message.content
                print(f"✅ API调用成功!")
                print(f"📄 结果预览: {generated_text[:100]}...")
                return (generated_text,)
                
            except Exception as e:
                error_msg = f"API调用失败: {str(e)}"
                print(f"❌ {error_msg}")
                return (error_msg,)
            
        except Exception as e:
            error_msg = f"文本生成失败: {str(e)}"
            print(f"❌ {error_msg}")
            return (error_msg,)

# 节点映射和显示名称映射
NODE_CLASS_MAPPINGS = {
    "ModelscopeApiTextGenerationNode": ModelscopeApiTextGenerationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelscopeApiTextGenerationNode": "魔搭API-文本生成",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']