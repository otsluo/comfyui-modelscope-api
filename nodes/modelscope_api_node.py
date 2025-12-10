import requests
import json
import time
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import os
import base64
import tempfile
import random

# 配置相关函数
def load_config():
    # 使用默认配置
    return {
        "default_lora_model": "qiyuanai/TikTok_Xiaohongshu_career_line_beauty_v1",
        "timeout": 720,
        "image_download_timeout": 30,
        "default_prompt": "Career line,with prominent breasts,A very realistic style,high definition photography style,a young woman,long black hair,holding a badminton shuttlecock,standing,outdoors.",
        "default_edit_prompt": "修改图片中的内容",
        "default_negative_prompt": "",
        "default_width": 512,
        "default_height": 512,
        "default_seed": -1,
        "default_steps": 30,
        "default_guidance": 7.5
    }

def save_config(config: dict) -> bool:
    print("配置保存功能已禁用，不再使用modelscope_api_node.json文件")
    return True

def load_api_token():
    return ""

def save_api_token(token):
    print("Token保存功能已禁用，不再使用.modelscope_api_token文件")
    return True

def tensor_to_base64_url(image_tensor):
    try:
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor.squeeze(0)
        
        if image_tensor.max() <= 1.0:
            image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        else:
            image_np = image_tensor.cpu().numpy().astype(np.uint8)
        
        pil_image = Image.fromarray(image_np)
        
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/jpeg;base64,{img_base64}"
        
    except Exception as e:
        print(f"图像转换失败: {e}")
        raise Exception(f"图像格式转换失败: {str(e)}")

# 支持的基础模型列表
SUPPORTED_TEXT_TO_IMAGE_MODELS = [
    ("Qwen/Qwen-Image", "Qwen-Image"),
    ("black-forest-labs/FLUX.1-schnell", "FLUX.1-schnell"),
    ("stabilityai/stable-diffusion-3-medium-diffusers", "SD3 Medium"),
    ("segmind/Segmind-Vega", "Segmind-Vega"),
    ("stabilityai/stable-diffusion-xl-base-1.0", "SDXL 1.0"),
]

SUPPORTED_IMAGE_EDIT_MODELS = [
    ("Qwen/Qwen-Image-Edit", "Qwen-Image-Edit"),
    ("Qwen/Qwen-Image-Edit-2509", "Qwen-Image-Edit-2509"),
    ("runwayml/stable-diffusion-inpainting", "SD Inpainting"),
]

class modelscopeLoraTextToImageNode:
    """支持多种基础模型的文生图节点，包含LoRA支持和批次生成功能"""
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        saved_token = load_api_token()
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Career line,with prominent breasts,A very realistic style,high definition photography style,a young woman,long black hair,holding a badminton shuttlecock,standing,outdoors.",
                    "label": "提示词",
                    "description": "描述您想要生成的图像内容",
                    "placeholder": "描述您想要生成的图像内容"
                }),
                "api_token": ("STRING", {
                    "default": saved_token,
                    "label": "API Token",
                    "description": "modelscope API 令牌，用于调用服务",
                    "placeholder": "请输入您的 modelscope API Token"
                }),
                "base_model": ("STRING", {
                    "default": SUPPORTED_TEXT_TO_IMAGE_MODELS[0][0],
                    "options": [model[0] for model in SUPPORTED_TEXT_TO_IMAGE_MODELS],
                    "labels": {model[0]: model[1] for model in SUPPORTED_TEXT_TO_IMAGE_MODELS},
                    "label": "基础模型"
                }),
                "lora_model": ("STRING", {
                    "default": "qiyuanai/TikTok_Xiaohongshu_career_line_beauty_v1",
                    "label": "LoRA模型"
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "label": "负面提示词",
                    "placeholder": "描述您不想在图像中出现的内容"
                }),
                "width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 2048,
                    "step": 64,
                    "label": "宽度"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 2048,
                    "step": 64,
                    "label": "高度"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "label": "随机种子"
                }),
                "steps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 100,
                    "label": "采样步数"
                }),
                "guidance": ("FLOAT", {
                    "default": 7.5,
                    "min": 1.5,
                    "max": 20.0,
                    "step": 0.1,
                    "label": "引导系数"
                }),
                "lora_weight": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "label": "LoRA权重"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 8,
                    "label": "批次大小",
                    "description": "一次生成的图片数量"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate_with_lora"
    CATEGORY = "modelscope-api"
    
    def generate_with_lora(self, prompt, api_token, base_model, lora_model, batch_size=1, negative_prompt="", 
                          width=512, height=512, seed=-1, steps=30, guidance=7.5, lora_weight=0.8, generate_control="fixed"):
        
        # 验证API Token
        if not api_token or api_token.strip() == "" or api_token.strip() == "api_token":
            raise Exception("请输入有效的API Token（当前配置无效）")
        
        # 保存API Token（如果有变化）
        saved_token = load_api_token()
        if api_token != saved_token:
            if save_api_token(api_token):
                print("✅ API Token已自动保存")
            else:
                print("⚠️ API Token保存失败，但不影响当前使用")
        
        try:
            # 为每个批次生成使用不同的种子
            base_seed = seed if seed != -1 else random.randint(0, 20251003)
            
            # 存储所有生成的图像
            image_tensors = []
            
            # 为每个批次生成图像
            for i in range(batch_size):
                current_seed = base_seed + i if seed != -1 else random.randint(0, 20251003)
                
                # 准备API请求参数
                url = 'https://api-inference.modelscope.cn/v1/images/generations'
                
                # 基础payload
                payload = {
                    'model': base_model,  # 使用用户选择的基础模型
                    'prompt': prompt,
                    'size': f"{width}x{height}",
                    'steps': steps,
                    'guidance_scale': guidance,
                    'generate_control': generate_control,
                    'seed': current_seed
                }
                
                # 修复LoRA参数格式 - 按照官方文档要求
                if lora_model and lora_model.strip() != "":
                    # 单个LoRA模型格式：{"model_id": weight}
                    payload['loras'] = {lora_model: lora_weight}
                
                # 添加可选参数
                if negative_prompt.strip():
                    payload['negative_prompt'] = negative_prompt
                
                # 准备请求头 - 统一格式
                common_headers = {
                    'Authorization': f'Bearer {api_token}',
                    'Content-Type': 'application/json',
                }
                
                headers = {** common_headers, "X-ModelScope-Async-Mode": "true"}
                
                # 发送请求
                print(f"📤 正在提交第 {i+1}/{batch_size} 个LoRA图像生成任务，种子: {current_seed}")
                submission_response = requests.post(
                    url,
                    data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                    headers=headers,
                    timeout=60
                )
                
                # 处理请求响应
                if submission_response.status_code != 200:
                    error_detail = submission_response.text
                    print(f"❌ API请求失败详情:")
                    print(f"   状态码: {submission_response.status_code}")
                    print(f"   响应内容: {error_detail}")
                    try:
                        error_json = submission_response.json()
                        if "errors" in error_json:
                            error_message = error_json["errors"].get("message", "未知错误")
                            error_code = error_json["errors"].get("code", "未知错误码")
                            raise Exception(f"API请求失败 [{submission_response.status_code}]: {error_code} - {error_message}")
                    except:
                        pass
                    raise Exception(f"API请求失败: {submission_response.status_code}, {error_detail}")
                
                submission_json = submission_response.json()
                
                # 处理异步任务
                image_url = None
                if 'task_id' in submission_json:
                    task_id = submission_json['task_id']
                    print(f"🕒 已提交第 {i+1} 个任务，任务ID: {task_id}，开始轮询...")
                    poll_start = time.time()
                    max_wait_seconds = 720
                    
                    while True:
                        # 查询任务状态 - 修复请求头格式
                        task_resp = requests.get(
                            f"https://api-inference.modelscope.cn/v1/tasks/{task_id}",
                            headers={**common_headers, "X-ModelScope-Task-Type": "image_generation"},
                            timeout=120
                        )
                        
                        if task_resp.status_code != 200:
                            raise Exception(f"任务查询失败: {task_resp.status_code}, {task_resp.text}")
                        
                        data = task_resp.json()
                        task_status = data.get("task_status")
                        
                        if task_status == "SUCCEED":
                            if not data.get("output_images") or len(data["output_images"]) == 0:
                                raise Exception("任务成功但未返回图片URL")
                            
                            image_url = data["output_images"][0]
                            print(f"✅ 第 {i+1} 个任务完成，开始下载图片...")
                            
                            # 下载图片
                            img_response = requests.get(image_url, timeout=30)
                            if img_response.status_code != 200:
                                raise Exception(f"图片下载失败: {img_response.status_code}")
                            
                            # 处理图片
                            pil_image = Image.open(BytesIO(img_response.content))
                            if pil_image.mode != 'RGB':
                                pil_image = pil_image.convert('RGB')
                            
                            # 转换为ComfyUI需要的格式
                            image_np = np.array(pil_image).astype(np.float32) / 255.0
                            image_tensor = torch.from_numpy(image_np)[None,]
                            
                            # 添加到图像列表
                            image_tensors.append(image_tensor)
                            break
                            
                        elif task_status == "FAILED":
                            error_message = data.get("errors", {}).get("message", "未知错误")
                            error_code = data.get("errors", {}).get("code", "未知错误码")
                            raise Exception(f"任务失败: 错误码 {error_code}, 错误信息: {error_message}")
                            
                        # 检查超时
                        if time.time() - poll_start > max_wait_seconds:
                            raise Exception("任务轮询超时，请稍后重试或降低并发")
                            
                        # 未完成，继续轮询
                        time.sleep(5)
                else:
                    raise Exception(f"未识别的API返回格式: {submission_json}")
            
            # 合并所有图像张量
            if len(image_tensors) > 0:
                final_tensor = torch.cat(image_tensors, dim=0)
                print(f"🎉 批次图片生成完成！共生成 {len(image_tensors)} 张图片")
                return (final_tensor,)
            else:
                raise Exception("未生成任何图片")
        
        except Exception as e:
            print(f"魔搭API-LoRA调用失败: {str(e)}")
            # 创建一个红色错误图像作为回退
            error_image = Image.new('RGB', (width, height), color='red')
            error_np = np.array(error_image).astype(np.float32) / 255.0
            error_tensor = torch.from_numpy(error_np)[None,]
            # 如果是批次生成，复制图像以匹配批次大小
            if batch_size > 1:
                error_tensor = error_tensor.repeat(batch_size, 1, 1, 1)
            return (error_tensor,)

class modelscopeLoraImageEditNode:
    """支持多种基础模型的图像编辑节点，包含LoRA支持"""
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        saved_token = load_api_token()
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "修改图片中的内容",
                    "label": "编辑提示词",
                    "description": "描述您想要如何编辑图像",
                    "placeholder": "描述您想要如何编辑图像"
                }),
                "api_token": ("STRING", {
                    "default": saved_token,
                    "placeholder": "请输入您的魔搭API Token",
                    "label": "API令牌"
                }),
                "base_model": ("STRING", {
                    "default": SUPPORTED_IMAGE_EDIT_MODELS[0][0],
                    "options": [model[0] for model in SUPPORTED_IMAGE_EDIT_MODELS],
                    "labels": {model[0]: model[1] for model in SUPPORTED_IMAGE_EDIT_MODELS},
                    "label": "基础模型"
                }),
                "lora_model": ("STRING", {
                    "default": "qiyuanai/TikTok_Xiaohongshu_career_line_beauty_v1",
                    "label": "LoRA模型"
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "label": "负面提示词",
                    "placeholder": "描述您不想在编辑后图像中出现的内容"
                }),
                "use_custom_size": ("BOOLEAN", {
                    "default": False,
                    "label": "使用自定义尺寸",
                    "description": "开启时使用自定义宽度和高度，关闭时自动获取输入图像尺寸"
                }),
                "width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 1664,
                    "step": 8,
                    "label": "宽度"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 1664,
                    "step": 8,
                    "label": "高度"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 20251003,
                    "label": "随机种子"
                }),
                "steps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 100,
                    "label": "采样步数"
                }),
                "guidance": ("FLOAT", {
                    "default": 3.5,
                    "min": 1.5,
                    "max": 20.0,
                    "step": 0.1,
                    "label": "引导系数"
                }),
                "lora_weight": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "label": "LoRA权重"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "edit_with_lora"
    CATEGORY = "modelscope-api"
    
    def edit_with_lora(self, image, prompt, api_token, base_model, lora_model, negative_prompt="", 
                       use_custom_size=False, width=512, height=512, seed=-1, steps=30, guidance=3.5, lora_weight=0.8, generate_control="fixed"):
        
        # 验证API Token
        if not api_token or api_token.strip() == "" or api_token.strip() == "api_token":
            raise Exception("请输入有效的API Token（当前配置无效）")
        
        # 保存API Token（如果有变化）
        saved_token = load_api_token()
        if api_token != saved_token:
            if save_api_token(api_token):
                print("✅ API Token已自动保存")
            else:
                print("⚠️ API Token保存失败，但不影响当前使用")
        
        try:
            # 直接使用base64编码方式
            print("📤 使用base64编码方式上传图像...")
            image_data = tensor_to_base64_url(image)
            
            # 准备通用请求头
            common_headers = {
                'Authorization': f'Bearer {api_token}',
                'Content-Type': 'application/json',
            }
            
            payload = {
                'model': base_model,
                'prompt': prompt,
                'image': image_data,
                'generate_control': generate_control
            }
            
            # 修复LoRA参数格式 - 按照官方文档要求
            if lora_model and lora_model.strip() != "":
                payload['loras'] = {lora_model: lora_weight}
            
            # 添加可选参数
            if negative_prompt.strip():
                payload['negative_prompt'] = negative_prompt
                print(f"🚫 负向提示词: {negative_prompt}")
            
            # 处理图像尺寸
            if use_custom_size:
                if width != 512 or height != 512:
                    size = f"{width}x{height}"
                    payload['size'] = size
                    print(f"� 使用自定义图像尺寸: {size}")
            else:
                if len(image.shape) == 4:
                    img = image[0]
                else:
                    img = image
                
                img_height, img_width = img.shape[:2]
                img_width = (img_width // 8) * 8
                img_height = (img_height // 8) * 8
                
                size = f"{img_width}x{img_height}"
                payload['size'] = size
                print(f"📏 自动获取输入图像尺寸: {size}")
            
            # 添加其他参数
            if steps != 30:
                payload['steps'] = steps
                print(f"🔄 采样步数: {steps}")
            
            if guidance != 3.5:
                payload['guidance_scale'] = guidance
                print(f"🧭 引导系数: {guidance}")
            
            # 处理种子
            if seed != -1:
                payload['seed'] = seed
                print(f"🎲 随机种子: {seed}")
            
            model_display_name = next((model[1] for model in SUPPORTED_IMAGE_EDIT_MODELS if model[0] == base_model), base_model)
            print(f"🔧 使用基础模型: {model_display_name} ({base_model})")
            print(f"🧩 使用LoRA模型: {lora_model}")
            print(f"⚖️ LoRA权重: {lora_weight}")
            
            # 发送请求
            print("📤 正在提交LoRA图像编辑任务...")
            url = 'https://api-inference.modelscope.cn/v1/images/generations'
            headers = {** common_headers, "X-ModelScope-Async-Mode": "true"}
            
            submission_response = requests.post(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                headers=headers,
                timeout=60
            )
            
            # 处理请求响应
            if submission_response.status_code != 200:
                error_detail = submission_response.text
                print(f"❌ API请求失败详情:")
                print(f"   状态码: {submission_response.status_code}")
                print(f"   响应内容: {error_detail}")
                try:
                    error_json = submission_response.json()
                    if "errors" in error_json:
                        error_message = error_json["errors"].get("message", "未知错误")
                        error_code = error_json["errors"].get("code", "未知错误码")
                        raise Exception(f"API请求失败 [{submission_response.status_code}]: {error_code} - {error_message}")
                except:
                    pass
                raise Exception(f"API请求失败: {submission_response.status_code}, {error_detail}")
            
            submission_json = submission_response.json()
            
            # 处理异步任务
            result_image_url = None
            if 'task_id' in submission_json:
                task_id = submission_json['task_id']
                print(f"🕒 已提交任务，任务ID: {task_id}，开始轮询...")
                poll_start = time.time()
                max_wait_seconds = 720
                
                while True:
                    # 查询任务状态 - 修复请求头格式
                    task_resp = requests.get(
                        f"https://api-inference.modelscope.cn/v1/tasks/{task_id}",
                        headers={**common_headers, "X-ModelScope-Task-Type": "image_editing"},
                        timeout=120
                    )
                    
                    if task_resp.status_code != 200:
                        raise Exception(f"任务查询失败: {task_resp.status_code}, {task_resp.text}")
                    
                    data = task_resp.json()
                    task_status = data.get("task_status")
                    
                    if task_status == "SUCCEED":
                        if not data.get("output_images") or len(data["output_images"]) == 0:
                            raise Exception("任务成功但未返回图片URL")
                        
                        result_image_url = data["output_images"][0]
                        print("✅ 任务完成，开始下载编辑后的图片...")
                        break
                        
                    elif task_status == "FAILED":
                        error_message = data.get("errors", {}).get("message", "未知错误")
                        error_code = data.get("errors", {}).get("code", "未知错误码")
                        raise Exception(f"任务失败: 错误码 {error_code}, 错误信息: {error_message}")
                        
                    # 检查超时
                    if time.time() - poll_start > max_wait_seconds:
                        raise Exception("任务轮询超时，请稍后重试或降低并发")
                        
                    # 未完成，继续轮询
                    time.sleep(5)
            else:
                raise Exception(f"未识别的API返回格式: {submission_json}")
            
            # 下载编辑后的图片
            img_response = requests.get(result_image_url, timeout=30)
            if img_response.status_code != 200:
                raise Exception(f"图片下载失败: {img_response.status_code}")
            
            # 处理图片
            pil_image = Image.open(BytesIO(img_response.content))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # 转换为ComfyUI需要的格式
            image_np = np.array(pil_image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            print("🎉 图片编辑完成！")
            return (image_tensor,)
            
        except Exception as e:
            print(f"魔搭API-LoRA图像编辑调用失败: {str(e)}")
            # 返回原图像作为错误回退
            return (image.unsqueeze(0),)

# 节点映射和显示名称映射
NODE_CLASS_MAPPINGS = {
    "modelscopeLoraTextToImageNode": modelscopeLoraTextToImageNode,
    "modelscopeLoraImageEditNode": modelscopeLoraImageEditNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "modelscopeLoraTextToImageNode": "魔搭API-文生图",
    "modelscopeLoraImageEditNode": "魔搭API-图像编辑"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']