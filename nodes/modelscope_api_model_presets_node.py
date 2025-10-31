import json
import os

# 定义支持的魔搭API模型列表
SUPPORTED_MODELS = [
    ["Qwen/Qwen-Image", "Qwen-Image"],
    ["black-forest-labs/FLUX.1-schnell", "FLUX.1-schnell"],
    ["black-forest-labs/FLUX.1-Krea-dev", "FLUX.1-Krea-dev"],
    ["stabilityai/stable-diffusion-xl-base-1.0", "SDXL 1.0"],
    ["segmind/Segmind-Vega", "Segmind-Vega"],
    ["Qwen/Qwen-Image-Edit", "Qwen-Image-Edit"],
    ["stabilityai/stable-diffusion-xl-refiner-1.0", "SDXL Refiner"],
    ["runwayml/stable-diffusion-inpainting", "SD Inpainting"],
]

# 预设配置相关函数
def load_model_config():
    """加载模型配置文件，包含预设和用户自定义的模型"""
    config = {}
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 预设配置文件路径
    preset_config_path = os.path.join(current_dir, "modelscope_api_model_presets.json")
    
    # 用户自定义配置文件路径
    custom_config_path = os.path.join(current_dir, "modelscope_api_model_presets_custom.json")
    
    # 加载预设配置
    preset_config = {}
    if os.path.exists(preset_config_path):
        try:
            with open(preset_config_path, 'r', encoding='utf-8') as f:
                preset_config = json.load(f)
        except Exception as e:
            print(f"警告: 无法加载预设配置文件 {preset_config_path}: {str(e)}")
    
    # 加载用户自定义配置
    custom_config = {}
    if os.path.exists(custom_config_path):
        try:
            with open(custom_config_path, 'r', encoding='utf-8') as f:
                custom_config = json.load(f)
        except Exception as e:
            print(f"警告: 无法加载用户自定义配置文件 {custom_config_path}: {str(e)}")
    
    # 合并配置，用户自定义模型优先
    config["preset_models"] = preset_config.get("models", [])
    config["custom_models"] = custom_config.get("models", [])
    config["models"] = config["preset_models"] + config["custom_models"]
    
    return config

def save_model_config(config: dict) -> bool:
    """保存用户自定义的模型配置到文件"""
    try:
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 用户自定义配置文件路径
        custom_config_path = os.path.join(current_dir, "modelscope_api_model_presets_custom.json")
        
        # 准备要保存的数据
        custom_data = {
            "models": config.get("custom_models", [])
        }
        
        # 保存到用户自定义配置文件
        with open(custom_config_path, 'w', encoding='utf-8') as f:
            json.dump(custom_data, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"保存模型配置文件失败: {str(e)}")
        return False

def save_preset_model_config(config: dict) -> bool:
    """保存预设模型配置到文件"""
    try:
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 预设配置文件路径
        preset_config_path = os.path.join(current_dir, "modelscope_api_model_presets.json")
        
        # 准备要保存的数据
        preset_data = {
            "models": config.get("preset_models", [])
        }
        
        # 保存到预设配置文件
        with open(preset_config_path, 'w', encoding='utf-8') as f:
            json.dump(preset_data, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"保存预设模型配置文件失败: {str(e)}")
        return False

class ModelscopeApiSelector:
    """魔搭API模型选择器节点 - 仅提供模型名称选择功能"""
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        config = load_model_config()
        models = config.get("models", SUPPORTED_MODELS)
        
        # 提取所有模型ID
        model_ids = [model[0] for model in models]
        
        # 返回输入类型配置
        return {
            "required": {
                "model_name": (model_ids, {
                    "default": model_ids[0] if model_ids else "",
                    "label": "模型名称",
                    "description": "选择要使用的魔搭API模型名称，可通过魔搭API列表管理节点添加新模型"
                })
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_name",)
    FUNCTION = "get_model_name"
    CATEGORY = "魔搭api"
    
    def get_model_name(self, model_name):
        """返回选中的魔搭API模型名称"""
        return (model_name,)

class ModelscopeApiManager:
    """魔搭API列表管理节点 - 用于查看和管理可用模型
    
    主要功能：
    1. 列出所有可用的魔搭API模型列表
    2. 添加新的魔搭API模型到预设列表中
    3. 删除已有的魔搭API模型
    
    使用说明：
    - 选择"list"操作可查看当前所有可用模型
    - 选择"add"操作可添加新的模型（需要填写模型ID和显示名称）
    - 选择"delete"操作可删除模型（通过下拉列表选择模型）
    - 所有操作结果会通过status_message输出
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        config = load_model_config()
        models = config.get("models", SUPPORTED_MODELS)
        
        # 提取所有模型ID，用于delete操作的下拉选择
        model_ids = [model[0] for model in models]
        
        return {
            "required": {
                "action": (["list", "add", "delete"], {
                    "default": "list",
                    "label": "操作类型",
                    "description": "选择要执行的操作：list(查看模型列表)、add(添加新模型)或delete(删除模型)"
                })
            },
            "optional": {
                "model_id": ("STRING", {
                    "default": "",
                    "placeholder": "例如: black-forest-labs/new-model",
                    "label": "模型ID",
                    "description": "魔搭平台上的完整模型ID，仅在add操作时需要填写"
                }),
                "model_to_delete": (model_ids, {
                    "default": model_ids[0] if model_ids else "",
                    "label": "要删除的模型",
                    "description": "选择要删除的模型，仅在delete操作时有效"
                })
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status_message",)
    FUNCTION = "manage_models"
    CATEGORY = "魔搭api"
    
    def manage_models(self, action, model_id="", model_to_delete=""):
        config = load_model_config()
        models = config.get("models", SUPPORTED_MODELS)
        preset_models = config.get("preset_models", [])
        custom_models = config.get("custom_models", [])
        
        if action == "list":
            # 列出所有可用模型
            message = "可用魔搭API模型列表:\n"
            message += "==============\n"
            message += "使用说明:\n"
            message += "1. 选择模型ID可直接用于魔搭API模型选择器节点\n"
            message += "2. 通过add操作可添加新的模型\n"
            message += "3. 通过delete操作可删除现有模型（直接下拉选择）\n"
            message += "4. 添加或删除后需重启ComfyUI才能在选择器中看到变化\n"
            message += "5. 注意：模型显示名称功能已移除，直接使用模型ID进行操作\n"
            message += "==============\n"
            message += "模型列表:\n"
            
            for i, model_tuple in enumerate(models):
                try:
                    # 安全地获取模型信息
                    if isinstance(model_tuple, list) and len(model_tuple) >= 1:
                        model_id_val = model_tuple[0]
                        # 标记模型来源
                        source = ""
                        if model_id_val in [p[0] for p in preset_models]:
                            source = " [预设]"
                        elif model_id_val in [c[0] for c in custom_models]:
                            source = " [自定义]"
                        else:
                            source = " [内置]"
                        
                        message += f"{i+1}. {model_id_val} {source}\n"
                    else:
                        # 处理不完整的模型元组
                        message += f"{i+1}. 无效模型配置: {str(model_tuple)}\n"
                except Exception as e:
                    message += f"{i+1}. 处理模型时出错: {str(e)}\n"
            
            return (message,)
        
        elif action == "add":
            # 添加新模型
            if not model_id:
                return ("添加失败: 模型ID不能为空\n提示：请填写模型ID",)
            
            # 检查是否已存在
            try:
                for model_tuple in models:
                    if isinstance(model_tuple, list) and len(model_tuple) > 0:
                        existing_id = model_tuple[0]  # 只获取第一个元素作为模型ID
                        if existing_id == model_id:
                            return ("添加失败: 该模型ID已存在\n提示：请尝试使用不同的模型ID",)
            except Exception as e:
                return (f"检查模型是否存在时出错: {str(e)}",)
            
            # 添加新模型到用户自定义列表（只保存模型ID）
            custom_models.append([model_id])
            config["custom_models"] = custom_models
            success = save_model_config(config)
            
            if success:
                return (f"成功添加魔搭API模型: {model_id}\n提示：请重启ComfyUI以在魔搭API模型选择器中看到新添加的模型",)
            else:
                return ("添加魔搭API模型失败，请检查日志\n可能原因：文件权限问题或磁盘空间不足",)
        
        elif action == "delete":
            # 删除模型（支持删除用户自定义模型和预设模型）
            if not model_to_delete:
                return ("删除失败: 请选择要删除的模型\n提示：请使用下拉菜单选择要删除的模型",)
            
            # 检查要删除的模型是否为用户自定义模型或预设模型
            is_custom_model = False
            is_preset_model = False
            
            # 检查是否为用户自定义模型
            for custom_model in custom_models:
                if isinstance(custom_model, list) and len(custom_model) > 0:
                    if custom_model[0] == model_to_delete:
                        is_custom_model = True
                        break
            
            # 检查是否为预设模型
            for preset_model in preset_models:
                if isinstance(preset_model, list) and len(preset_model) > 0:
                    if preset_model[0] == model_to_delete:
                        is_preset_model = True
                        break
            
            # 如果既不是自定义模型也不是预设模型，则检查是否为内置模型
            is_builtin_model = False
            if not is_custom_model and not is_preset_model:
                for builtin_model in SUPPORTED_MODELS:
                    if isinstance(builtin_model, list) and len(builtin_model) > 0:
                        if builtin_model[0] == model_to_delete:
                            is_builtin_model = True
                            # 将内置模型添加到预设模型中以便删除
                            preset_models.append(builtin_model)
                            config["preset_models"] = preset_models
                            break
            
            # 内置模型、预设模型和自定义模型都支持删除
            if not is_custom_model and not is_preset_model and not is_builtin_model:
                return ("删除失败: 未找到指定的模型\n提示：请选择有效的模型进行删除",)
            
            # 删除用户自定义模型
            if is_custom_model:
                deleted = False
                try:
                    # 创建一个新的列表，排除要删除的模型
                    new_custom_models = []
                    for model_tuple in custom_models:
                        if isinstance(model_tuple, list) and len(model_tuple) > 0:
                            existing_id = model_tuple[0]
                            if existing_id == model_to_delete:
                                deleted = True
                                # 不添加到新列表中，相当于删除
                            else:
                                new_custom_models.append(model_tuple)
                        else:
                            # 保留无效的模型定义（虽然不应该存在）
                            new_custom_models.append(model_tuple)
                    
                    if deleted:
                        # 更新配置并保存
                        config["custom_models"] = new_custom_models
                        success = save_model_config(config)
                        
                        if success:
                            return (f"成功删除用户自定义模型: {model_to_delete}\n提示：请重启ComfyUI以从魔搭API模型选择器中移除此模型",)
                        else:
                            return ("删除用户自定义模型失败，请检查日志\n可能原因：文件权限问题或磁盘空间不足",)
                    else:
                        return ("删除失败: 未找到指定的用户自定义模型\n提示：请确认选择的模型ID是否正确",)
                except Exception as e:
                    return (f"删除用户自定义模型时出错: {str(e)}",)
            
            # 删除预设模型或内置模型（通过从预设配置中排除来实现）
            elif is_preset_model or is_builtin_model:
                try:
                    # 从预设模型列表中排除要删除的模型
                    new_preset_models = []
                    for model_tuple in preset_models:
                        if isinstance(model_tuple, list) and len(model_tuple) > 0:
                            existing_id = model_tuple[0]
                            if existing_id == model_to_delete:
                                # 不添加到新列表中，相当于删除
                                pass
                            else:
                                new_preset_models.append(model_tuple)
                        else:
                            # 保留无效的模型定义
                            new_preset_models.append(model_tuple)
                    
                    # 更新配置并保存
                    config["preset_models"] = new_preset_models
                    
                    # 同时更新预设配置文件
                    success = save_preset_model_config(config)
                    
                    model_type = "内置" if is_builtin_model else "预设"
                    if success:
                        return (f"成功删除{model_type}模型: {model_to_delete}\n提示：请重启ComfyUI以从魔搭API模型选择器中移除此模型",)
                    else:
                        return (f"删除{model_type}模型失败，请检查日志\n可能原因：文件权限问题或磁盘空间不足",)
                except Exception as e:
                    return (f"删除{model_type}模型时出错: {str(e)}",)
        
        return ("未知操作\n提示：请选择有效的操作类型（list、add或delete）",)

# 出节点映射和显示名称映射
NODE_CLASS_MAPPINGS = {
    "ModelscopeApiSelector": ModelscopeApiSelector,
    "ModelscopeApiManager": ModelscopeApiManager
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelscopeApiSelector": "魔搭API-大模型选择器",
    "ModelscopeApiManager": "魔搭API-大模型列表管理"  
}

# 确保模块被正确导入
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS"
]