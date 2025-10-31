import json
import os

# 定义支持的魔搭API Lora模型列表
SUPPORTED_LORA_MODELS = [
    ["xingchensong/sd_xl_lora_test", "SDXL测试Lora"],
    ["xingchensong/flux_lora_test", "FLUX测试Lora"],
]

# Lora预设配置相关函数
def load_lora_config():
    """加载Lora配置文件，包含预设和用户自定义的Lora模型"""
    config = {}
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 预设配置文件路径
    preset_config_path = os.path.join(current_dir, "modelscope_api_lora_presets.json")
    
    # 用户自定义配置文件路径
    custom_config_path = os.path.join(current_dir, "modelscope_api_lora_presets_custom.json")
    
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
    config["preset_models"] = preset_config.get("lora_models", [])
    config["custom_models"] = custom_config.get("lora_models", [])
    config["lora_models"] = config["preset_models"] + config["custom_models"]
    
    return config

def save_lora_config(config: dict) -> bool:
    """保存用户自定义的Lora配置到文件"""
    try:
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 用户自定义配置文件路径
        custom_config_path = os.path.join(current_dir, "modelscope_api_lora_presets_custom.json")
        
        # 准备要保存的数据
        custom_data = {
            "lora_models": config.get("custom_models", [])
        }
        
        # 保存到用户自定义配置文件
        with open(custom_config_path, 'w', encoding='utf-8') as f:
            json.dump(custom_data, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"保存Lora配置文件失败: {str(e)}")
        return False

class ModelscopeApiLoraSelector:
    """魔搭API-Lora模型选择器节点 - 仅提供Lora模型名称选择功能"""
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        # 加载配置获取Lora模型列表
        config = load_lora_config()
        lora_models = config.get("lora_models", SUPPORTED_LORA_MODELS)
        
        # 提取模型ID
        model_ids = []
        
        # 安全地提取模型信息
        for model_tuple in lora_models:
            try:
                if isinstance(model_tuple, list) and len(model_tuple) >= 1:
                    model_ids.append(model_tuple[0])
                else:
                    # 处理无效的模型定义
                    print(f"警告: 发现无效的Lora模型定义: {model_tuple}")
            except Exception as e:
                print(f"警告: 处理Lora模型时出错: {str(e)}")
        
        # 如果没有有效的模型，使用内置默认模型
        if not model_ids:
            model_ids = [model[0] for model in SUPPORTED_LORA_MODELS]
            print("警告: 未找到有效的Lora模型配置，使用默认内置模型")
        
        return {
            "required": {
                "lora_model": (model_ids, {
                    "default": model_ids[0] if model_ids else SUPPORTED_LORA_MODELS[0][0]
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_model_name",)
    FUNCTION = "get_lora_model_name"
    CATEGORY = "魔搭api"
    
    def get_lora_model_name(self, lora_model):
        """返回选中的魔搭API Lora模型名称"""
        return (lora_model,)

class ModelscopeApiLoraManager:
    """魔搭API-Lora列表管理节点 - 用于查看和管理可用Lora模型
    
    主要功能：
    1. 列出所有可用的魔搭API Lora模型列表
    2. 添加新的魔搭API Lora模型到预设列表中
    3. 删除已有的魔搭API Lora模型
    
    使用说明：
    - 选择"list"操作可查看当前所有可用Lora模型
    - 选择"add"操作可添加新的Lora模型（需要填写模型ID和显示名称）
    - 选择"delete"操作可删除Lora模型（通过下拉列表选择模型）
    - 所有操作结果会通过status_message输出
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        # 加载配置获取Lora模型列表
        config = load_lora_config()
        lora_models = config.get("lora_models", SUPPORTED_LORA_MODELS)
        preset_models = config.get("preset_models", [])
        custom_models = config.get("custom_models", [])
        
        # 提取模型ID用于删除操作的下拉菜单（所有模型）
        model_ids_for_delete = []
        for model_tuple in lora_models:
            try:
                if isinstance(model_tuple, list) and len(model_tuple) >= 1:
                    model_ids_for_delete.append(model_tuple[0])
            except Exception as e:
                print(f"警告: 处理Lora模型时出错: {str(e)}")
        
        return {
            "required": {
                "action": (["list", "add", "delete"], {
                    "default": "list",
                    "label": "操作类型",
                    "description": "选择要执行的操作类型"
                }),
            },
            "optional": {
                "lora_model_id": ("STRING", {
                    "default": "",
                    "label": "Lora模型ID",
                    "description": "要添加的Lora模型ID（仅在add操作时使用）"
                }),
                "lora_model_to_delete": (model_ids_for_delete, {
                    "default": model_ids_for_delete[0] if model_ids_for_delete else "",
                    "label": "要删除的Lora模型",
                    "description": "选择要删除的Lora模型（仅在delete操作时使用）"
                })
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status_message",)
    FUNCTION = "manage_lora_models"
    CATEGORY = "魔搭api"
    
    def manage_lora_models(self, action, lora_model_id="", lora_model_display_name="", lora_model_to_delete=""):
        config = load_lora_config()
        lora_models = config.get("lora_models", SUPPORTED_LORA_MODELS)
        preset_models = config.get("preset_models", [])
        custom_models = config.get("custom_models", [])
        
        if action == "list":
            # 列出所有可用Lora模型
            message = "可用魔搭API Lora模型列表:\n"
            message += "==============\n"
            message += "使用说明:\n"
            message += "1. 选择模型ID可直接用于魔搭API-Lora模型选择器节点\n"
            message += "2. 通过add操作可添加新的Lora模型\n"
            message += "3. 通过delete操作可删除现有Lora模型（直接下拉选择）\n"
            message += "4. 添加或删除后需重启ComfyUI才能在选择器中看到变化\n"
            message += "5. name：SDXL测试Lora(可随意) 、ID: xingchensong/sd_xl_lora_test\n"
            message += "==============\n"
            message += "Lora模型列表:\n"
            
            for i, model_tuple in enumerate(lora_models):
                try:
                    # 安全地获取模型信息
                    if isinstance(model_tuple, list) and len(model_tuple) >= 1:
                        model_id_val = model_tuple[0]
                        # 标记模型来源
                        source = ""
                        for preset_model in preset_models:
                            if preset_model[0] == model_id_val:
                                source = " [预设]"
                                break
                        if not source:  # 如果不是预设模型，则检查是否为用户自定义模型
                            for custom_model in custom_models:
                                if custom_model[0] == model_id_val:
                                    source = " [自定义]"
                                    break
                        if not source:  # 如果既不是预设也不是自定义，则为内置模型
                            for builtin_model in SUPPORTED_LORA_MODELS:
                                if builtin_model[0] == model_id_val:
                                    source = " [内置]"
                                    break
                        message += f"{i+1}. {model_id_val}{source}\n"
                    else:
                        message += f"{i+1}. 无效的模型定义: {model_tuple}\n"
                except Exception as e:
                    message += f"{i+1}. 处理模型时出错: {str(e)}\n"
            
            return (message,)
        
        elif action == "add":
            # 添加新Lora模型
            if not lora_model_id:
                return ("添加失败: Lora模型ID不能为空\n提示：请填写模型ID",)
            
            # 检查是否已存在
            try:
                for model_tuple in lora_models:
                    if isinstance(model_tuple, list) and len(model_tuple) > 0:
                        existing_id = model_tuple[0]  # 只获取第一个元素作为模型ID
                        if existing_id == lora_model_id:
                            return ("添加失败: 该Lora模型ID已存在\n提示：请尝试使用不同的模型ID",)
            except Exception as e:
                return (f"检查Lora模型是否存在时出错: {str(e)}",)
            
            # 添加新Lora模型到用户自定义列表
            custom_models.append([lora_model_id])
            config["custom_models"] = custom_models
            success = save_lora_config(config)
            
            if success:
                return (f"成功添加魔搭API Lora模型: {lora_model_id}\n提示：请重启ComfyUI以在魔搭API-Lora模型选择器中看到新添加的模型",)
            else:
                return ("添加魔搭API Lora模型失败，请检查日志\n可能原因：文件权限问题或磁盘空间不足",)
        
        elif action == "delete":
            # 删除Lora模型（支持删除用户自定义模型和预设模型）
            if not lora_model_to_delete:
                return ("删除失败: 请选择要删除的Lora模型\n提示：请使用下拉菜单选择要删除的模型",)
            
            # 检查要删除的模型是否为用户自定义模型或预设模型
            is_custom_model = False
            is_preset_model = False
            model_display_name = lora_model_to_delete  # 默认使用模型ID作为显示名称
            
            # 查找模型显示名称和类型
            for model_tuple in lora_models:
                if isinstance(model_tuple, list) and len(model_tuple) >= 2:
                    if model_tuple[0] == lora_model_to_delete:
                        model_display_name = model_tuple[1]
                        break
            
            # 检查是否为用户自定义模型
            for custom_model in custom_models:
                if isinstance(custom_model, list) and len(custom_model) > 0:
                    if custom_model[0] == lora_model_to_delete:
                        is_custom_model = True
                        break
            
            # 检查是否为预设模型
            for preset_model in preset_models:
                if isinstance(preset_model, list) and len(preset_model) > 0:
                    if preset_model[0] == lora_model_to_delete:
                        is_preset_model = True
                        break
            
            # 内置模型无法删除
            if not is_custom_model and not is_preset_model:
                return ("删除失败: 无法删除内置模型\n提示：只能删除用户自定义模型或预设模型",)
            
            # 删除用户自定义模型
            if is_custom_model:
                deleted = False
                try:
                    # 创建一个新的列表，排除要删除的模型
                    new_custom_models = []
                    for model_tuple in custom_models:
                        if isinstance(model_tuple, list) and len(model_tuple) > 0:
                            existing_id = model_tuple[0]
                            if existing_id == lora_model_to_delete:
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
                        success = save_lora_config(config)
                        
                        if success:
                            return (f"成功删除用户自定义Lora模型: {model_display_name} (ID: {lora_model_to_delete})\n提示：请重启ComfyUI以从魔搭API-Lora模型选择器中移除此模型",)
                        else:
                            return ("删除用户自定义Lora模型失败，请检查日志\n可能原因：文件权限问题或磁盘空间不足",)
                    else:
                        return ("删除失败: 未找到指定的用户自定义Lora模型\n提示：请确认选择的模型ID是否正确",)
                except Exception as e:
                    return (f"删除用户自定义Lora模型时出错: {str(e)}",)
            
            # 删除预设模型（通过从预设配置中排除来实现）
            elif is_preset_model:
                try:
                    # 从预设模型列表中排除要删除的模型
                    new_preset_models = []
                    for model_tuple in preset_models:
                        if isinstance(model_tuple, list) and len(model_tuple) > 0:
                            existing_id = model_tuple[0]
                            if existing_id == lora_model_to_delete:
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
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    preset_config_path = os.path.join(current_dir, "modelscope_api_lora_presets.json")
                    
                    # 准备要保存的数据
                    preset_data = {
                        "lora_models": new_preset_models
                    }
                    
                    # 保存到预设配置文件
                    with open(preset_config_path, 'w', encoding='utf-8') as f:
                        json.dump(preset_data, f, ensure_ascii=False, indent=2)
                    
                    return (f"成功删除预设Lora模型: {model_display_name} (ID: {lora_model_to_delete})\n提示：请重启ComfyUI以从魔搭API-Lora模型选择器中移除此模型",)
                except Exception as e:
                    return (f"删除预设Lora模型时出错: {str(e)}",)
        
        return ("未知操作\n提示：请选择有效的操作类型（list、add或delete）",)

# 节点映射和显示名称映射
NODE_CLASS_MAPPINGS = {
    "ModelscopeApiLoraSelector": ModelscopeApiLoraSelector,
    "ModelscopeApiLoraManager": ModelscopeApiLoraManager
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelscopeApiLoraSelector": "魔搭API-Lora模型选择器",
    "ModelscopeApiLoraManager": "魔搭API-Lora列表管理"
}

# 确保模块被正确导入
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS"
]