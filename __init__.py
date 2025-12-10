# ModelScope API节点模块初始化文件

# 导入ModelScope API相关节点
from .nodes.modelscope_api_node import modelscopeLoraTextToImageNode, modelscopeLoraImageEditNode
from .nodes.modelscope_api_model_presets_node import ModelscopeApiSelector, ModelscopeApiManager
from .nodes.modelscope_api_lora_presets_node import ModelscopeApiLoraSelector, ModelscopeApiLoraManager
from .nodes.modelscope_api_text_generation_node import ModelscopeApiTextGenerationNode
from .nodes.modelscope_api_image_caption_node import ModelscopeApiImageCaptionNode
from .nodes.modelscope_api_video_caption_node import ModelscopeApiVideoCaptionNode

# 定义节点映射
NODE_CLASS_MAPPINGS = {
    "modelscopeLoraTextToImageNode": modelscopeLoraTextToImageNode,
    "modelscopeLoraImageEditNode": modelscopeLoraImageEditNode,
    "ModelscopeApiSelector": ModelscopeApiSelector,
    "ModelscopeApiManager": ModelscopeApiManager,
    "ModelscopeApiLoraSelector": ModelscopeApiLoraSelector,
    "ModelscopeApiLoraManager": ModelscopeApiLoraManager,
    "ModelscopeApiTextGenerationNode": ModelscopeApiTextGenerationNode,
    "ModelscopeApiImageCaptionNode": ModelscopeApiImageCaptionNode,
    "ModelscopeApiVideoCaptionNode": ModelscopeApiVideoCaptionNode,
}

# 定义节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "modelscopeLoraTextToImageNode": "魔搭API-文生图",
    "modelscopeLoraImageEditNode": "魔搭API-图像编辑",
    "ModelscopeApiSelector": "魔搭API-大模型选择器",
    "ModelscopeApiManager": "魔搭API-大模型列表管理",
    "ModelscopeApiLoraSelector": "魔搭API-Lora模型选择器",
    "ModelscopeApiLoraManager": "魔搭API-Lora列表管理",
    "ModelscopeApiTextGenerationNode": "魔搭API-文本生成",
    "ModelscopeApiImageCaptionNode": "魔搭API-图片反推",
    "ModelscopeApiVideoCaptionNode": "魔搭API-视频反推",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']