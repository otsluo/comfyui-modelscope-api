# ComfyUI ModelScope API 插件

<p align="center">
  <img src="https://img.shields.io/badge/ComfyUI-Plugin-blueviolet" alt="ComfyUI Plugin">
  <img src="https://img.shields.io/badge/ModelScope-API-orange" alt="ModelScope API">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

这是一个为 ComfyUI 设计的插件，允许用户通过 ModelScope API 进行图像生成和编辑操作。该插件支持多种基础模型和 LoRA 模型，提供了丰富的参数配置选项。

## 📝 作者留言

> 思考了许久，终究还是决定拆分出来，毕竟现在还是作者还只是一个不出名的小小虾米而已，拆分出来之后，才有机会被更多的用户发现并使用。
>
> 此项目主要是魔搭API的部分，完整的功能是在【https://github.com/otsluo/comfyui-xnantool.git】 ，功能会优先在完整项目更新。

## 🌟 功能特性

### ⚠️ 注意事项
- 与 `hujuying/ComfyUI-ModelScope-API` 插件冲突

### 🖼️ 文生图节点 (魔搭API-文生图节点)

支持多种基础模型：
- Qwen/Qwen-Image
- black-forest-labs/FLUX.1-schnell
- black-forest-labs/FLUX.1-Krea-dev
- black-forest-labs/FLUX.1-Kontext
- stabilityai/stable-diffusion-xl-base-1.0
- stabilityai/stable-diffusion-xl-refiner-1.0
- segmind/Segmind-Vega
- Tongyi-MAI/Z-Image-Turbo

主要功能：
- ✅ 支持 LoRA 模型微调
- ✅ 批次生成功能（一次生成多张图片）
- ✅ 丰富的参数配置：
  - 提示词和负面提示词
  - 图像尺寸（宽度和高度）
  - 随机种子
  - 采样步数
  - 引导系数
  - LoRA 权重
- ✅ 自动保存 API Token
- ✅ 异步任务处理机制

### 🎨 图像编辑节点 (魔搭API-图像编辑节点)

支持多种图像编辑模型：
- Qwen/Qwen-Image-Edit
- Qwen/Qwen-Image-Edit-2509
- stabilityai/stable-diffusion-xl-refiner-1.0
- runwayml/stable-diffusion-inpainting


主要功能：
- ✅ 支持 LoRA 模型微调
- ✅ 智能图像尺寸处理（自动获取输入图像尺寸或自定义尺寸）
- ✅ 丰富的参数配置：
  - 编辑提示词和负面提示词
  - 图像尺寸
  - 随机种子
  - 采样步数
  - 引导系数
  - LoRA 权重
- ✅ Base64图像编码上传

## 🚀 安装方法

1. 进入 ComfyUI 的 `custom_nodes` 目录
2. 克隆此仓库：
   ```bash
   git clone https://github.com/otsluo/comfyui-modelscope-api.git
   ```
3. 安装依赖：
   ```bash
   pip install -r comfyui-modelscope-api/requirements.txt
   ```
4. 重启 ComfyUI

## 📖 使用方法

### 🔑 获取 API Token

1. 访问 [ModelScope 官网](https://modelscope.cn/)
2. 注册账号并登录
3. 在个人中心或 API 设置页面获取 API Token

### 🖼️ 文生图节点使用

1. 在 ComfyUI 中找到 "魔搭api" 分类下的 "魔搭API-文生图节点"
2. 输入提示词和 API Token
3. 选择基础模型和 LoRA 模型
4. 调整其他参数（可选）
5. 运行工作流生成图像

### 🎨 图像编辑节点使用

1. 在 ComfyUI 中找到 "魔搭api" 分类下的 "魔搭API-图像编辑节点"
2. 连接输入图像
3. 输入编辑提示词和 API Token
4. 选择基础模型和 LoRA 模型
5. 调整其他参数（可选）
6. 运行工作流编辑图像

## ⚙️ 配置文件

插件会自动生成以下配置文件：

- `config.json`: 主要配置文件，包含默认参数设置
- `.modelscope_api_token`: 保存 API Token

## 📋 支持的模型

### 🖼️ 文生图模型
- Qwen/Qwen-Image
- black-forest-labs/FLUX.1-schnell
- black-forest-labs/FLUX.1-Krea-dev
- black-forest-labs/FLUX.1-Kontext
- stabilityai/stable-diffusion-xl-base-1.0
- stabilityai/stable-diffusion-xl-refiner-1.0
- segmind/Segmind-Vega

### 🎨 图像编辑模型
- Qwen/Qwen-Image-Edit
- Qwen/Qwen-Image-Edit-2509
- stabilityai/stable-diffusion-xl-refiner-1.0
- runwayml/stable-diffusion-inpainting

## 🎛️ 预设配置

插件提供了预设配置文件，方便快速选择常用的模型：

- `modelscope_api_model_presets.json`: 基础模型预设
- `modelscope_api_lora_presets.json`: LoRA模型预设

## 📝 注意事项

1. API Token 是必需的，请确保输入有效的 Token
2. 生成图像需要网络连接
3. 首次使用时可能需要较长时间等待模型加载
4. 批次生成时请注意 API 的使用限制
5. 插件会自动保存您输入的 API Token，下次使用时无需重新输入
6. 当安装了完整项目，那当前的节点将自动归类到XnanTool文件夹下。

## 🛠️ 故障排除

如果遇到问题，请尝试以下解决方法：

1. 检查 API Token 是否正确
2. 确认网络连接正常
3. 查看 ComfyUI 控制台输出的错误信息
4. 确认所选模型是否支持当前功能
5. 检查图像尺寸是否符合模型要求

## 📈 下载统计

<p align="center">
  <img src="https://img.shields.io/github/downloads/otsluo/comfyui-modelscope-api/total?color=blue&label=Total%20Downloads" alt="Total Downloads">
  <img src="https://img.shields.io/github/downloads/otsluo/comfyui-modelscope-api/latest/total?color=green&label=Latest%20Release" alt="Latest Release Downloads">
  <img src="https://img.shields.io/github/stars/otsluo/comfyui-modelscope-api?color=orange&label=Stars" alt="GitHub Stars">
</p>

> ⚠️ 注意：由于 GitHub API 限制，下载统计可能不是实时的。数据更新频率为每日一次。

## 📄 许可证

本项目采用 MIT 许可证，详情请见 [LICENSE](LICENSE) 文件。