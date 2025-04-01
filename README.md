## Shotla BroadCast(修特拉快报)

此项目作为`MoeSpeech`的一个子集项目，为`ACT_TTS`提供一种可用的，本地可部署的`TTS`后端，基于[RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)变声器模型实现。

## 项目结构

- warmup_audios：用于实现ONNX模型预测的音频文件，通常此文件夹若没有文件时，则将会从ModelScope仓库自动下载。
- models：用于存储必要的模型，包括`Hubert`模型和`RVC`模型。
- logs：用于存储日志文件。

## 开发环境依赖

### CUDA开发
考虑到ONNXRuntime对CUDA以及cuDNN的版本有特别的要求，因此当你试图使用CUDA分支进行开发时，您可以使用：
- CUDA 11.6
- cuDNN 8.5.0.96

同时不要忘记安装`zlib`，此库作为cuDNN的依赖库，需要提前安装。可参考[CUDNN安装指南](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-890/install-guide/index.html#install-zlib-windows)进行安装

### AMD开发
如果您正在使用AMD GPU显卡，您可能需要手动安装[pyadl](https://github.com/nicolargo/pyadl)库，此库将作为`pywmi`的后置库，从而读取AMD GPU的显存信息，并进行ONNX显存分配。  
同事考虑到AMD GPU无法在Windows上安装完整的ROCm SDK，而HIP只能提供部分的ROCm功能，因此AMD 后端将默认视为DirectML后端。因此您的ONNXRuntime应安装如下方式安装：

```shell
pip install onnxruntime-directml
```

### Python环境
经过测试，当前在Python 3.10.0环境下可以正常运行，你可以通过以下方式创建新环境：

- venv:

```shell
python -m venv shotla
source shotla/bin/activate
pip install -r requirements.txt
```

- conda:

```shell
conda create -n shotla python=3.10 -y
conda activate shotla
pip install -r requirements.txt
```

## 打包
你可以通过使用项目下的`build.bat`进行打包，需要注意的是：exe打包和发布在Github时应标注其模型执行后端，例如：`Shotla-0.1.0-cuda.exe`或`Shotla-0.1.0-directml.exe`。


## API调用

**/convert**

- Method: POST
- Request (form):
    - Name: WAVBuffer(待变声的音频文件二进制流)
    - Type: file
- Response:
    - WAVBuffer: 已经通过RVC模型转换后的音频文件的二进制流

### 调用示例
```python
import requests
from time import perf_counter

api:str = "http://localhost:8000/convert"

input_text:str = "今天天气真的热"

print(f"Input Length: {len(input_bytes)}")

start = perf_counter()
response:bytes = requests.post(
    api,
    params = {
        "text": input_text
    }
)

print(f"Response Legnth: {len(response.content)},inference cost: {round(perf_counter() - start,2)} seconds.")
```

### 结果
```shell
Input Length: 43920
Response Legnth: 175724,inference cost: 2.14 seconds.
```
