## Shotla BroadCast(修特拉快报)

此项目作为`MoeSpeech`的一个子集项目，为`ACT_TTS`提供一种可用的，本地可部署的`TTS`后端，基于[RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)变声器模型实现。

## 项目结构
- models：用于存储必要的模型，包括`Hubert`模型和`RVC`模型。
- logs：用于存储日志文件。

## 开发环境依赖
# 依赖？我不懂什么是依赖！因为我用的是DirectML！

### 只要我的Windows版本>=1903，且显卡支持DX12，就可以直接开玩儿！

### 安装Python依赖
```shell
pip install onnxruntime-directml
```

### Python环境
经过测试，当前在Python 3.10.12环境下可以正常运行，你可以通过以下方式创建新环境：

- venv:

```shell
python -m venv shotla
source shotla/bin/activate
pip install -r requirements.txt
```

- conda:

```shell
conda create -n shotla python=3.10.12 -y
conda activate shotla
pip install -r requirements.txt
```

## 模型下载
为了规避`modelscope`库产生的其他依赖，现在你需要前往[ModelScope](https://www.modelscope.cn/models/ElinLiu/RVC-Resaech/files)中下载对应**标记了_fp16**的ONNX模型文件，并将其放置在`models`目录下。

## 打包
你可以通过使用项目下的`build.cmd`进行打包，~~但由于打包框架从最开始的`pyinstaller`更换为了`Nuitka`，所以暂时还没来得及验证可用性，实在不行再换回`pyinstaller`~~。

### 调用示例
详见[BenchMark](./BenchMarks.cs)

## 更新说明
~~- 对比之前的Patch，使用`Nuitka`打包框架，打包体积可能会比之前小，运行效率能好一些。~~
- 使用了DirectML作为后端，支持更多的显卡，甚至是Intel 独显GPU（有待测试）。
- 最**核心**的更新，现已支持FP16权重的RVC和Hubert模型，显著降低了显存占用和运行时内存占用！

