"""
ShotlaBroadCast.py
用于实现FF14角色变声的API

此代码的原始实现来自于
- Retrieval-based-Voice-Conversion-WebUI/infer/lib/infer_pack/onnx_inference.py

F0计算实现来自于
- Retrieval-based-Voice-Conversion-WebUI/infer/lib/infer_pack/modules/F0Predictor/DioF0Predictor.py

Hubert模型到处验证实现来自于

- ElinLiu0/RVC-Researching/blob/master/HubertExporationProving.ipynb
"""


from time import perf_counter
from pathlib import Path
from datetime import datetime
from modelscope.hub.snapshot_download import snapshot_download
from typing import *
import wmi
from fastapi import (
    UploadFile,
    File
)
from fastapi.responses import (
    Response,
    JSONResponse
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi import FastAPI
from io import BytesIO
from scipy.io.wavfile import write
from loguru import logger
# from pyadl import ADLManager


import pynvml
import os
import pyworld
import librosa
import numpy as np
import onnxruntime as ort
import uvicorn

# 初始化日志和模型路径
Path(os.getcwd() + '/logs').mkdir(parents=True, exist_ok=True)
Path(os.getcwd() + '/models').mkdir(parents=True, exist_ok=True)

logger.add(
    os.getcwd() + f'/logs/ShotlaBroadCast_{datetime.now().strftime("%Y-%m-%d")}.log',
    rotation = "10 MB",
    retention = "10 days",
    level = "INFO",
)

# 创建WMI客户端
WmiClinet:Any = wmi.WMI()

# 创建FastAPI实例
app:FastAPI = FastAPI()
# 配置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

# 配置GZip中间件
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 创建F0预测器
class DioF0Predictor(object):
    """
    用于执行DiO算法的F0预测器
    Attributes:
        hop_length (int): 帧移
        f0_min (int): 最小F0
        f0_max (int): 最大F0
        sampling_rate (int): 采样率
    """
    def __init__(
        self,
        hop_length:int = 512, 
        f0_min:int = 50, 
        f0_max:int = 1100, 
        sampling_rate:int = 44100
    ) -> None:
        self.hop_length:int = hop_length
        self.f0_min:int = f0_min
        self.f0_max:int = f0_max
        self.sampling_rate:int = sampling_rate

    def interpolate_f0(self,f0) -> np.ndarray:
        """
        用于执行F0的插值函数
        Args:
            f0 (np.ndarray): 输入F0
        Returns:
            np.ndarray: 插值后的F0
        """
        data = f0.copy()
        vuv_vector = (data > 0).astype(np.float32) # 生成VUV向量，即非0值向量
        
        # 定位所有非零点
        nonzero_indices = np.where(data > 0)[0] # 获取所有非零点的索引
        if len(nonzero_indices) == 0: # 如果全是0，则说明WORLD无法提取F0
            return np.zeros_like(data), vuv_vector
        
        # 构建插值区间映射表（关键修正点）
        prev_valid = np.concatenate([[-1], nonzero_indices])
        next_valid = np.concatenate([nonzero_indices, [len(data)]])
        
        # 生成有效区间对（确保维度一致）
        start_indices = prev_valid[:-1] + 1  # 前导区间起始点
        end_indices = next_valid[1:]         # 后续区间结束点
        
        # 修正索引越界问题（关键修改）
        valid_mask = (prev_valid[:-1] >= 0) & (next_valid[1:] < len(data))  # 严格小干
        
        # 应用有效掩码
        start_indices = start_indices[valid_mask]
        end_indices = end_indices[valid_mask]
        prev_values = data[prev_valid[:-1][valid_mask]]
        next_values = data[next_valid[1:][valid_mask]]  # 此时next_values索引已安全
        
        # 向量化插值计算
        for start, end, prev, nxt in zip(start_indices, end_indices, prev_values, next_values):
            if start < end:
                interval_length = end - start
                data[start:end] = np.linspace(prev, nxt, interval_length, endpoint=False)
        
        # 增强边界处理（应对最后一个非零点后的区间）
        last_nonzero = nonzero_indices[-1]
        if last_nonzero < len(data) - 1:
            # 计算需要填充的长度
            fill_length = len(data) - (last_nonzero + 1)
            if fill_length > 0:
                data[last_nonzero+1:] = data[last_nonzero]
        
        # 处理起始边界（应对第一个非零点前的区间）
        first_nonzero = nonzero_indices[0]
        if first_nonzero > 0:
            data[:first_nonzero] = data[first_nonzero]
        
        return data, vuv_vector

    def resize_f0(
        self, 
        x : np.ndarray,
        target_len : int
    ) -> np.ndarray:
        """
        对F0进行重塑
        Args:
            x (np.ndarray): 输入F0
            target_len (int): 目标长度
        Returns:
            np.ndarray: 重塑后的F0
        """
        source:np.ndarray = np.array(x)
        source[source < 0.001] = np.nan
        target:np.ndarray = np.interp(
            np.arange(0, len(source) * target_len, len(source)) / target_len,
            np.arange(0, len(source)),
            source,
        )
        res:np.ndarray = np.nan_to_num(target)
        return res

    def compute_f0(
        self, 
        wav: np.ndarray,
        p_len:Optional[int] = None
    ) -> np.ndarray:
        """
        用于使用DiO算法计算F0的函数
        Args:
            wav (np.ndarray): 输入音频
            p_len (Optional[int], optional): 预测长度. 默认为Hubert特征的长度
        Returns:
            np.ndarray: F0
        """
        if p_len is None:
            p_len:int = wav.shape[0] // self.hop_length
        f0, t = pyworld.dio(
            wav.astype(np.double),
            fs=self.sampling_rate,
            f0_floor=self.f0_min,
            f0_ceil=self.f0_max,
            frame_period=1000 * self.hop_length / self.sampling_rate,
        )
        f0:np.ndarray = pyworld.stonemask(wav.astype(np.double), f0, t, self.sampling_rate)
        for index, pitch in enumerate(f0):
            f0[index] = round(pitch, 1)
        return self.interpolate_f0(self.resize_f0(f0, p_len))[0]

class ContentVec:
    def __init__(
            self,
            IsDirectML:bool = False,
            AllocatedMemory:int = 0,
        ):
        self.Provider:str = "CUDAExecutionProvider" if not IsDirectML else "DmlExecutionProvider"
        logger.debug(f"使用{self.Provider}进行推理")
        self.model:ort.InferenceSession = ort.InferenceSession(
            os.getcwd() + "/models/hubert.onnx",
            providers = [
                (
                    self.Provider,
                    {
                        "device_id": "0",
                        "arena_extend_strategy": "kSameAsRequested",
                        "gpu_mem_limit": AllocatedMemory * (1024 ** 3),
                        "cudnn_conv_algo_search": "HEURISTIC",
                        "do_copy_in_default_stream": True
                    }
                )
            ]
        )

    def __call__(
        self, 
        wav:np.ndarray
    ) -> np.ndarray:
        return self.forward(wav)

    def forward(
        self, 
        wav: np.ndarray
    ) -> np.ndarray:
        """
        用于计算Hubert音频特征的函数
        Args:
            wav (np.ndarray): 输入音频
        Returns:
            np.ndarray: Hubert特征
        """
        feats:np.ndarray = wav
        if feats.ndim == 2:  # 双通道处理
            feats = feats.mean(-1)
        assert feats.ndim == 1, feats.ndim
        feats:np.ndarray = np.expand_dims(feats, 0)
        onnx_input:Dict[str,np.ndarray] = {self.model.get_inputs()[0].name: feats}
        logits:np.ndarray = self.model.run(None, onnx_input)[0]
        return logits.transpose(0, 2, 1)  

# 创建音频转换器
class OnnxRVC:
    def __init__(
        self,
        model_path: str,
        sr:int = 40000,
        hop_size:int = 512,
    ):
        # 先获取GPU信息和内存分配策略
        self.GPUName, self.HubertAllocatedMemory, self.RVCAllocatedMemory = self.GetGPUInfoAndMemoryAllocation()
        logger.info(f"发现GPU: {self.GPUName}，Hubert模型内存分配策略: {self.HubertAllocatedMemory}GB，RVC模型内存分配策略: {self.RVCAllocatedMemory}GB")

        # 检查模型是否存在
        if not os.path.exists(os.getcwd() + "/models/Shotla.onnx") or not os.path.exists(os.getcwd() + "/models/hubert.onnx"):
            logger.warning("未发现到指定的模型，正在通过ModelScope下载")
            snapshot_download(
                "ElinLiu/RVC-Resaech",
                local_dir = os.getcwd() + "/models",
            )

        self.vec_model:ContentVec = ContentVec(
            AllocatedMemory = self.HubertAllocatedMemory,
            IsDirectML = "AMD" in self.GPUName
        )
        self.model:ort.InferenceSession = ort.InferenceSession(
            model_path,
            providers = [
                (
                    "CUDAExecutionProvider" if "NVIDIA" in self.GPUName else "DmlExecutionProvider",
                    {
                        "device_id": "0",
                        "arena_extend_strategy": "kSameAsRequested",
                        "gpu_mem_limit": self.RVCAllocatedMemory * (1024 ** 3),
                        "cudnn_conv_algo_search": "HEURISTIC",
                        "do_copy_in_default_stream": True
                    }
                )
            ]
        )
        self.sampling_rate:int = sr
        self.hop_size:int = hop_size
        self.f0_predictor:DioF0Predictor = DioF0Predictor(
            hop_length=hop_size, sampling_rate=sr
        )
    
    def GetGPUInfoAndMemoryAllocation(
        self,
    ) -> Tuple[str, int, int]:
        """
        用于获取GPU信息和ONNX模型内存分配策略
        Returns:
            Tuple[str, int, int]: GPU名称，Hubert模型内存分配策略，RVC模型内存分配策略
        """
        for gpu in WmiClinet.Win32_VideoController():
            if "NVIDIA" in gpu.Name or "AMD" in gpu.Name:
                # 获取GPU名称
                GPUName:str = gpu.Name
                GPUMemory:int = 0

                # 获取GPU内存
                if "NVIDIA" in gpu.Name:
                    pynvml.nvmlInit() # 初始化NVIDIA Management Library句柄
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 获取ID为0的GPU句柄
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle) # 获取GPU内存信息
                    GPUMemory:int = info.total // 1024 // 1024 // 1024 # 获取GPU内存大小
                    pynvml.nvmlShutdown()
                elif "AMD" in gpu.Name:
                    # raise NotImplementedError("暂不支持AMD GPU及其信息获取")
                    GPUMemory:int = gpu.AdapterRAM & 0xffffffff
                logger.info(f"发现GPU: {GPUName}，内存: {GPUMemory}GB")
                if GPUMemory < 8:
                    logger.warning("GPU显存小于8GB，这可能会导致游戏卡顿，或RVC推理效率大打折扣！")
                return GPUName, 2, 2 
        # 如果没有找到GPU，则抛出异常
        raise Exception("未找到可用于RVC运行的GPU，其应为NVIDIA或AMD显卡")

    def forward(
        self, 
        hubert: np.ndarray,
        hubert_length : np.ndarray,
        pitch : np.ndarray,
        pitchf : np.ndarray,
        ds : np.ndarray,
        rnd : np.ndarray,
    ) -> np.ndarray:
        """
        用于执行ONNX模型推理的函数
        Args:
            hubert (np.ndarray): Hubert特征
            hubert_length (np.ndarray): Hubert特征长度
            pitch (np.ndarray): F0
            pitchf (np.ndarray): F0
            ds (np.ndarray): 说话人ID
            rnd (np.ndarray): 随机数
        Returns:
            np.ndarray: 输出音频
        """
        onnx_input:Dict[str,np.ndarray] = {
            self.model.get_inputs()[0].name: hubert,
            self.model.get_inputs()[1].name: hubert_length,
            self.model.get_inputs()[2].name: pitch,
            self.model.get_inputs()[3].name: pitchf,
            self.model.get_inputs()[4].name: ds,
            self.model.get_inputs()[5].name: rnd,
        }
        return (self.model.run(None, onnx_input)[0] * 32767).astype(np.int16)

    def inference(
        self,
        raw_path: BytesIO,
        sid: int, # 说话人ID，默认为0
        f0_up_key:int = 0, # 变调的8度数，默认为0
    ) -> bytes:
        """
        用于执行音频变声的函数
        Args:
            raw_path (BytesIO): 输入音频
            sid (int): 说话人ID
            f0_up_key (int, optional): 变调的8度数. 默认为0
        Returns:
            bytes: 输出音频
        """
        f0_min:int = 50
        f0_max:int = 1100
        f0_mel_min:int = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max:int = 1127 * np.log(1 + f0_max / 700)
        start:float = perf_counter()
        raw_path.seek(0) # 重置文件指针
        wav, sr = librosa.load(raw_path, sr=self.sampling_rate)
        org_length:int = len(wav)

        wav16k:np.ndarray = librosa.resample(wav, orig_sr=self.sampling_rate, target_sr=16000)
        wav16k:np.ndarray = wav16k
        logger.info(f"音频读取并重采样完成，大小: {wav.shape}，耗时: {perf_counter() - start:.4f}s")

        start:float = perf_counter()
        hubert:np.ndarray = self.vec_model(wav16k)
        hubert:np.ndarray = np.repeat(hubert, 2, axis=2).transpose(0, 2, 1).astype(np.float32)
        hubert_length:int = hubert.shape[1]
        logger.info(f"音频特征提取完成，大小: {hubert.shape}，耗时: {perf_counter() - start:.4f}s")

        start:float = perf_counter()
        pitchf:np.ndarray = self.f0_predictor.compute_f0(wav, hubert_length)
        pitchf:np.ndarray = pitchf * 2 ** (f0_up_key / 12)
        pitch:np.ndarray = pitchf.copy()
        f0_mel:int = 1127 * np.log(1 + pitch / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        pitch:np.ndarray = np.rint(f0_mel).astype(np.int64)
        logger.info(f"F0计算完成，大小: {pitch.shape}，耗时: {perf_counter() - start:.4f}s")

        pitchf:np.ndarray = pitchf.reshape(1, len(pitchf)).astype(np.float32)
        pitch:np.ndarray = pitch.reshape(1, len(pitch))
        ds:np.ndarray = np.array([sid]).astype(np.int64)

        rnd:np.ndarray = np.random.randn(1, 192, hubert_length).astype(np.float32)
        hubert_length:np.ndarray = np.array([hubert_length]).astype(np.int64)

        start:float = perf_counter()
        out_wav:np.ndarray = self.forward(hubert, hubert_length, pitch, pitchf, ds, rnd).squeeze()
        out_wav:np.ndarray = np.pad(out_wav, (0, 2 * self.hop_size), "constant")
        logger.info(f"音频变声完成，大小: {out_wav.shape}，耗时: {perf_counter() - start:.4f}s")
        return self.wav_to_bytes(self.sampling_rate, out_wav)
    
    def wav_to_bytes(self, sample_rate: int, data: np.ndarray) -> bytes:
        data = data.astype(np.int16).flatten()
        num_samples = data.size
        bytes_per_sample = 2
        subchunk2_size = num_samples * bytes_per_sample
        chunk_size = 36 + subchunk2_size
        
        header = b'RIFF'
        header += chunk_size.to_bytes(4, 'little')
        header += b'WAVEfmt '
        header += (16).to_bytes(4, 'little')
        header += (1).to_bytes(2, 'little') 
        header += (1).to_bytes(2, 'little') 
        header += sample_rate.to_bytes(4, 'little')
        byte_rate = sample_rate * 1 * 2
        header += byte_rate.to_bytes(4, 'little')
        header += (2).to_bytes(2, 'little') 
        header += (16).to_bytes(2, 'little') 
        header += b'data'
        header += subchunk2_size.to_bytes(4, 'little')
        
        return header + data.tobytes()

# 创建变声器
try:
    VC:OnnxRVC = OnnxRVC(
        model_path = os.getcwd() + "/models/Shotla.onnx",
    )
    # 尝试读取warmup_audios文件夹下用于预热的音频
    Path(os.getcwd() + "/warmup_audios").mkdir(parents=True, exist_ok=True)
    # 如果预热目录下没有音频，则调用ModelScope下载
    if list(Path(os.getcwd() + "/warmup_audios").glob("*.wav")) == []:
        logger.warning("未发现预热音频，正在通过ModelScope下载")
        snapshot_download(
            "ElinLiu/RVC-WarmupAudios",
            local_dir = os.getcwd() + "/warmup_audios",
        )
    # 预热模型
    warmup_audios:List[Path] = list(Path(os.getcwd() + "/warmup_audios").glob("*.wav"))
    for i in range(len(warmup_audios)):
        logger.info(f"正在执行第{i + 1}次预热")
        logger.info(f"=" * 10)
        # 读取音频为BytesIO
        buffer:BytesIO = BytesIO()
        buffer.write(warmup_audios[i].read_bytes())
        buffer.seek(0)
        # 执行变声
        VC.inference(
            buffer,
            0
        )
except Exception as e:
    logger.error(f"初始化变声器失败，错误：{e}")
    os.system("pause")
    os._exit(-1)

@app.post("/convert")
async def convert(
    WAVBuffer: UploadFile = File(...),
) -> Response:
    """
    用于执行音频变声的API
    Args:
        WAVBuffer (UploadFile): 输入音频
    Returns:
        Response: 输出音频
    """
    # 执行变声
    try:
        start = perf_counter()
        content = await WAVBuffer.read()
        audio_buffer = BytesIO(content)  # 直接初始化BytesIO
        
        output = VC.inference(audio_buffer, 0)
        
        logger.info(f"E2E耗时: {perf_counter()-start:.4f}s")
        return Response(content=output, media_type="audio/wav")
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        return Response(content=await WAVBuffer.read(), media_type="audio/wav")
    
@app.post("/change_character")
async def change_character(
    charactor: str
) -> JSONResponse:
    """
    用于变更模型说话人的API
    Args:
        charactor (str): 说话人ID
    Returns:
        JSONResponse: 返回结果
    """
    try:
        VC = OnnxRVC(
            model_path = os.getcwd() + f"/models/{charactor}.onnx",
        )
        logger.info(f"变更模型说话人为{charactor}")
        return JSONResponse(
            content = {
                "code": 200,
                "message": "成功"
            }
        )
    except Exception as e:
        logger.error(f"变更模型说话人失败，错误：{e}")
        return JSONResponse(
            content = {
                "code": 500,
                "message": f"错误：{e}"
            }
        )


if __name__ == "__main__":
    try:
        uvicorn.run(
            app = app,
            host = "0.0.0.0",
            port = 8000,
        )
    except Exception as e:
        logger.error(f"启动API失败，错误：{e}")
        os.system("pause")