# -*- coding: utf-8 -*-
import soundfile as sf
import numpy as np
import onnxruntime as ort
import pyworld
from typing import (
    Optional,
    Dict
)
import os
import socket
import resampy
from loguru import logger
import pathlib

if not pathlib.Path(os.getcwd() + "/logs").exists():
    pathlib.Path(os.getcwd() + "/logs").mkdir(parents=True, exist_ok=True)

if not pathlib.Path(os.getcwd() + "/models").exists():
    pathlib.Path(os.getcwd() + "/models").mkdir(parents=True, exist_ok=True)

if list( pathlib.Path(os.getcwd() + "/models").rglob("*.onnx")) == []:
    logger.error(
        "请将Shotla.onnx和hubert_fp16.onnx模型放入当前目录的models文件夹中"
    )
    os.system("pause")

# 设置日志记录
logger.add(
    os.getcwd() + "/logs/ShotlaBroadCast.log",
    rotation="1 day",
    retention="7 days",
    level="INFO",
)

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
    
    def interpolate_f0(
        self,
        f0: np.ndarray
    ) -> np.ndarray:
        """
        对F0音调进行线性插值
        Args:
            f0 (np.ndarray): F0音调数组
        Returns:
            np.ndarray: 插值后的F0音调数组
        """
        N:int = f0.shape[0]

        # 如果F0全部为0，说明PyWorld无法计算F0
        if np.all(f0 <= 0):
            return np.zeros_like(f0)
        
        # 找出所有非零点的位置
        nz:np.ndarray = np.nonzero(f0 > 0)[0]
        # 如果只有一个非零点，则默认全部使用该值
        if nz.size == 1:
            return np.full_like(
                f0,
                f0[nz[0]]
            )
        
        # 计算每个位置在非零点数组上的索引
        pos:np.ndarray = np.arange(N)
        idx:np.ndarray = np.searchsorted(
            nz,
            pos,
            side="left"
        )

        # 限制idx的范围到合法的索引范围
        # 即：0,nz.size-1
        max_i:int = nz.size - 1
        idx_low = np.clip(idx - 1, 0, max_i)
        idx_high = np.clip(idx, 0, max_i)

        # 左右端点在原数组中的索引
        left_idx:np.ndarray = nz[idx_low]
        right_idx:np.ndarray = nz[idx_high]

        # 两端的值
        left_val:np.ndarray = f0[left_idx]
        right_val:np.ndarray = f0[right_idx]

        # 计算分母避免除以0
        denom:float = (right_idx - left_idx).astype(float)
        denom[denom == 0] = 1.0

        # 插值比例
        t:float = (pos - left_idx).astype(float) / denom

        # 线性插值
        out:np.ndarray = left_val + t * (
            right_val - left_val
        )

        # 原非零位置保留原值
        out[f0 > 0] = f0[f0 > 0]
        return out

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
        return self.interpolate_f0(self.resize_f0(f0, p_len))
    
class ContentVec(object):
    """
    用于执行Hubert Wav2Vec音频特征提取的类
    Attributes:
        model (ort.InferenceSession): ONNX模型会话
    """
    def __init__(self):
        self.sess_options:ort.SessionOptions = ort.SessionOptions()
        self.model:ort.InferenceSession = ort.InferenceSession(
            os.getcwd() + "/models/hubert_fp16.onnx",
            providers = [
                "DmlExecutionProvider",
            ]
        )
    
    def __call__(self,wav: np.ndarray) -> np.ndarray:
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
        feats:np.ndarray = wav.astype(np.float16)
        if feats.ndim == 2:  # 双通道处理
            feats = feats.mean(-1)
        assert feats.ndim == 1, feats.ndim
        feats:np.ndarray = np.expand_dims(feats, 0)
        onnx_input:Dict[str,np.ndarray] = {self.model.get_inputs()[0].name: feats}
        logits:np.ndarray = self.model.run(None, onnx_input)[0]
        return logits.transpose(0, 2, 1) # 转置并转换为float32类型 


class ShotlaBroadCast(object):
    """
    ShotlaBroadCast主类
    用于执行基于RVC模型的音频推理
    """
    def __init__(self):
        self.vec_model:ContentVec = ContentVec()
        self.sess_options:ort.SessionOptions = ort.SessionOptions()
        self.model = ort.InferenceSession(
            os.getcwd() + "/models/Shotla_fp16.onnx",
            providers = [
                "DmlExecutionProvider",
            ]
        )
        self.f0_predictor:DioF0Predictor = DioF0Predictor(
            hop_length = 160,
            sampling_rate = 16000,
            f0_min = 30,
            f0_max = 8000
        )
        self.f0_min:int = 30
        self.f0_max:int = 8000
        self.f0_mel_min:int = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max:int = 1127 * np.log(1 + self.f0_max / 700)
    
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
    
    def load(
        self,
        path: str
    ) -> np.ndarray:
        """
        将音频加载为FP32数组
        Args:
            path (str): 音频文件路径
        Returns:
            np.ndarray: 加载的音频数据
        """
        context:sf.SoundFile = sf.SoundFile(path)
        with context as f:
            audio:np.ndarray = f.read(dtype=np.float32).T
            audio:np.ndarray = resampy.resample(
                audio, 
                f.samplerate, 
                16000
            )
            return audio.astype(np.float32)
    
    def __call__(
        self,
        input_path: str,
        output_path: str,
    ) -> None:
        """
        主调用函数
        Args:
            input_path (str): 输入音频路径
            output_path (str): 输出音频路径
        """
        wav16k:np.ndarray = self.load(input_path)
        logger.info(f"已读取音频文件: {input_path}, 长度: {wav16k.shape[0]}")
        hubert_feature:np.ndarray = self.vec_model(wav16k)
        hubert_feature:np.ndarray = np.repeat(
            hubert_feature, 
            2, 
            axis=2
        ).transpose(0, 2, 1).astype(np.float32)
        hubert_length:int = hubert_feature.shape[1]
        logger.info(f"Hubert特征长度: {hubert_length}, 形状: {hubert_feature.shape}")

        pitchf:np.ndarray = self.f0_predictor.compute_f0(
            wav16k, 
            p_len=hubert_length
        )
        pitchf:np.ndarray = pitchf * 2 ** (0 / 12)
        pitch:np.ndarray = pitchf.copy()
        f0_mel:np.ndarray = 1127 * np.log(1 + pitch / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (
            self.f0_mel_max - self.f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        pitch:np.ndarray = np.rint(f0_mel).astype(np.int64)
        logger.info(f"F0计算完成, F0长度: {len(pitchf)}, F0范围: {pitchf.min()} - {pitchf.max()}")

        pitchf:np.ndarray = pitchf.reshape(1, len(pitchf)).astype(np.float32)
        pitch:np.ndarray = pitch.reshape(1, len(pitch))
        ds:np.ndarray = np.array([0]).astype(np.int64)

        rnd:np.ndarray = np.random.randn(1, 192, hubert_length).astype(np.float32)
        hubert_length:np.ndarray = np.array([hubert_length]).astype(np.int64)

        out_wav:np.ndarray = self.forward(
            hubert_feature.astype(np.float16), 
            hubert_length.astype(np.int64), 
            pitch.astype(np.int64), 
            pitchf.astype(np.float16), 
            ds.astype(np.int64), 
            rnd.astype(np.float16)
        ).squeeze()
        out_wav:np.ndarray = np.pad(out_wav, (0, 2 * 160), "constant")
        logger.info(f"模型推理完成, 输出音频长度: {len(out_wav)}")

        sf.write(
            output_path, 
            out_wav, 
            40000, 
            subtype="PCM_16"
        )
        logger.info(f"输出音频已保存到: {output_path}")


def main():
    # 创建Socket连接
    sock:socket.socket = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )
    sock.bind(('0.0.0.0',8023)) # 小彩蛋，80倒过来是我的生日 : )
    # 初始化ShotlaBroadCast实例
    shotla_broadcast:ShotlaBroadCast = ShotlaBroadCast()
    logger.info(f"ShotlaBroadCast服务器已启动，端口8023...")
    while True:
        data, addr = sock.recvfrom(1024)
        # 规约使用\t来标识输入路径和输出路径
        input_path, output_path = data.decode('utf-8').split('\t')
        logger.info(f"已从{addr}接收到请求: {input_path} -> {output_path}")
        try:
            shotla_broadcast(input_path, output_path)
            logger.info(f"输入{input_path}已处理完毕，并保存到{output_path}")
            sock.sendto("0".encode('utf-8'), addr)  # 发送成功信号
        except Exception as e:
            logger.error(f"处理输入：{input_path}时发生错误: {e}")
            sock.sendto("1".encode('utf-8'), addr)

if __name__ == "__main__":
    main()
