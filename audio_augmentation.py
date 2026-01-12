"""
音频扩展脚本 - 将短命令音频扩展到指定长度
支持多种扩展方法：时间拉伸、添加噪声、混响、重复等
"""

import numpy as np
from scipy.io import wavfile
import librosa
import argparse


class AudioAugmentor:
    def __init__(self, target_duration=5.0, sample_rate=22050):
        """
        初始化音频扩展器
        
        Args:
            target_duration: 目标音频长度（秒）
            sample_rate: 采样率
        """
        self.target_duration = target_duration
        self.sample_rate = sample_rate
        self.target_length = int(target_duration * sample_rate)
    
    def load_audio(self, audio_path):
        """加载音频文件"""
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        return audio, sr
    
    def add_white_noise(self, audio, noise_level=0.005):
        """添加白噪声"""
        noise = np.random.normal(0, noise_level, len(audio))
        return audio + noise
    
    def add_background_noise(self, audio, noise_audio, noise_level=0.1):
        """
        添加背景噪声
        
        Args:
            audio: 原始音频
            noise_audio: 背景噪声音频
            noise_level: 噪声强度
        """
        if len(noise_audio) < len(audio):
            # 如果噪声太短，重复它
            repeats = int(np.ceil(len(audio) / len(noise_audio)))
            noise_audio = np.tile(noise_audio, repeats)
        
        noise_audio = noise_audio[:len(audio)]
        # 调整噪声强度
        noise_audio = noise_audio * noise_level
        return audio + noise_audio
    
    def time_stretch(self, audio, rate=1.5):
        """
        时间拉伸（不改变音高）
        
        Args:
            audio: 输入音频
            rate: 拉伸比率（>1为拉长，<1为压缩）
        """
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(self, audio, n_steps=0):
        """
        音高调整
        
        Args:
            audio: 输入音频
            n_steps: 半音数（正数为升高，负数为降低）
        """
        return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
    
    def add_reverb(self, audio, room_size=0.5):
        """
        添加混响效果（简单实现）
        
        Args:
            audio: 输入音频
            room_size: 房间大小参数（0-1）
        """
        # 创建简单的混响效果
        delay = int(self.sample_rate * 0.05 * room_size)  # 延迟时间
        decay = 0.3 * room_size  # 衰减系数
        
        reverb = np.zeros(len(audio) + delay)
        reverb[:len(audio)] = audio
        reverb[delay:] += audio * decay
        
        return reverb[:len(audio)]
    
    def repeat_and_fade(self, audio, num_repeats=3):
        """
        重复音频并添加淡入淡出效果
        
        Args:
            audio: 输入音频
            num_repeats: 重复次数
        """
        repeated = np.tile(audio, num_repeats)
        
        # 添加淡入淡出
        fade_length = int(0.1 * self.sample_rate)  # 0.1秒淡入淡出
        fade_in = np.linspace(0, 1, fade_length)
        fade_out = np.linspace(1, 0, fade_length)
        
        if len(repeated) > fade_length * 2:
            repeated[:fade_length] *= fade_in
            repeated[-fade_length:] *= fade_out
        
        return repeated
    
    def pad_with_silence(self, audio, silence_before=0.5, silence_after=0.5):
        """
        在音频前后添加静音
        
        Args:
            audio: 输入音频
            silence_before: 前面的静音时长（秒）
            silence_after: 后面的静音时长（秒）
        """
        silence_before_samples = int(silence_before * self.sample_rate)
        silence_after_samples = int(silence_after * self.sample_rate)
        
        padded = np.concatenate([
            np.zeros(silence_before_samples),
            audio,
            np.zeros(silence_after_samples)
        ])
        
        return padded
    
    def extend_to_target_length(self, audio, method='stretch_and_repeat'):
        """
        将音频扩展到目标长度
        
        Args:
            audio: 输入音频
            method: 扩展方法
                - 'stretch': 纯时间拉伸
                - 'repeat': 重复
                - 'stretch_and_repeat': 时间拉伸 + 重复
                - 'pad_silence': 添加静音填充
        """
        current_length = len(audio)
        
        if current_length >= self.target_length:
            return audio[:self.target_length]
        
        if method == 'stretch':
            # 计算需要的拉伸比率
            rate = current_length / self.target_length
            stretched = self.time_stretch(audio, rate=rate)
            return stretched[:self.target_length]
        
        elif method == 'repeat':
            # 计算需要重复的次数
            num_repeats = int(np.ceil(self.target_length / current_length))
            repeated = np.tile(audio, num_repeats)
            return repeated[:self.target_length]
        
        elif method == 'stretch_and_repeat':
            # 先拉伸到一定长度
            stretch_rate = 0.7  # 拉伸到70%速度
            stretched = self.time_stretch(audio, rate=stretch_rate)
            
            # 然后重复到目标长度
            if len(stretched) < self.target_length:
                num_repeats = int(np.ceil(self.target_length / len(stretched)))
                result = np.tile(stretched, num_repeats)
                return result[:self.target_length]
            else:
                return stretched[:self.target_length]
        
        elif method == 'pad_silence':
            # 计算需要的静音长度
            silence_needed = self.target_length - current_length
            silence_before = silence_needed // 2
            silence_after = silence_needed - silence_before
            
            return np.concatenate([
                np.zeros(silence_before),
                audio,
                np.zeros(silence_after)
            ])
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def augment_audio(self, audio, method='comprehensive', noise_level=0.005):
        """
        综合音频扩展方法
        
        Args:
            audio: 输入音频
            method: 扩展方法
                - 'simple': 简单重复
                - 'stretch': 时间拉伸
                - 'comprehensive': 综合方法（推荐）
            noise_level: 噪声强度
        """
        if method == 'simple':
            # 简单重复
            extended = self.extend_to_target_length(audio, method='repeat')
        
        elif method == 'stretch':
            # 时间拉伸
            extended = self.extend_to_target_length(audio, method='stretch')
        
        elif method == 'comprehensive':
            # 综合方法：时间拉伸 + 重复 + 噪声 + 混响
            # 1. 轻微时间拉伸
            stretched = self.time_stretch(audio, rate=0.8)
            
            # 2. 重复到目标长度
            num_repeats = int(np.ceil(self.target_length / len(stretched)))
            repeated = np.tile(stretched, num_repeats)[:self.target_length]
            
            # 3. 添加淡入淡出效果
            fade_length = int(0.05 * self.sample_rate)
            for i in range(1, num_repeats):
                start_idx = i * len(stretched)
                if start_idx < len(repeated) and start_idx + fade_length < len(repeated):
                    fade = np.linspace(0, 1, fade_length)
                    repeated[start_idx:start_idx + fade_length] *= fade
            
            # 4. 添加白噪声
            noisy = self.add_white_noise(repeated, noise_level=noise_level)
            
            # 5. 添加轻微混响
            extended = self.add_reverb(noisy, room_size=0.3)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # 确保长度正确
        if len(extended) > self.target_length:
            extended = extended[:self.target_length]
        elif len(extended) < self.target_length:
            extended = np.pad(extended, (0, self.target_length - len(extended)))
        
        # 归一化
        extended = extended / np.max(np.abs(extended)) * 0.95
        
        return extended
    
    def save_audio(self, audio, output_path):
        """保存音频文件"""
        # 确保音频在有效范围内
        audio = np.clip(audio, -1.0, 1.0)
        wavfile.write(output_path, self.sample_rate, audio.astype(np.float32))
        print(f"音频已保存到: {output_path}")
        print(f"音频时长: {len(audio) / self.sample_rate:.2f} 秒")


def main():
    parser = argparse.ArgumentParser(description='音频扩展工具')
    parser.add_argument('--input', type=str, default='output.wav',
                        help='输入音频文件路径')
    parser.add_argument('--output', type=str, default='output_extended.wav',
                        help='输出音频文件路径')
    parser.add_argument('--duration', type=float, default=5.0,
                        help='目标音频长度（秒）')
    parser.add_argument('--sample_rate', type=int, default=22050,
                        help='采样率')
    parser.add_argument('--method', type=str, default='comprehensive',
                        choices=['simple', 'stretch', 'comprehensive'],
                        help='扩展方法')
    parser.add_argument('--noise_level', type=float, default=0.005,
                        help='噪声强度')
    
    args = parser.parse_args()
    
    # 创建扩展器
    augmentor = AudioAugmentor(
        target_duration=args.duration,
        sample_rate=args.sample_rate
    )
    
    # 加载音频
    print(f"加载音频: {args.input}")
    audio, sr = augmentor.load_audio(args.input)
    print(f"原始音频时长: {len(audio) / sr:.2f} 秒")
    
    # 扩展音频
    print(f"使用方法: {args.method}")
    extended_audio = augmentor.augment_audio(
        audio,
        method=args.method,
        noise_level=args.noise_level
    )
    
    # 保存音频
    augmentor.save_audio(extended_audio, args.output)


if __name__ == '__main__':
    main()
