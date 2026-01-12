"""
简单的音频扩展脚本
直接处理 output.wav 并生成多个扩展版本
"""

import numpy as np
from scipy.io import wavfile
import librosa
import os


def load_audio(audio_path, target_sr=22050):
    """加载音频"""
    audio, sr = librosa.load(audio_path, sr=target_sr)
    return audio, sr


def add_white_noise(audio, noise_level=0.005):
    """添加白噪声"""
    noise = np.random.normal(0, noise_level, len(audio))
    return audio + noise


def simple_reverb(audio, sr, room_size=0.3):
    """简单混响效果"""
    delay = int(sr * 0.05 * room_size)
    decay = 0.3 * room_size
    
    reverb = np.zeros(len(audio) + delay)
    reverb[:len(audio)] = audio
    reverb[delay:] += audio * decay
    
    return reverb[:len(audio)]


def extend_audio_method1_repeat(audio, sr, target_duration=5.0):
    """
    方法1: 重复音频
    简单地将音频重复多次达到目标长度
    """
    target_length = int(target_duration * sr)
    current_length = len(audio)
    
    # 计算需要重复的次数
    num_repeats = int(np.ceil(target_length / current_length))
    
    # 重复音频
    extended = np.tile(audio, num_repeats)
    
    # 添加淡入淡出，使重复处平滑
    fade_length = int(0.05 * sr)  # 50ms淡入淡出
    for i in range(1, num_repeats):
        start_idx = i * current_length
        if start_idx < len(extended) and start_idx + fade_length < len(extended):
            # 交叉淡入淡出
            fade_out = np.linspace(1, 0, fade_length)
            fade_in = np.linspace(0, 1, fade_length)
            
            if start_idx - fade_length >= 0:
                extended[start_idx - fade_length:start_idx] *= fade_out
            extended[start_idx:start_idx + fade_length] *= fade_in
    
    # 裁剪到目标长度
    extended = extended[:target_length]
    
    return extended


def extend_audio_method2_stretch_repeat(audio, sr, target_duration=5.0):
    """
    方法2: 时间拉伸 + 重复
    先将音频拉伸（减慢速度），然后重复到目标长度
    """
    target_length = int(target_duration * sr)
    
    # 时间拉伸（减慢到原来的80%）
    stretched = librosa.effects.time_stretch(audio, rate=0.8)
    
    # 重复到目标长度
    if len(stretched) < target_length:
        num_repeats = int(np.ceil(target_length / len(stretched)))
        extended = np.tile(stretched, num_repeats)
    else:
        extended = stretched
    
    # 裁剪到目标长度
    extended = extended[:target_length]
    
    return extended


def extend_audio_method3_comprehensive(audio, sr, target_duration=5.0, noise_level=0.005):
    """
    方法3: 综合方法
    时间拉伸 + 重复 + 噪声 + 混响
    推荐使用此方法，效果最自然
    """
    target_length = int(target_duration * sr)
    
    # 1. 时间拉伸（减慢到原来的75-85%之间，随机选择）
    stretch_rate = np.random.uniform(0.75, 0.85)
    stretched = librosa.effects.time_stretch(audio, rate=stretch_rate)
    
    # 2. 重复到目标长度
    num_repeats = int(np.ceil(target_length / len(stretched)))
    repeated = np.tile(stretched, num_repeats)[:target_length]
    
    # 3. 添加平滑的交叉淡化
    fade_length = int(0.05 * sr)
    for i in range(1, num_repeats):
        start_idx = i * len(stretched)
        if start_idx < len(repeated) and start_idx + fade_length < len(repeated):
            fade = np.linspace(0, 1, fade_length)
            repeated[start_idx:start_idx + fade_length] *= fade
    
    # 4. 添加轻微白噪声
    noisy = add_white_noise(repeated, noise_level=noise_level)
    
    # 5. 添加混响效果
    extended = simple_reverb(noisy, sr, room_size=0.3)
    
    # 6. 裁剪到精确长度
    extended = extended[:target_length]
    
    return extended


def extend_audio_method4_silence_padding(audio, sr, target_duration=5.0):
    """
    方法4: 静音填充
    在音频前后添加静音，适合需要保持原始命令清晰度的场景
    """
    target_length = int(target_duration * sr)
    current_length = len(audio)
    
    # 计算需要的静音长度
    silence_needed = target_length - current_length
    
    # 在前后各添加一半静音
    silence_before = silence_needed // 2
    silence_after = silence_needed - silence_before
    
    # 添加轻微噪声到静音部分，使其更自然
    noise_before = np.random.normal(0, 0.001, silence_before)
    noise_after = np.random.normal(0, 0.001, silence_after)
    
    extended = np.concatenate([noise_before, audio, noise_after])
    
    return extended


def normalize_audio(audio, target_peak=0.95):
    """归一化音频"""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * target_peak
    return audio


def save_audio(audio, sr, output_path):
    """保存音频"""
    # 归一化并限制范围
    audio = normalize_audio(audio)
    audio = np.clip(audio, -1.0, 1.0)
    
    # 保存为float32格式
    wavfile.write(output_path, sr, audio.astype(np.float32))
    
    duration = len(audio) / sr
    print(f"✓ 已保存: {output_path}")
    print(f"  时长: {duration:.2f}秒")


def main():
    # 配置参数
    input_file = "OPENTHEDOOR.wav"  # TTS生成的原始音频
    target_duration = 5.0  # 目标时长（秒）
    sample_rate = 22050  # 采样率
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        print("请先运行 inference.ipynb 生成 output.wav")
        return
    
    # 加载音频
    print(f"\n加载音频: {input_file}")
    audio, sr = load_audio(input_file, target_sr=sample_rate)
    print(f"原始时长: {len(audio) / sr:.2f}秒")
    print(f"目标时长: {target_duration}秒")
    print(f"采样率: {sr}Hz")
    
    print("\n" + "="*60)
    print("开始生成扩展音频...")
    print("="*60)
    
    # 方法1: 简单重复
    print("\n[方法1] 简单重复...")
    extended1 = extend_audio_method1_repeat(audio, sr, target_duration)
    save_audio(extended1, sr, "output_extended_repeat.wav")
    
    # 方法2: 时间拉伸 + 重复
    print("\n[方法2] 时间拉伸 + 重复...")
    extended2 = extend_audio_method2_stretch_repeat(audio, sr, target_duration)
    save_audio(extended2, sr, "output_extended_stretch.wav")
    
    # 方法3: 综合方法（推荐）
    print("\n[方法3] 综合方法（推荐）...")
    extended3 = extend_audio_method3_comprehensive(audio, sr, target_duration, noise_level=0.005)
    save_audio(extended3, sr, "output_extended_comprehensive.wav")
    
    # 方法4: 静音填充
    print("\n[方法4] 静音填充...")
    extended4 = extend_audio_method4_silence_padding(audio, sr, target_duration)
    save_audio(extended4, sr, "output_extended_silence.wav")
    
    # 生成多个变体（用于数据增强）
    print("\n" + "="*60)
    print("生成额外的变体（用于数据增强）...")
    print("="*60)
    
    for i in range(3):
        print(f"\n[变体 {i+1}]")
        # 使用不同的噪声水平
        noise_level = np.random.uniform(0.003, 0.008)
        variant = extend_audio_method3_comprehensive(audio, sr, target_duration, noise_level)
        output_name = f"output_extended_variant_{i+1}.wav"
        save_audio(variant, sr, output_name)
    
    print("\n" + "="*60)
    print("完成！")
    print("="*60)
    print("\n生成的文件:")
    print("  1. output_extended_repeat.wav        - 简单重复")
    print("  2. output_extended_stretch.wav       - 时间拉伸+重复")
    print("  3. output_extended_comprehensive.wav - 综合方法（推荐）")
    print("  4. output_extended_silence.wav       - 静音填充")
    print("  5. output_extended_variant_1-3.wav   - 额外变体")
    print("\n推荐使用: output_extended_comprehensive.wav")
    print("或使用变体文件进行数据增强训练")


if __name__ == '__main__':
    main()
