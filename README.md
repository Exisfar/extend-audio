# 音频扩展工具 - 完整使用指南

## 📋 概述

这套工具用于将TTS生成的短命令音频（如"OPEN THE DOOR"）扩展到5秒，以便用于通用对抗样本训练。

## 📦 创建的文件

```
├── inference.ipynb              # TTS生成音频（如VITS）
├── audio_augmentation.py        # 高级音频扩展工具（命令行）
├── extend_audio.py              # 简单易用的扩展脚本（推荐）
```

## 🚀 快速开始

### 步骤1: 安装依赖

```bash
pip install numpy scipy librosa soundfile
```

### 步骤2: 生成TTS音频

打开并运行 `inference.ipynb`，会生成 `output.wav`

### 步骤3: 扩展音频

```bash
python extend_audio.py
```

完成！现在你有了扩展到5秒的音频文件。

## 📝 使用方法

### 方法1: 使用简单脚本（推荐新手）

```bash
python extend_audio.py
```

**输出文件：**
- `output_extended_repeat.wav` - 简单重复方法
- `output_extended_stretch.wav` - 时间拉伸方法
- `output_extended_comprehensive.wav` - **综合方法（推荐）**
- `output_extended_silence.wav` - 静音填充方法
- `output_extended_variant_1.wav` - 变体1（数据增强）
- `output_extended_variant_2.wav` - 变体2（数据增强）
- `output_extended_variant_3.wav` - 变体3（数据增强）

**推荐使用：** `output_extended_comprehensive.wav`

### 方法2: 使用高级工具（支持自定义参数）

```bash
# 基本使用
python audio_augmentation.py --input output.wav --output my_extended.wav

# 自定义时长和噪声
python audio_augmentation.py \
    --input output.wav \
    --output my_extended.wav \
    --duration 5.0 \
    --method comprehensive \
    --noise_level 0.005
```

**可用参数：**
- `--input`: 输入音频文件
- `--output`: 输出音频文件
- `--duration`: 目标时长（秒）
- `--method`: 扩展方法（simple/stretch/comprehensive）
- `--noise_level`: 噪声水平（0.001-0.01）

### 方法3: 在 Jupyter Notebook 中使用

打开 `inference.ipynb`，运行音频扩展单元格：

```python
# 已经集成在 notebook 中，直接运行相应的 cell 即可
audio_extended = extend_audio_for_adversarial(audio, hps.data.sampling_rate, target_duration=5.0)
write("output_extended_5s.wav", hps.data.sampling_rate, audio_extended)
```

## 🎯 扩展方法对比

| 方法           | 优点       | 缺点       | 推荐场景             |
| -------------- | ---------- | ---------- | -------------------- |
| **简单重复**   | 保持清晰度 | 明显重复感 | 需要保持原始质量     |
| **时间拉伸**   | 减少重复感 | 略微失真   | 需要更自然的音频     |
| **综合方法** ⭐ | 最自然     | 略慢       | 对抗样本训练（推荐） |
| **静音填充**   | 保持完整性 | 密度低     | 需要命令间隔         |

## 💡 应用示例

### 示例1: 批量处理多个命令

```python
commands = ["OPEN THE DOOR", "TURN ON THE LIGHT", "PLAY MUSIC"]

for cmd in commands:
    # 1. 使用TTS生成音频
    stn_tst = get_text(cmd, hps)
    audio = net_g.infer(...)[0][0,0].data.cpu().float().numpy()
    
    # 2. 保存原始音频
    write(f"{cmd.replace(' ', '_').lower()}.wav", sr, audio)
    
    # 3. 扩展音频
    os.system(f"python audio_augmentation.py --input {cmd.replace(' ', '_').lower()}.wav --output {cmd.replace(' ', '_').lower()}_extended.wav")
```

### 示例2: 生成数据增强变体

```bash
# 为一个命令生成10个不同的变体
for i in {1..10}; do
    python audio_augmentation.py \
        --input output.wav \
        --output variant_$i.wav \
        --method comprehensive \
        --noise_level $(awk -v seed=$RANDOM 'BEGIN{srand(seed); print 0.003 + rand()*0.005}')
done
```

### 示例3: 集成到训练脚本

```python
from audio_augmentation import AudioAugmentor

# 初始化
augmentor = AudioAugmentor(target_duration=5.0, sample_rate=22050)

# 准备训练数据
def prepare_training_data(tts_audio_files):
    dataset = []
    for audio_file in tts_audio_files:
        audio, sr = augmentor.load_audio(audio_file)
        
        # 生成5个变体
        for i in range(5):
            extended = augmentor.augment_audio(audio, method='comprehensive')
            dataset.append(extended)
    
    return dataset
```

## 🔧 技术细节

### 扩展算法流程

```
原始音频 (1秒)
    ↓
1. 时间拉伸 (→ 1.25秒, 保持音高)
    ↓
2. 重复 (→ 5秒)
    ↓
3. 交叉淡化 (平滑过渡)
    ↓
4. 添加白噪声 (增加真实感)
    ↓
5. 混响效果 (模拟环境)
    ↓
6. 归一化 (→ 最终5秒音频)
```

### 参数调优建议

**噪声水平 (noise_level):**
- 0.001-0.003: 轻微噪声，适合高质量训练
- 0.003-0.006: 中等噪声，推荐（默认0.005）
- 0.006-0.010: 较强噪声，适合鲁棒性训练

**时间拉伸比率:**
- 0.7-0.8: 明显减慢（推荐0.8）
- 0.8-0.9: 轻微减慢
- 0.9-1.0: 几乎不变

**混响强度 (room_size):**
- 0.1-0.3: 小房间效果（推荐0.3）
- 0.3-0.5: 中等房间
- 0.5-0.8: 大厅效果

## 📊 预期输出

```
原始音频 (output.wav):
- 时长: ~1.2秒
- 采样率: 22050Hz
- 文件大小: ~53KB

扩展音频 (output_extended_comprehensive.wav):
- 时长: 5.00秒 ✓
- 采样率: 22050Hz
- 文件大小: ~220KB
- 质量: 适合对抗样本训练 ✓
```

## ❓ 常见问题

### Q1: 扩展后的音频听起来不自然怎么办？
**A:** 尝试降低噪声水平或使用 `stretch_and_repeat` 方法：
```bash
python audio_augmentation.py --noise_level 0.002 --method stretch
```

### Q2: 如何生成不同长度的音频？
**A:** 使用 `--duration` 参数：
```bash
python audio_augmentation.py --duration 3.0  # 3秒
python audio_augmentation.py --duration 10.0  # 10秒
```

### Q3: 可以处理其他格式的音频吗？
**A:** 可以，librosa 支持多种格式（mp3, flac, ogg等）：
```bash
python audio_augmentation.py --input my_audio.mp3 --output extended.wav
```

### Q4: 如何批量处理大量文件？
**A:** 使用 shell 脚本：
```bash
for file in *.wav; do
    python audio_augmentation.py --input "$file" --output "${file%.wav}_extended.wav"
done
```