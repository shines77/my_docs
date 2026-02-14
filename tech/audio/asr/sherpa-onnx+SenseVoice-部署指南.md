# 告别 ModelScope 魔搭联网依赖！sherpa-onnx + SenseVoice 完全离线语音识别部署指南（2026版，离线语音识别、声纹鉴定、sherpa-onnx、SenseVoice）

> **关键词**：离线语音识别、声纹鉴定、sherpa-onnx、SenseVoice、FunASR、ModelScope魔搭社区、Kaldi、语音转文字、声纹比对、说话人识别

**GitHub**: <https://github.com/k2-fsa/sherpa-onnx>  
**官方文档**: <https://k2-fsa.github.io/sherpa/onnx/index.html>

## 一、问题背景：为什么需要完全离线？

如果你用过阿里的 FunASR 或 SenseVoice，可能遇到过这个痛点：

**明明模型已经下载到本地了，为什么还要联网？**

这是 FunASR/SenseVoice 的一个已知问题（GitHub Issues #2573、#1581、#1286）：它们内嵌了 ModelScope SDK，即使模型缓存在本地，启动时仍会尝试连接 modelscope.cn 进行校验。官方提供的环境变量绕过方案并不可靠：

```python
# 官方建议的绕过方案，但实测并不总是有效
os.environ['MODELSCOPE_DISABLE_REMOTE']='1'
model = AutoModel(model="/local/path", disable_update=True)
```

对于以下场景，这种联网检查是致命的：

|  场景  |  问题  |
| :---: | :----: |
| 内网/涉密环境 | 完全不能联网 |
| 边缘设备部署 | 网络不稳定或无网络 |
| 司法/公安系统 | 数据安全合规要求 |
| 嵌入式设备 | 资源有限，无法承担网络开销 |

**本文给你一个彻底的解决方案：sherpa-onnx**

## 二、技术选型：从 Kaldi 到 sherpa-onnx 的演进

### 2.1 语音识别技术代际

| 时期 | 技术 | 代表项目 | 当前状态 |
| :---: | :----: | :---: | :----: |
| 2011-2019 | HMM/GMM/WFST | Kaldi | ❌ 已停止开发（2019年后无重大更新） |
| 2018-2020 | i-vector/x-vector | Kaldi | ⚠️ 被 ECAPA-TDNN 超越 |
| 2020-2022 | ECAPA-TDNN | SpeechBrain/3D-Speaker | ✅ 声纹识别主流（2024年司法鉴定论文仍在使用） |
| 2022-至今 | 端到端Transformer | Whisper/FunASR/SenseVoice | ✅ ASR主流 |
| 2020-至今 | 新一代Kaldi (k2) | sherpa-onnx | ✅ 离线部署首选 |

### 2.2 FunASR 和 SenseVoice 是什么关系？

很多人分不清这两个项目，这里理清一下：

```
FunASR（框架/工具包）
 │
 ├── Paraformer（ASR模型）
 ├── FSMN-VAD（语音端点检测模型）
 ├── CT-Transformer（标点模型）
 └── SenseVoice（多功能模型）← 单独拿出来包装
```

SenseVoice 本质上是 FunASR 框架下的一个模型，但阿里为了营销（对标Whisper）单独开了仓库。它比普通ASR多了情感识别和音频事件检测功能。

**关键问题**：两者底层都依赖 ModelScope SDK，所以联网检查问题是共享的。

### 2.3 为什么选 sherpa-onnx？

sherpa-onnx 是 Kaldi 之父 Daniel Povey 加入小米后主导的"新一代 Kaldi"项目（k2-fsa）的推理引擎。

| 对比维度 | FunASR / SenseVoice | sherpa-onnx |
| :--- | :---: | :---: |
| **联网依赖** | ⚠️ ModelScope SDK 会检查 | ❌ 完全不需要 |
| **部署方式** | pip install + 自动下载 | 本地文件路径，零联网 |
| **依赖体积** | 大（PyTorch全家桶） | 小（纯ONNX Runtime） |
| **跨平台** | 主要服务器端 | ✅ 手机/树莓派/浏览器/RISC-V/RK3588/华为昇腾 |
|**语言支持**|Python为主|12种语言（C/C++/Python/Go/Rust/Swift/JS/Kotlin…）|
| **模型来源** | 从 ModelScope 下载 | 和 SenseVoice 相同，但已转换为 ONNX 格式，效果一致 |

**一句话总结：sherpa-onnx = 指定本地文件路径，完事。永不联网。**

## 三、快速上手

### 3.1 安装

```bash
pip install sherpa-onnx
```

就这一行。没有 modelscope，没有 funasr，没有任何隐藏的网络依赖。

### 3.2 下载模型

从 GitHub Releases 下载预转换的 ONNX 模型：

```bash
# SenseVoice 模型（中英日韩粤，229MB int8版本）
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
```

*注意：GitHub 地址可能需要代理访问，具体链接请参考文末资源汇总。*

**模型文件结构**：

```bash
sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/
├── model.int8.onnx   # 229MB，量化版，推荐
├── model.onnx        # 895MB，完整版
├── tokens.txt        # 词表
└── test_wavs/        # 测试音频
```

### 3.3 基础用法

```bash
import sherpa_onnx
import wave
import numpy as np

# 创建识别器（指定本地路径，零联网）
recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
    model="./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx",
    tokens="./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt",
    num_threads=4,
    use_itn=True,      # 启用逆文本正则化（数字、日期等转换）
    language="zh",     # 语言：zh/en/ja/ko/yue/auto
)

# 读取音频
def read_wave(wav_path):
    with wave.open(wav_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        data = wf.readframes(wf.getnframes())
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sample_rate

# 识别
samples, sample_rate = read_wave("test.wav")
stream = recognizer.create_stream()
stream.accept_waveform(sample_rate, samples)
recognizer.decode_stream(stream)

print(stream.result.text)
```

## 四、实战：批量视频/音频转文字工具

以下是一个完整的批量转写工具，支持递归处理目录、保持层级结构、跳过已处理文件：

```python
"""
视频/音频批量转文字工具
使用 sherpa-onnx + SenseVoice 模型
完全离线，零联网依赖
"""

import os
import subprocess
import wave
import numpy as np
from pathlib import Path
import sherpa_onnx

# ============ 配置区 ============
INPUT_DIR = r"C:\Users\xxx\Desktop\测试视频"      # 输入目录
OUTPUT_DIR = r"C:\Users\xxx\Desktop\输出文件txt"  # 输出目录
MODEL_DIR = "./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
# ================================

VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.ts', '.mts'}
AUDIO_EXTS = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma', '.opus', '.amr'}
ALL_EXTS = VIDEO_EXTS | AUDIO_EXTS

def create_recognizer():
    """创建离线识别器"""
    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=f"{MODEL_DIR}/model.int8.onnx",
        tokens=f"{MODEL_DIR}/tokens.txt",
        num_threads=4,
        use_itn=True,      # 启用逆文本正则化（数字、日期等转换）
        language="zh",     # 语言：zh/en/ja/ko/yue/auto
    )

def read_wave(wav_path: str):
    """读取WAV文件"""
    with wave.open(wav_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        data = wf.readframes(num_frames)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sample_rate

def extract_audio(input_path: str, output_wav: str) -> bool:
    """使用ffmpeg提取/转换音频为16kHz单声道WAV"""
    try:
        cmd = [
            "ffmpeg", "-i", input_path,
            "-ar", "16000",      # 采样率16kHz
            "-ac", "1",          # 单声道
            "-f", "wav",
            "-y", "-loglevel", "error",
            output_wav
        ]
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"  ✗ ffmpeg转换失败")
        return False

def transcribe_audio(recognizer, wav_path: str) -> str:
    """语音识别"""
    samples, sample_rate = read_wave(wav_path)
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    return stream.result.text

def process_file(recognizer, file_path: Path, input_root: Path, 
                 output_root: Path, temp_wav: str) -> bool:
    """处理单个文件"""
    # 计算相对路径，保持层级结构
    rel_path = file_path.relative_to(input_root)
    txt_path = output_root / rel_path.with_suffix('.txt')
    
    # 创建输出子目录
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 跳过已处理
    if txt_path.exists():
        print(f"  ⊘ 已存在，跳过")
        return True
    
    # 转换音频
    print(f"  → 转换音频...")
    if not extract_audio(str(file_path), temp_wav):
        return False
    
    # 识别
    print(f"  → 识别中...")
    try:
        text = transcribe_audio(recognizer, temp_wav)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"  ✓ 完成 -> {txt_path.name}")
        return True
    except Exception as e:
        print(f"  ✗ 识别失败: {e}")
        return False

# ============ 主程序 ============
input_path = Path(INPUT_DIR)
output_path = Path(OUTPUT_DIR)

if not os.path.exists(MODEL_DIR):
    print(f"错误: 模型目录不存在 - {MODEL_DIR}")
elif not input_path.exists():
    print(f"错误: 输入目录不存在 - {INPUT_DIR}")
else:
    # 创建输出根目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 收集媒体文件（排除输出目录）
    media_files = []
    for ext in ALL_EXTS:
        for f in input_path.rglob(f"*{ext}"):
            if not str(f).startswith(str(output_path)):
                media_files.append(f)
        for f in input_path.rglob(f"*{ext.upper()}"):
            if not str(f).startswith(str(output_path)):
                media_files.append(f)
    media_files = sorted(set(media_files))
    
    if not media_files:
        print("未找到任何媒体文件")
    else:
        print(f"找到 {len(media_files)} 个媒体文件")
        print(f"输出目录: {OUTPUT_DIR}\n")
        print("加载模型...")
        recognizer = create_recognizer()
        
        temp_wav = os.path.join(os.environ.get('TEMP', '/tmp'), 'temp_audio.wav')
        success, failed = 0, 0
        
        for i, file_path in enumerate(media_files, 1):
            rel_path = file_path.relative_to(input_path)
            print(f"\n[{i}/{len(media_files)}] {rel_path}")
            if process_file(recognizer, file_path, input_path, output_path, temp_wav):
                success += 1
            else:
                failed += 1
        
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        
        print(f"\n{'='*40}")
        print(f"完成: 成功 {success}, 失败 {failed}")
        print(f"输出位置: {OUTPUT_DIR}")
```

以下是一个完整的批量音频文件转写工具。

```python
import os
import sherpa_onnx
from pathlib import Path

# 初始化识别器（以SenseVoice模型为例）
recognizer = sherpa_onnx.OfflineRecognizer(
    encoder="./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-06-26/model.int8.onnx",
    decoder="./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-06-26/model.int8.onnx",
    tokens="./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-06-26/tokens.txt",
    num_threads=2,
    sample_rate=16000,
    decoding_method="greedy_search",
    debug=False,
)

def transcribe_audio(audio_path):
    """转写单个音频文件"""
    try:
        audio = sherpa_onnx.read_audio(audio_path)
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, audio)
        stream.input_finished()
        recognizer.decode_stream(stream)
        return stream.result.text
    except Exception as e:
        print(f"处理失败 {audio_path}: {e}")
        return None

def batch_process(input_dir, output_dir):
    """批量处理目录下的所有音频文件"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    for audio_file in input_path.rglob("*.wav"):  # 支持 wav, mp3, m4a 等
        relative = audio_file.relative_to(input_path)
        txt_file = output_path / relative.with_suffix(".txt")
        
        if txt_file.exists():  # 跳过已处理文件
            continue
            
        txt_file.parent.mkdir(parents=True, exist_ok=True)
        text = transcribe_audio(str(audio_file))
        if text:
            txt_file.write_text(text, encoding="utf-8")
            print(f"已转写: {audio_file} -> {txt_file}")

if __name__ == "__main__":
    batch_process("./audios", "./transcripts")
```

## 五、其他功能：不只是语音识别

sherpa-onnx 支持的功能远不止 ASR（GitHub需代理）：

| 功能 | 说明 | 模型下载 |
| :---: | :---: | :---: |
| 语音识别 (ASR) | SenseVoice/Whisper/Paraformer等 | asr-models |
| 文本转语音 (TTS) | 多种中英文语音，支持VITS等模型 | tts-models |
| 声纹识别/鉴定 | 说话人验证/识别<br/>支持3D-Speaker等模型，见下节 | speaker-recongition-models |
| 语音端点检测 (VAD) |检测人声起止，Silero VAD | asr-models (silero_vad.onnx) |
| 关键词唤醒 (KWS) | 语音唤醒，支持自定义唤醒词 | kws-models |
| 说话人分离 | Speaker Diarization | 同speaker-recognition |
| 语音增强 | 降噪 (GTCRN) | speech-enhancement-models |

## 六、声纹识别/鉴定场景

如果你的需求是声纹比对（判断两段音频是否同一人），sherpa-onnx 同样支持：

```python
import sherpa_onnx
import numpy as np

# 创建声纹提取器（使用阿里3D-Speaker的ECAPA-TDNN模型）
extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
    model="./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
)

def get_embedding(wav_path):
    """提取声纹向量"""
    samples, sample_rate = read_wave(wav_path)  # 需要实现read_wave函数
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate, samples)
    stream.input_finished()
    return extractor.compute(stream)

# 提取两段音频的声纹
emb1 = get_embedding("speaker1.wav")
emb2 = get_embedding("speaker2.wav")

# 计算余弦相似度
similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
print(f"相似度: {similarity:.4f}") 

# 通常 > 0.5 认为是同一人，具体阈值需根据场景调整
```

**声纹模型下载**：

```bash
# 阿里 3D-Speaker 的 ECAPA-TDNN 模型（当前声纹识别主流架构）
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
```

### 关于司法鉴定的重要说明

声纹识别用于技术验证没问题，但如果是司法鉴定，关键不是用什么软件，而是需要有资质的鉴定机构出具报告。

| 层面 | 说明 |
| :--- | :--- |
| **技术层面** | ECAPA-TDNN 是2024年司法鉴定论文仍在使用的主流架构 |
| **法律层面** | 自行使用任何工具分析，都没有法律效力 |
| **合规流程** | 需要有资质的司法鉴定机构 + 规范的取证流程 + 正式鉴定报告 |

### 新挑战：AI语音合成检测

华东政法大学2024年研究指出，AI语音克隆对司法鉴定带来新挑战：

- **检测方法**：MFCC特征、共振峰分析、过渡特征
- **关键发现**：AI生成语音的过渡特征呈"突变型"，人类语音呈"平滑型"
- **未来趋势**：声纹鉴定需要双重能力：身份验证 + Deepfake检测

## 七、与其他方案对比总结

| 维度 | Kaldi (传统) | FunASR/SenseVoice | Whisper | sherpa-onnx |
| :---: | :---: | :---: | :---: | :---: |
| 完全离线 | ✅ 但极难部署 | ⚠️ ModelScope检查 | ✅  | ✅ 最简单 |
| 中文效果 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐<br/>(用SenseVoice模型) |
| 部署难度 | 极难<br/>（3-6个月学习曲线） | 中等 | 中等 | 简单 |
| 推理速度 | 快 | 快<br/>（比Whisper快15倍） | 慢 | 快 |
| 硬件支持 | 仅CPU | CPU/GPU | CPU/GPU | **全平台（移动/Web/嵌入式）** |
| 跨平台 | ⚠️ 主要Linux | ⚠️ 服务器端 | ⚠️ 服务器端 | ✅ 12种语言+嵌入式+NPU |
| 依赖体积 | 大 | 大（PyTorch） | 大（PyTorch） | 小（ONNX Runtime） |
| 声纹识别 | ✅ (x-vector已过时) | 有限 | ❌ | ✅ (ECAPA-TDNN) |
| 维护状态 | ❌ 停止开发 | ✅ 活跃 | ✅ 活跃 | ✅ 活跃 |

### 场景化选择建议

| 场景 | 推荐方案
| :---: | :---: |
| 内网/离线部署 | sherpa-onnx |
| 中文语音识别（有网络） | FunASR/SenseVoice |
| 多语言/英文为主 | Whisper |
| 嵌入式/边缘设备 | sherpa-onnx |
| 声纹比对/说话人识别 | sherpa-onnx + 3D-Speaker |
| 学习ASR原理 | 阅读 Kaldi 文档，实践用现代框架 |

- **中文语音识别（有网络）**：FunASR（直接pip安装最方便）
- **离线/内网部署**：**sherpa-onnx + SenseVoice（本文方案）**
- **多语言/通用场景**：sherpa-onnx + Whisper（转换后的 ONNX 版）
- **学习ASR原理**：阅读 Kaldi 文档，实践用现代框架

## 八、资源汇总

（注意：GitHub 地址可能需要代理访问）

| 资源 | 地址 |
| :--- | :--- |
| **GitHub 仓库** | <https://github.com/k2-fsa/sherpa-onnx> |
| **官方文档** | <https://k2-fsa.github.io/sherpa/onnx/index.html> |
| **模型下载** | <https://github.com/k2-fsa/sherpa-onnx/releases> |
| **ASR 模型下载** | <https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models> |
| **SenseVoice ONNX模型** | <https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models> |
| **3D-Speaker声纹模型** | <https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recognition-models> |
| **VAD 模型下载** | <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx> |
| **TTS 模型下载** | <https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models> |
| **SenseVoice 专题** | <https://k2-fsa.github.io/sherpa/onnx/sense-voice/index.html> |
| **在线 Demo** | <https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition> |

## 九、常见问题

**Q1: 和直接用 FunASR 有什么区别？**
FunASR 底层依赖 ModelScope SDK，即使本地有模型也会尝试联网校验（GitHub Issues #2573、#1581、#1286 都在讨论这个问题）。sherpa-onnx 是纯本地文件加载，彻底解耦。

**Q2: 模型是一样的吗？**
是的。sherpa-onnx 使用的 SenseVoice 模型就是从阿里 ModelScope 转换来的 ONNX 格式，效果完全一致。转换脚本在模型目录的 `export-onnx.py`。

**Q3: 支持流式识别吗？**
支持。sherpa-onnx 同时提供：

- `OfflineRecognizer`（非流式，整段音频识别）
- `OnlineRecognizer`（流式，边说边识别）

**Q4: 性能如何？**
SenseVoice int8 模型在 CPU 上 10 秒音频识别约 70ms，比 Whisper 快约 15 倍。

**Q5: 支持哪些硬件？**
sherpa-onnx 支持极其广泛的硬件平台：

- **服务器**: x86-64, ARM64
- **移动端**: Android、iOS、HarmonyOS
- **浏览器**: WebAssembly
- **嵌入式**: Raspberry Pi, 树莓派

**Q6: Kaldi 还值得学吗？**
不建议投入时间学习 Kaldi 的实际部署（学习曲线3-6个月）。但 Kaldi 的文档仍然是理解 ASR 原理（声学模型、语言模型、解码器、WFST）的优质资源。

## 十、写在最后

如果你被 ModelScope 的联网检查困扰，或者需要在内网/边缘设备部署语音识别，**sherpa-onnx 是目前最省心的选择**：

1. `pip install sherpa-onnx`
2. 下载 ONNX 模型到本地
3. 指定路径，开始用

没有 ModelScope，没有 PyTorch 全家桶，没有任何隐藏的网络请求。

**这才是真正的"离线部署"。**

---
> 微信公众号：**小胡说技书** 不定时发布书籍杂谈

---
**版权声明**：本文为CSDN博主「小胡说技书」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
**原文链接**：[https://h-y-c.blog.csdn.net/article/details/157221056]()
