import numpy as np
import librosa
import soundfile as sf
import whisper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import jieba

# === 音频扰动核心逻辑 ===

def load_audio(path, sr=16000):
    """
    加载音频文件
    :param path: 音频文件路径
    :param sr: 采样率，默认16000Hz
    :return: 返回音频数据和采样率
    """
    audio, _ = librosa.load(path, sr=sr)
    return audio, sr

def save_audio(path, audio, sr=16000):
    """
    保存音频文件
    :param path: 保存路径
    :param audio: 音频数据
    :param sr: 采样率，默认16000Hz
    """
    sf.write(path, audio, sr)

def strong_perturb_segment(audio, sr):
    """
    对音频段添加强扰动
    :param audio: 输入音频段
    :param sr: 采样率
    :return: 添加扰动后的音频段
    """
    t = np.linspace(0, len(audio) / sr, len(audio), endpoint=False)  # 时间轴
    hf = 0.02 * np.sin(2 * np.pi * 8000 * t)  # 高频扰动
    drift = 0.01 * np.sin(2 * np.pi * 20 * t)  # 低频漂移
    noise = 0.005 * np.random.randn(len(audio))  # 随机噪声
    perturbed = audio + hf + drift + noise  # 添加扰动
    perturbed /= np.max(np.abs(perturbed) + 1e-6)  # 归一化
    return perturbed

def time_drift(audio, sr):
    """
    对音频进行时间漂移
    :param audio: 输入音频
    :param sr: 采样率
    :return: 时间漂移后的音频
    """
    stretch_factor = np.random.uniform(0.95, 1.05)  # 随机伸缩因子
    audio_stretched = librosa.effects.time_stretch(audio, rate=stretch_factor)  # 进行时间伸缩
    target_len = len(audio)  # 目标长度
    if len(audio_stretched) > target_len:
        return audio_stretched[:target_len]  # 截取到目标长度
    else:
        pad = np.zeros(target_len - len(audio_stretched))  # 补零
        return np.concatenate([audio_stretched, pad])  # 返回补零后的音频

def process_with_periodic_perturb(input_path, output_path, period_sec=5):
    """
    处理音频并添加周期性扰动
    :param input_path: 输入音频路径
    :param output_path: 输出音频路径
    :param period_sec: 每段的时长（秒），默认5秒
    :return: 输出音频路径
    """
    audio, sr = load_audio(input_path)  # 加载音频
    period_len = int(sr * period_sec)  # 计算每段的样本长度
    segments = []  # 存储处理后的音频段

    for i in range(0, len(audio), period_len):
        segment = audio[i:i + period_len]  # 切分音频段
        if len(segment) < period_len:
            # 最后不足 5 秒的段不做扰动，保持原样
            segments.append(segment)
            break

        drifted = time_drift(segment, sr)  # 进行时间漂移
        perturbed = strong_perturb_segment(drifted, sr)  # 添加扰动
        segments.append(perturbed)  # 存储扰动后的音频段

    full_audio = np.concatenate(segments)  # 合并所有音频段
    save_audio(output_path, full_audio, sr)  # 保存处理后的音频
    return output_path

# === Whisper 转写 + NLP 相似度 ===

def whisper_transcribe(path):
    """
    使用 Whisper 模型对音频进行转写
    :param path: 音频文件路径
    :return: 转写后的文本
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 检查是否有可用的 GPU
    model = whisper.load_model("base").to(device)  # 加载 Whisper 模型
    result = model.transcribe(path)  # 转写音频
    return result["text"].strip()  # 返回转写文本

def nlp_similarity(text1, text2):
    """
    计算两个文本之间的 NLP 相似度
    :param text1: 第一个文本
    :param text2: 第二个文本
    :return: 两个文本的相似度分数
    """
    # 使用 jieba 分词并用空格连接，兼容 TF-IDF 中文处理
    seg1 = " ".join(jieba.cut(text1))  # 对第一个文本分词
    seg2 = " ".join(jieba.cut(text2))  # 对第二个文本分词
    vectorizer = TfidfVectorizer().fit([seg1, seg2])  # 计算 TF-IDF
    tfidf = vectorizer.transform([seg1, seg2])  # 转换为 TF-IDF 矩阵
    score = cosine_similarity(tfidf[0], tfidf[1])[0][0]  # 计算余弦相似度
    return score

# === 主函数 ===
if __name__ == "__main__":
    input_audio_path = "input_audio.wav"  # 输入音频路径
    output_audio_path = "perturbed_output.wav"  # 输出音频路径

    # Step 1: 扰动处理
    process_with_periodic_perturb(input_audio_path, output_audio_path)

    # Step 2: Whisper 转写
    text_orig = whisper_transcribe(input_audio_path)  # 转写原始音频
    text_pert = whisper_transcribe(output_audio_path)  # 转写扰动后的音频

    # Step 3: NLP 相似度
    similarity = nlp_similarity(text_orig, text_pert)  # 计算转写文本相似度
    similarity2 = nlp_similarity("我们都是中国人", "我们都中国人")  # 示例相似度计算

    # 输出结果
    print("🎙️ 原始识别文本:", text_orig)  # 输出原始文本
    print("🌀 扰动后识别文本:", text_pert)  # 输出扰动后文本
    print(f"🧠 NLP 相似度: {similarity:.4f}")  # 输出相似度分数
    print(f"🧠 NLP 相似度: {similarity2:.4f}")  # 输出示例相似度分数