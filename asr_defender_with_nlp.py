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
    audio, _ = librosa.load(path, sr=sr)
    return audio, sr

def save_audio(path, audio, sr=16000):
    sf.write(path, audio, sr)

def strong_perturb_segment(audio, sr):
    t = np.linspace(0, len(audio) / sr, len(audio), endpoint=False)
    hf = 0.02 * np.sin(2 * np.pi * 8000 * t)
    drift = 0.01 * np.sin(2 * np.pi * 20 * t)
    noise = 0.005 * np.random.randn(len(audio))
    perturbed = audio + hf + drift + noise
    perturbed /= np.max(np.abs(perturbed) + 1e-6)
    return perturbed

def time_drift(audio, sr):
    stretch_factor = np.random.uniform(0.95, 1.05)
    audio_stretched = librosa.effects.time_stretch(audio, rate=stretch_factor)
    target_len = len(audio)
    if len(audio_stretched) > target_len:
        return audio_stretched[:target_len]
    else:
        pad = np.zeros(target_len - len(audio_stretched))
        return np.concatenate([audio_stretched, pad])

def process_with_periodic_perturb(input_path, output_path, period_sec=5):
    audio, sr = load_audio(input_path)
    period_len = int(sr * period_sec)
    segments = []

    for i in range(0, len(audio), period_len):
        segment = audio[i:i + period_len]
        if len(segment) < period_len:
            # 最后不足 5 秒的段不做扰动，保持原样
            segments.append(segment)
            break

        drifted = time_drift(segment, sr)
        perturbed = strong_perturb_segment(drifted, sr)
        segments.append(perturbed)

    full_audio = np.concatenate(segments)
    save_audio(output_path, full_audio, sr)
    return output_path

# === Whisper 转写 + NLP 相似度 ===
def whisper_transcribe(path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base").to(device)
    result = model.transcribe(path)
    return result["text"].strip()

def nlp_similarity(text1, text2):
    # 使用 jieba 分词并用空格连接，兼容 TF-IDF 中文处理
    seg1 = " ".join(jieba.cut(text1))
    seg2 = " ".join(jieba.cut(text2))
    vectorizer = TfidfVectorizer().fit([seg1, seg2])
    tfidf = vectorizer.transform([seg1, seg2])
    score = cosine_similarity(tfidf[0], tfidf[1])[0][0]
    return score

# === 主函数 ===
if __name__ == "__main__":
    input_audio_path = "input_audio.wav"
    output_audio_path = "perturbed_output.wav"

    # Step 1: 扰动处理
    process_with_periodic_perturb(input_audio_path, output_audio_path)

    # Step 2: Whisper 转写
    text_orig = whisper_transcribe(input_audio_path)
    text_pert = whisper_transcribe(output_audio_path)

    # Step 3: NLP 相似度
    similarity = nlp_similarity(text_orig, text_pert)
    similarity2 = nlp_similarity("我们都是中国人", "我们都中国人")

    # 输出结果
    print("🎙️ 原始识别文本:", text_orig)
    print("🌀 扰动后识别文本:", text_pert)
    print(f"🧠 NLP 相似度: {similarity:.4f}")
    print(f"🧠 NLP 相似度: {similarity2:.4f}")