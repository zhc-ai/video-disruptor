import numpy as np
import soundfile as sf

sr = 16000
duration = 5.0
t = np.linspace(0, duration, int(sr * duration), endpoint=False)

# 构造扰动信号（高频 + 漂移 + 白噪）
hf_noise = 0.02 * np.sin(2 * np.pi * 8000 * t)
drift = 0.01 * np.sin(2 * np.pi * 30 * t)
white_noise = 0.005 * np.random.randn(len(t))

adv_audio = hf_noise + drift + white_noise
adv_audio /= np.max(np.abs(adv_audio))

# 保存为本地 wav 文件
sf.write("universal_adv_perturb.wav", adv_audio, sr)
print("✅ 已保存 universal_adv_perturb.wav")