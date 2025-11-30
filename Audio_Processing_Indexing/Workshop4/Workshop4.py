import time
import torch
import soundfile as sf
from gtts import gTTS
import ChatTTS
import numpy as np

# ==========================================
# 1. 准备评估文本 (源自作业文档 Page 2)
# ==========================================
EVAL_TEXT = (
    "He found himself standing in a landscape that looked exactly like a giant chessboard. "
    "On every black square there was a monster: there were two-tongued snakes and lions "
    "with three rows of teeth, and four-headed dogs and five-headed demon kings and so on. "
    "He was, so to speak, looking out through the eyes of the young hero of the story. "
    "It was like being in the passenger seat of an automobile: all he had to do was watch, "
    "while the hero dispatched one monster after another and advanced up the chessboard "
    "towards the white stone tower at the end."
)


def run_gtts_baseline():
    print("--- 正在运行 Task A: gTTS (Baseline) ---")

    start_time = time.time()

    # 生成语音
    tts = gTTS(text=EVAL_TEXT, lang='en')

    # 保存文件
    output_filename = "tacotron.wav"
    tts.save(output_filename)

    end_time = time.time()
    latency = end_time - start_time

    print(f"✅ gTTS 生成完毕: {output_filename}")
    print(f"⏱️ gTTS 耗时 (Latency): {latency:.4f} 秒\n")
    return latency


def run_chattts_sota():
    print("--- 正在运行 Task B: ChatTTS (SOTA) ---")

    # 1. 初始化模型
    print("正在加载 ChatTTS 模型 (首次运行可能需要下载权重)...")
    chat = ChatTTS.Chat()

    # 【修复点】新版 ChatTTS 使用 .load() 而不是 .load_models()
    # compile=False 能提高兼容性，避免某些 Windows 环境下的编译错误
    try:
        chat.load(compile=False)
    except AttributeError:
        # 为了防备极少数旧版本，保留一个回退（虽然不太可能）
        chat.load_models(compile=False)

    # 2. 推理 (生成)
    print("正在生成语音 (SOTA 模型通常较慢，请耐心等待)...")

    start_time = time.time()

    # infer 返回一个列表
    wavs = chat.infer([EVAL_TEXT])

    end_time = time.time()
    latency = end_time - start_time

    # 3. 保存文件
    output_filename = "chattts_natural.wav"

    if wavs and len(wavs) > 0:
        # 获取音频数据 (通常是第一个元素)
        audio_data = wavs[0]

        # 确保数据是 numpy 数组 (如果是 Tensor 则转换)
        if isinstance(audio_data, (list, tuple)):
            audio_data = audio_data[0]  # 有时它嵌套在列表中
        if hasattr(audio_data, 'cpu'):
            audio_data = audio_data.cpu().numpy()
        if isinstance(audio_data, np.ndarray) and audio_data.ndim > 1:
            audio_data = audio_data.flatten()  # 展平多维数组

        # ChatTTS 默认采样率为 24000
        sf.write(output_filename, audio_data, 24000)
        print(f"✅ ChatTTS 生成完毕: {output_filename}")
        print(f"⏱️ ChatTTS 推理耗时 (Latency): {latency:.4f} 秒\n")
    else:
        print("❌ ChatTTS 生成失败，未返回音频数据。")
        latency = 0

    return latency


if __name__ == "__main__":
    # 运行任务 A
    t_gtts = run_gtts_baseline()

    # 运行任务 B
    try:
        t_chat = run_chattts_sota()
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"❌ 运行 ChatTTS 时发生严重错误: {e}")
        t_chat = 0

    # 结果汇总
    print("=" * 30)
    print("实验结果汇总 (用于 Technical Note)")
    print("=" * 30)
    print(f"1. Baseline (gTTS) Latency : {t_gtts:.4f} s")
    print(f"2. SOTA (ChatTTS) Latency  : {t_chat:.4f} s")
    print("=" * 30)
    print("提示：请务必试听 chattts_natural.wav，注意其中的呼吸声、停顿等细节。")