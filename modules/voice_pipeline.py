#!/usr/bin/env python3
"""
语音交互管线：录音 → ASR → LLM → TTS → 播放

硬件依赖（不可改动）:
  - USB 麦克风: plughw:2,0, 增益 29, AGC on
  - 录音前会停止 PipeWire, 用 arecord 直录, aplay 直播

对外接口（供 main.py 调用）:
  setup_audio()              — 初始化声卡（启动时调用一次）
  run_conversation_turn()    — 一轮完整对话，返回 (孩子说的话, LLM回复)
  speak_directly(text)       — 直接合成+播放（用于主动问候，不需要录音）

独立测试:
  python3 modules/voice_pipeline.py
"""

import os
import sys
import time
import subprocess
import tempfile
import requests
import urllib3

urllib3.disable_warnings()

# ============================================================
#  项目根目录（基于本文件位置自动定位）
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
#  API 密钥 — 使用前请替换为你的真实凭证
# ============================================================
ASR_KEY = "YOUR_FUNASR_API_KEY"
LLM_KEY = "YOUR_QWEN_API_KEY"
WS_ID   = "YOUR_FUNASR_WORKSPACE_ID"

# ============================================================
#  硬件配置 — 不可改动
# ============================================================
AUDIO_DEV = "plughw:2,0"
MIC_GAIN  = 29
REQ       = {"verify": False, "timeout": 30}

# ============================================================
#  API 端点
# ============================================================
ASR_SUBMIT_URL = f"https://{WS_ID}.cn-beijing.maas.aliyuncs.com/api/v1/services/audio/asr/transcription"
ASR_QUERY_URL  = f"https://{WS_ID}.cn-beijing.maas.aliyuncs.com/api/v1/tasks/"
LLM_URL        = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

# ============================================================
#  系统提示词
# ============================================================
BASE_PROMPT = (
    "你叫小伴，是面向3-12岁儿童的智能情绪陪伴机器人。"
    "说话温柔友善，用简单易懂的语言，回答简短不超过2句话，多鼓励孩子。"
    "不提及不安全内容。孩子表达负面情绪时先共情再引导。"
)

STORY_PROMPT = (
    "你叫小伴，是面向3-12岁儿童的智能情绪陪伴机器人。"
    "现在孩子想听故事。请用温柔的声音讲一个温馨、积极向上的小故事，"
    "长度控制在200字以内，语言简单适合3-12岁儿童理解。"
    "故事要有简单的开头、发展和结局。不提及不安全内容。"
)


# ============================================================
#  底层函数
# ============================================================

def setup_audio():
    """
    初始化声卡。main.py 启动时调用一次即可，可重复调用。
    操作：停 PipeWire → 设麦克风增益 → 开自动增益控制
    """
    os.system("systemctl --user stop pipewire.socket pipewire.service wireplumber 2>/dev/null")
    os.system(f"sudo amixer -c 2 cset numid=8 {MIC_GAIN} 2>/dev/null")
    os.system("sudo amixer -c 2 cset name='Auto Gain Control' on 2>/dev/null")


def record_audio(filepath, secs=8):
    """
    arecord 录音到指定路径。返回文件大小（字节）。
    调用者负责提供路径（可以是临时目录）。
    """
    subprocess.run(
        f"arecord -D {AUDIO_DEV} -d {secs} -f S16_LE -r 16000 -c 1 {filepath}",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        return os.path.getsize(filepath)
    except OSError:
        return 0


def asr(wav_path):
    """
    Fun-ASR 云端语音识别。
    步骤：获取 OSS 上传凭证 → 上传 wav → 提交异步任务 → 轮询结果。
    返回识别文字 (str)，失败返回 None。
    """
    try:
        # 1. 获取 OSS 上传凭证
        r = requests.get(
            "https://dashscope.aliyuncs.com/api/v1/uploads",
            params={"action": "getPolicy", "model": "fun-asr"},
            headers={"Authorization": f"Bearer {ASR_KEY}"},
            **REQ
        )
        r.raise_for_status()
        p = r.json()["data"]
        key = f"{p['upload_dir']}/{int(time.time() * 1000)}.wav"

        # 2. 上传音频到 OSS
        requests.post(
            p["upload_host"],
            data={
                "OSSAccessKeyId":      p["oss_access_key_id"],
                "Signature":           p["signature"],
                "policy":              p["policy"],
                "x-oss-object-acl":    p["x_oss_object_acl"],
                "x-oss-forbid-overwrite": p["x_oss_forbid_overwrite"],
                "key":                 key,
            },
            files={"file": open(wav_path, "rb")},
            **REQ
        )

        # 3. 提交异步识别任务
        r = requests.post(
            ASR_SUBMIT_URL,
            json={
                "model": "fun-asr",
                "input": {"file_urls": [f"oss://{key}"]},
                "parameters": {}
            },
            headers={
                "Authorization": f"Bearer {ASR_KEY}",
                "X-DashScope-Async": "enable",
                "X-DashScope-OssResourceResolve": "enable",
            },
            **REQ
        )
        r.raise_for_status()
        task_id = r.json()["output"]["task_id"]

        # 4. 轮询等待结果（最多 12 秒）
        for _ in range(12):
            time.sleep(1)
            r = requests.get(
                ASR_QUERY_URL + task_id,
                headers={"Authorization": f"Bearer {ASR_KEY}"},
                **REQ
            )
            d = r.json()
            st = d["output"]["task_status"]
            if st == "FAILED":
                return None
            if st == "SUCCEEDED":
                text = ""
                for rr in d["output"]["results"]:
                    # 兼容两种返回格式：transcription_url 或行内 results
                    if isinstance(rr, dict) and "transcription_url" in rr:
                        try:
                            trans = requests.get(rr["transcription_url"], **REQ).json()
                            if "transcripts" in trans:
                                for t in trans["transcripts"]:
                                    text += t.get("text", "")
                        except Exception:
                            pass
                    for sub in rr.get("results", []):
                        text += sub.get("text", "")
                return text if text else None
        return None
    except Exception as e:
        print(f"  [ASR] 错误: {e}")
        return None


def _build_system_prompt(emotion_context=None, is_story=False):
    """
    根据情绪上下文动态构建 system prompt。
    emotion_context: {"emotion_cn": "悲伤", "confidence": 0.72} 或 None
    """
    if is_story:
        prompt = STORY_PROMPT
    else:
        prompt = BASE_PROMPT

    if emotion_context:
        emo_cn = emotion_context.get("emotion_cn", "")
        conf   = emotion_context.get("confidence", 0)
        if emo_cn in ("悲伤", "恐惧", "愤怒"):
            prompt += (
                f" 你察觉到孩子看起来{emo_cn}"
                f"（置信度{conf*100:.0f}%），请先共情安抚，再温和引导。"
            )
        elif emo_cn == "开心":
            prompt += " 孩子现在情绪很好，保持愉快自然的互动。"

    return prompt


def llm_chat(text, emotion_context=None, is_story=False, max_tokens=120):
    """
    调用 Qwen-turbo 大模型。
    参数:
        text:             孩子说的话（ASR 结果）
        emotion_context:  情绪 dict，用于调整回复策略
        is_story:         是否切换到故事模式
        max_tokens:       最大回复长度（讲故事时建议 300）
    返回 LLM 回复文字，失败返回空字符串。
    """
    try:
        system_prompt = _build_system_prompt(emotion_context, is_story)
        headers = {
            "Authorization": f"Bearer {LLM_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "qwen-turbo",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            "temperature": 0.7,
            "max_tokens": max_tokens,
        }
        d = requests.post(LLM_URL, headers=headers, json=payload, **REQ).json()
        return d["choices"][0]["message"]["content"].strip() if "choices" in d else ""
    except Exception as e:
        print(f"  [LLM] 错误: {e}")
        return ""


def tts_gen(text, out_mp3, out_wav):
    """
    edge-tts 语音合成 (mp3) → ffmpeg 转 16kHz/16bit/mono wav。
    返回 True/False。
    """
    try:
        safe = text.replace('"', "'").replace("\n", " ")
        r = subprocess.run(
            [sys.executable, "-m", "edge_tts", "--voice", "zh-CN-XiaoyiNeural",
             "--text", safe, "--write-media", out_mp3],
            capture_output=True, text=True, timeout=30
        )
        if r.returncode != 0:
            print(f"  [TTS] edge-tts 失败: {r.stderr[:200]}")
            return False
        if not os.path.exists(out_mp3) or os.path.getsize(out_mp3) < 100:
            return False
        subprocess.run(
            f"ffmpeg -y -i {out_mp3} -ar 16000 -ac 1 -sample_fmt s16 {out_wav}",
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15
        )
        return os.path.exists(out_wav) and os.path.getsize(out_wav) > 0
    except Exception as e:
        print(f"  [TTS] 错误: {e}")
        return False


def play_audio(path):
    """aplay 直连 USB 声卡播放"""
    subprocess.run(
        f"aplay -D {AUDIO_DEV} {path}",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


# ============================================================
#  对外接口（供 main.py 调用）
# ============================================================

def run_conversation_turn(emotion_context=None, record_secs=8):
    """
    执行一轮完整对话：录音 → ASR → LLM（带情绪感知）→ TTS → 播放。

    参数:
        emotion_context:  视觉管线的情绪结果 dict，格式为 vision_pipeline
                          run_once() 返回的 faces[0]，例如:
                          {"emotion_cn":"悲伤","emotion_en":"Sadness","confidence":0.72,...}
                          传 None 表示本轮不使用情绪信息。
        record_secs:      录音时长（秒），默认 8。

    返回:
        (child_text, llm_reply)
        - child_text:  孩子说的话（ASR 结果），失败时为 None
        - llm_reply:   LLM 回复文字，失败时为 None 或默认安抚语

    注意:
        - 录音/ASR/LLM 任一阶段失败，会尝试用 TTS 播放一句安抚语
        - 自动检测孩子是否在要故事/听音乐，切换 LLM 模式
        - 临时文件使用 tempfile，函数返回后自动清理
    """
    child_text = None
    llm_reply  = None

    with tempfile.TemporaryDirectory() as tmpdir:
        rec_path  = os.path.join(tmpdir, "record.wav")
        tts_mp3   = os.path.join(tmpdir, "tts.mp3")
        tts_wav   = os.path.join(tmpdir, "tts.wav")

        # ---- 1. 录音 ----
        print(f"\n[录音] 请说话 ({record_secs}秒)...", flush=True)
        size = record_audio(rec_path, record_secs)
        print(f"[录音] {size} bytes", flush=True)
        if size < 2000:
            print("[录音] 警告：文件偏小，可能未录到有效声音", flush=True)

        # ---- 2. ASR ----
        print("[ASR] 识别中...", flush=True)
        child_text = asr(rec_path)
        if not child_text:
            print("[ASR] 未识别到文字", flush=True)
            # 播放一句安抚
            fallback = "我没听清，再说一遍好吗？"
            if tts_gen(fallback, tts_mp3, tts_wav):
                play_audio(tts_wav)
            return (None, fallback)

        print(f"孩子说: {child_text}", flush=True)

        # ---- 3. 检测意图，决定 LLM 模式 ----
        story_keywords = ["故事", "讲故事", "讲个故事", "讲故事吧", "讲一个故事"]
        music_keywords = ["音乐", "放音乐", "听音乐", "放歌", "唱歌"]
        is_story = any(kw in child_text for kw in story_keywords)
        is_music = any(kw in child_text for kw in music_keywords)

        if is_story:
            max_tokens = 300
            print("[意图] 检测到「讲故事」请求，切换到故事模式", flush=True)
        elif is_music:
            max_tokens = 120
            print("[意图] 检测到「音乐」请求", flush=True)
        else:
            max_tokens = 120

        # ---- 4. LLM ----
        print("[LLM] 思考中...", flush=True)
        llm_reply = llm_chat(
            child_text,
            emotion_context=emotion_context,
            is_story=is_story,
            max_tokens=max_tokens
        )
        if not llm_reply:
            print("[LLM] 未生成回复", flush=True)
            return (child_text, None)

        print(f"小伴: {llm_reply}", flush=True)

        # ---- 5. TTS ----
        print("[TTS] 合成语音...", flush=True)
        if not tts_gen(llm_reply, tts_mp3, tts_wav):
            print("[TTS] 合成失败", flush=True)
            return (child_text, llm_reply)

        # ---- 6. 播放 ----
        print("[播放]", flush=True)
        play_audio(tts_wav)

    return (child_text, llm_reply)


def speak_directly(text):
    """
    直接将一段文字合成语音并播放（不需要录音）。
    用于主动问候、系统提示等场景。

    参数:
        text: 要说的文本
    返回:
        True/False
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        mp3_path = os.path.join(tmpdir, "direct.mp3")
        wav_path = os.path.join(tmpdir, "direct.wav")
        if not tts_gen(text, mp3_path, wav_path):
            return False
        play_audio(wav_path)
        return True


# ============================================================
#  独立测试入口
#  python3 modules/voice_pipeline.py
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 40)
    print("  语音管线独立测试")
    print("  Enter = 开始一轮对话")
    print("  q     = 退出")
    print("=" * 40 + "\n")

    setup_audio()

    # ---- 可选：模拟一个情绪输入来测试情绪感知 ----
    # 改为 None 则不注入情绪
    test_emotion = None
    # test_emotion = {"emotion_cn": "悲伤", "confidence": 0.72}  # 取消注释以测试

    while True:
        try:
            cmd = input("[Enter] 开始 (q=退出): ").strip().lower()
            if cmd == "q":
                print("\n小伴: 再见啦，下次再一起玩~")
                break
        except (EOFError, KeyboardInterrupt):
            print("\n\n小伴: 再见啦~")
            break

        child, reply = run_conversation_turn(emotion_context=test_emotion)

        if child and reply:
            print(f"\n✓ 本轮完成: 「{child}」 → 「{reply}」")
        else:
            print(f"\n✗ 本轮未完整: ASR={child}, LLM={reply}")
