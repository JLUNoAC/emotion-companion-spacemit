#!/usr/bin/env python3
"""
儿童智能情绪陪伴设备 — 主入口

架构:
    视觉线程（后台，每 6 秒一帧）
      └─ vision_pipeline.run_once() → 更新 latest_vision
    主线程（轮询，每秒检查）
      └─ emotion_state.update(latest_vision) → 决策 → 触发语音/音乐

逻辑:
    窗口内负面情绪 ≥ 3 次 → chat
      └─ 播放问候语 → 立刻开麦克风录音 → LLM 对话 → 回复

    对话结束后：
      └─ 检测孩子是否要音乐 → 本地 mp3 播放
      └─ 检测孩子是否要安静 → 静默 60 秒

模块依赖:
    modules/vision_pipeline.py
    modules/emotion_state.py
    modules/voice_pipeline.py
    modules/audio_player.py

运行:
    python3 main.py
    Ctrl+C 退出
"""

import os
import sys
import time
import threading

import cv2
import onnxruntime as ort

# ============ 项目根目录 ============
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from modules.vision_pipeline import run_once, FACE_MODEL, EMO_MODEL, CAM_ID
from modules.emotion_state  import EmotionStateMachine
from modules.voice_pipeline import setup_audio, run_conversation_turn, speak_directly
from modules.audio_player   import play_random_music

# ============ 配置 ============
VISION_INTERVAL = 6   # 视觉推理间隔（秒），RISC-V 推理 ~4.5s 留余量

# 对话后检查的关键词
MUSIC_KEYWORDS   = ["音乐", "放音乐", "听音乐", "放歌", "唱歌", "听歌", "播音乐", "来点音乐"]
SILENCE_KEYWORDS = ["安静", "别说话", "不要说话", "别吵", "静一静", "不要讲话"]

# ============ 共享状态 ============
vision_lock    = threading.Lock()
latest_vision  = None
vision_version = 0
running        = True


# ============================================================
#  视觉线程
# ============================================================

def vision_loop(face_sess, emo_sess):
    global latest_vision, vision_version, running

    cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("[视觉线程] 摄像头打开失败！", flush=True)
        return

    for _ in range(10):
        cap.read()

    print("[视觉线程] 已启动。", flush=True)

    while running:
        try:
            result = run_once(cap, face_sess, emo_sess)
            with vision_lock:
                latest_vision = result
                vision_version += 1
        except Exception as e:
            print(f"[视觉线程] 异常: {e}", flush=True)

        for _ in range(VISION_INTERVAL):
            if not running:
                break
            time.sleep(1)

    cap.release()
    print("[视觉线程] 已退出。", flush=True)


# ============================================================
#  主线程
# ============================================================

def main():
    global running

    print("\n" + "=" * 50)
    print("  小伴 — 儿童智能情绪陪伴设备")
    print("  Ctrl+C 退出")
    print("=" * 50 + "\n")

    # ── 1. 加载模型 ──
    print("[启动] 加载模型...", flush=True)
    face_sess = ort.InferenceSession(FACE_MODEL, providers=['CPUExecutionProvider'])
    emo_sess  = ort.InferenceSession(EMO_MODEL, providers=['CPUExecutionProvider'])
    print("[启动] 模型就绪。", flush=True)

    # ── 2. 声卡 ──
    print("[启动] 初始化声卡...", flush=True)
    setup_audio()
    print("[启动] 声卡就绪。", flush=True)

    # ── 3. 状态机 ──
    sm = EmotionStateMachine()

    # ── 4. 视觉线程 ──
    vt = threading.Thread(target=vision_loop, args=(face_sess, emo_sess), daemon=True)
    vt.start()

    # 等第一帧
    print("[启动] 等待首次视觉推理（约 6 秒）...", flush=True)
    waited = 0
    global vision_version
    last_processed_version = -1
    while vision_version == 0 and waited < 15:
        time.sleep(1)
        waited += 1
    if vision_version == 0:
        print("[启动] 警告：视觉管线无结果，以无视觉模式运行。", flush=True)
    else:
        print("[启动] 视觉管线就绪。\n", flush=True)

    # ── 5. 主循环 ──
    print("[运行] 开始监测...\n", flush=True)

    while running:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            print("\n[退出] 正在退出...", flush=True)
            break

        with vision_lock:
            cur_ver = vision_version
            vis = latest_vision

        if vis is None or cur_ver == last_processed_version:
            continue
        last_processed_version = cur_ver

        # ── 状态机决策 ──
        decision = sm.update(vis)

        # 终端输出
        if vis["face_count"] > 0:
            f = vis["faces"][0]
            print(f"[视觉] {f['emotion_cn']}({f['confidence']*100:.0f}%) | "
                  f"负面={sm.negative_count}/{sm.window.maxlen} | {decision}",
                  flush=True)
        else:
            print(f"[视觉] 无人脸 | "
                  f"负面={sm.negative_count}/{sm.window.maxlen} | {decision}",
                  flush=True)

        # ── 执行决策 ──
        if decision == "monitor":
            pass

        elif decision == "chat":
            print("[动作] chat — 问候 + 对话", flush=True)

            # 第一步：播放问候语
            speak_directly("我看到你好像有点不开心，想和我聊聊吗？")

            # 第二步：立刻进入对话（问候播完麦克风就开了）
            emo_ctx = None
            if vis["face_count"] > 0:
                emo_ctx = vis["faces"][0]
            child_text, llm_reply = run_conversation_turn(emotion_context=emo_ctx)

            # 第三步：对话结束后检查特殊请求
            if child_text:
                if any(kw in child_text for kw in MUSIC_KEYWORDS):
                    print("[动作] 检测到音乐请求 → 播放本地音乐", flush=True)
                    play_random_music()
                elif any(kw in child_text for kw in SILENCE_KEYWORDS):
                    print("[动作] 检测到静默请求 → 60 秒", flush=True)
                    time.sleep(60)

            print("[状态] 对话结束，继续监测。\n", flush=True)

    # ── 清理 ──
    running = False
    vt.join(timeout=10)
    print("[退出] 小伴已停止。", flush=True)


if __name__ == "__main__":
    main()
