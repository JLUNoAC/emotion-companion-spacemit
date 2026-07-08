#!/usr/bin/env python3
"""
音乐播放器

从 audio/music/ 随机选文件，用 aplay 播放。

依赖:
    audio/music/  — 放入 mp3 或 wav 音乐文件

硬件: USB 声卡 plughw:2,0（与 voice_pipeline 一致）

对外接口（供 main.py 调用）:
    play_random_music()   → bool   随机播放一首音乐

独立测试:
    python3 modules/audio_player.py
"""

import os
import random
import subprocess
import tempfile

# ============ 项目根目录（基于本文件位置自动定位）============
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR    = os.path.join(PROJECT_ROOT, "audio")
MUSIC_DIR    = os.path.join(AUDIO_DIR, "music")

# ============ 硬件 — 与 voice_pipeline 一致 ============
AUDIO_DEV = "plughw:2,0"


def _list_files(directory):
    """列出目录下所有音频文件（mp3 / wav），不存在目录则返回空列表"""
    if not os.path.isdir(directory):
        return []
    return sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith((".mp3", ".wav"))
    ])


def _play_file(filepath):
    """
    播放单个音频文件。
    - wav 直接用 aplay
    - mp3 先用 ffmpeg 转成临时 wav 再用 aplay
    返回 True / False。
    """
    if not os.path.exists(filepath):
        print(f"  [audio_player] 文件不存在: {filepath}")
        return False

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".wav":
        subprocess.run(
            f"aplay -D {AUDIO_DEV} {filepath}",
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return True

    # mp3 → 临时转 wav
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_wav = os.path.join(tmpdir, "temp.wav")
        ret = subprocess.run(
            f"ffmpeg -y -i {filepath} -ar 16000 -ac 1 -sample_fmt s16 {tmp_wav}",
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30
        )
        if ret.returncode != 0 or not os.path.exists(tmp_wav):
            print(f"  [audio_player] ffmpeg 转码失败: {filepath}")
            return False
        subprocess.run(
            f"aplay -D {AUDIO_DEV} {tmp_wav}",
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    return True


# ============================================================
#  对外接口
# ============================================================

def play_random_music():
    """
    从 audio/music/ 随机选一首播放。
    返回 True（已播放） / False（目录为空或不存在）。
    """
    files = _list_files(MUSIC_DIR)
    if not files:
        print("[audio_player] audio/music/ 目录为空，请放入 mp3 或 wav 文件")
        return False
    chosen = random.choice(files)
    print(f"[audio_player] 播放音乐: {chosen}")
    return _play_file(os.path.join(MUSIC_DIR, chosen))


# ============================================================
#  独立测试入口
#  python3 modules/audio_player.py
# ============================================================

if __name__ == "__main__":
    print("=" * 40)
    print("  音乐播放器 — 独立测试")
    print(f"  音乐目录: {MUSIC_DIR}")
    print("=" * 40)

    music_files = _list_files(MUSIC_DIR)
    print(f"\n音乐文件 ({len(music_files)}):")
    for f in music_files:
        print(f"  {f}")
    if not music_files:
        print("  (空)")

    while True:
        print("\n[m] 随机播放音乐  [q] 退出")
        try:
            cmd = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if cmd == "q":
            print("退出。")
            break
        elif cmd == "m":
            play_random_music()
