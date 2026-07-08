#!/usr/bin/env python3
"""
情绪状态机：滑动窗口 + 决策逻辑

输入: vision_pipeline.run_once() 返回的 dict
      {
          "face_count": 1,
          "faces": [{"emotion_cn": "悲伤", "emotion_en": "Sadness", "confidence": 0.72, ...}]
      }

输出: "monitor" | "greet" | "chat"

与 voice_pipeline 的关系:
    main.py 拿到决策后决定是否调用 voice_pipeline.run_conversation_turn()
    comfort / silence 由 main.py 根据 ASR 文字判，不放这里

独立测试（有限但可用）:
    python3 modules/emotion_state.py
    跑 4 个构造场景验证决策逻辑，不需要摄像头/模型/网络
"""

from collections import deque

# ============ 配置 ============
WINDOW_SIZE     = 10          # 滑动窗口长度（以 5 秒/帧计，约覆盖 50 秒）
CHAT_THRESHOLD  = 4           # 窗口内负面情绪 ≥ 此值时触发 chat

# 被判定为负面的情绪
NEGATIVE_EMOTIONS = {"悲伤", "恐惧", "愤怒", "厌恶"}


class EmotionStateMachine:
    """
    情绪状态机。

    三个状态:
        monitor  — 正面为主，继续观察
        greet    — 窗口首次出现负面时主动问候（一次）
        chat     — 窗口内持续负面 ≥ 阈值，触发语音对话

    关键行为:
        - greet 触发后不会重复触发，直到窗口清零（负面完全消失）后再出现新的负面
        - chat 的优先级高于 greet（持续 4 次负面直接进入对话，跳过问候）
        - main.py 在对话结束后应调用 reset_after_interaction()

    用法:
        sm = EmotionStateMachine()

        # 每轮 vision 更新后:
        decision = sm.update(vision_dict)

        # 对话或问候交互结束后:
        sm.reset_after_interaction()
    """

    def __init__(self):
        self.window = deque(maxlen=WINDOW_SIZE)
        self._greet_fired       = False    # 当前负面周期中是否已问候过
        self._window_was_clean  = True     # 自上次问候后窗口是否曾清零
        self._current_state     = "monitor"

    # ================================================================
    #  公开接口 — 供 main.py 调用
    # ================================================================

    def update(self, vision_dict):
        """
        喂入最新一次 vision 结果，返回决策。

        参数:
            vision_dict: vision_pipeline.run_once() 的返回值。
                         至少需要 {"face_count": N, "faces": [{"emotion_cn": "..."}]}

        返回:
            "monitor" | "greet" | "chat"
        """
        emo = self._extract_emotion(vision_dict)
        self.window.append(emo)

        neg_count = self._count_negatives()

        # 窗口清零 → 允许下一次负面触发 greet
        if neg_count == 0:
            self._window_was_clean = True

        decision = self._decide(neg_count)
        self._current_state = decision
        return decision

    def reset_after_interaction(self):
        """
        main.py 在一次 greet 问候或 chat 对话结束后调用。
        重置问候标记，允许未来再次触发 greet。

        注意：不会立即重新触发（需等窗口先清零再出现负面），
        避免对话刚结束又被同一次负面周期再次问候。
        """
        self._greet_fired = False

    # ================================================================
    #  只读属性（调试 / main.py 参考用）
    # ================================================================

    @property
    def current_state(self):
        """最近一次决策结果"""
        return self._current_state

    @property
    def window_snapshot(self):
        """当前窗口内容（按时间从旧到新）"""
        return list(self.window)

    @property
    def negative_count(self):
        """当前窗口内负面情绪数量"""
        return self._count_negatives()

    # ================================================================
    #  内部方法
    # ================================================================

    def _extract_emotion(self, vision_dict):
        """
        从 vision dict 中取出主导情绪的中文标签。
        无人脸 / 数据异常时返回 '无人脸'（不计入负面）。
        """
        if not vision_dict or vision_dict.get("face_count", 0) == 0:
            return "无人脸"
        try:
            return vision_dict["faces"][0]["emotion_cn"]
        except (IndexError, KeyError, TypeError):
            return "无人脸"

    def _count_negatives(self):
        """统计窗口内负面情绪出现次数"""
        return sum(1 for e in self.window if e in NEGATIVE_EMOTIONS)

    def _decide(self, neg_count):
        """
        按优先级返回决策:
            1. 持续负面 (≥ CHAT_THRESHOLD) → "chat"
            2. 首次负面 (窗口刚清零 + 未问候过) → "greet"
            3. 其余 → "monitor"
        """
        # 优先级 1: 持续负面，直接对话
        if neg_count >= CHAT_THRESHOLD:
            return "chat"

        # 优先级 2: 窗口刚清零后的第一个负面 → 问候一次
        if neg_count >= 1 and self._window_was_clean and not self._greet_fired:
            self._greet_fired = True
            self._window_was_clean = False
            return "greet"

        # 其余: 正面为主 / 已问候过 / 负面不够多 → 继续观察
        return "monitor"


# ============================================================
#  简易独立测试
#  不需要摄像头、模型、网络。纯逻辑验证。
#  python3 modules/emotion_state.py
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("  情绪状态机 — 逻辑测试")
    print("=" * 50)

    # ---- 测试 1: 全正面 → 一直 monitor ----
    print("\n[测试1] 连续 10 帧「开心」→ 应全部返回 monitor")
    sm = EmotionStateMachine()
    for i in range(10):
        fake = {"face_count": 1, "faces": [{"emotion_cn": "开心"}]}
        d = sm.update(fake)
        print(f"  帧{i+1}: {d}  负面={sm.negative_count}")
    assert sm.negative_count == 0 and sm.current_state == "monitor"
    print("  ✓ 通过")

    # ---- 测试 2: 首次负面 greet → 后续 monitor → 负面≥4 触发 chat ----
    print("\n[测试2] 连续悲伤: 帧1→greet, 帧2-3→monitor, 帧4→chat")
    sm = EmotionStateMachine()
    decisions = []
    for i in range(6):
        fake = {"face_count": 1, "faces": [{"emotion_cn": "悲伤"}]}
        d = sm.update(fake)
        decisions.append(d)
        print(f"  帧{i+1}: {d}  负面={sm.negative_count}")

    assert decisions[0] == "greet",   f"帧1 应为 greet，实际 {decisions[0]}"
    assert decisions[1] == "monitor", f"帧2 应为 monitor（已问候过），实际 {decisions[1]}"
    assert decisions[2] == "monitor", f"帧3 应为 monitor（负面<4），实际 {decisions[2]}"
    assert decisions[3] == "chat",    f"帧4 应为 chat（负面≥4），实际 {decisions[3]}"
    print("  ✓ 通过")

    # ---- 测试 3: reset 后窗口先清零，再遇负面 → 重新 greet ----
    print("\n[测试3] reset + 窗口清零 → 新负面应再次触发 greet")
    sm = EmotionStateMachine()

    # 制造一波负面
    for _ in range(6):
        sm.update({"face_count": 1, "faces": [{"emotion_cn": "悲伤"}]})
    sm.reset_after_interaction()
    # 清零窗口（填满开心）
    for _ in range(10):
        sm.update({"face_count": 1, "faces": [{"emotion_cn": "开心"}]})

    print(f"  reset+清零后: 负面={sm.negative_count}, clean={sm._window_was_clean}")

    d = sm.update({"face_count": 1, "faces": [{"emotion_cn": "恐惧"}]})
    print(f"  新负面「恐惧」→ {d}")
    assert d == "greet", f"应为 greet，实际 {d}"
    print("  ✓ 通过")

    # ---- 测试 4: 无人脸 → 不影响判断 ----
    print("\n[测试4] 无人脸 + 悲伤混合")
    sm = EmotionStateMachine()
    decisions = []
    for emo in ["无人脸", "悲伤", "无人脸", "悲伤", "悲伤", "悲伤", "悲伤", "无人脸"]:
        if emo == "无人脸":
            fake = {"face_count": 0, "faces": []}
        else:
            fake = {"face_count": 1, "faces": [{"emotion_cn": emo}]}
        d = sm.update(fake)
        decisions.append((emo, d))
        print(f"  {emo} → {d}  负面={sm.negative_count}")

    # "无人脸"不计入负面，所以虽然跑了 8 帧，只有 4 帧 悲伤是负面
    # 第 1 帧 悲伤应 greet，第 4 帧 悲伤（累计 4）触发 chat
    greet_idx = next(i for i, (e, _) in enumerate(decisions) if e == "悲伤")
    chat_seen = any(d == "chat" for _, d in decisions)
    assert decisions[greet_idx][1] == "greet", f"首次悲伤应在第{greet_idx+1}帧触发 greet"
    assert chat_seen, "累计 4 次悲伤应触发 chat"
    print("  ✓ 通过")

    print("\n" + "=" * 50)
    print("  全部测试通过 ✓")
    print("=" * 50)
