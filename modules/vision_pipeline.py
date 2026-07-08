#!/usr/bin/env python3
"""
视觉管线：摄像头拍照 → YuNet人脸检测 → HSEmotion表情识别
运行方式: python3 modules/vision_pipeline.py
输出: output/result.txt + output/screenshots/ 标注全景图 + output/crops/ 人脸裁剪图

也可被 import：
    from modules.vision_pipeline import run_once
    result = run_once(cap, face_sess, emo_sess)   # 返回 dict，同时写文件
"""
import cv2
import numpy as np
import onnxruntime as ort
import os
from datetime import datetime

# ============ 项目根目录（基于本文件位置自动定位）============
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# ============ 模型路径 ============
FACE_MODEL = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
EMO_MODEL  = os.path.join(MODEL_DIR, "enet_b2_7.onnx")

# ============ 输出路径 ============
OUTPUT_DIR      = os.path.join(PROJECT_ROOT, "output")
SCREENSHOT_DIR  = os.path.join(OUTPUT_DIR, "screenshots")
CROP_DIR        = os.path.join(OUTPUT_DIR, "crops")
RESULT_FILE     = os.path.join(OUTPUT_DIR, "result.txt")

# ============ 摄像头 ============
CAM_ID = 20

# ============ 情绪标签 ============
EMOTIONS_EN = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"]
EMOTIONS_CN = ["愤怒",   "厌恶",    "恐惧", "开心",      "悲伤",    "惊讶",    "中性"]

# ============ HSEmotion 预处理参数 ============
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 260


# ============ NMS ============
def nms(boxes, scores, thresh):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        ovr = w * h / (area[i] + area[order[1:]] - w * h)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep


# ============ YuNet 人脸检测 ============
def detect_faces(img, sess):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/127.5, (640, 640), (127.5, 127.5, 127.5), swapRB=True)
    outs = sess.run(None, {sess.get_inputs()[0].name: blob})

    all_boxes, all_scores = [], []
    strides = [8, 16, 32]
    for idx, stride in enumerate(strides):
        cls_out = outs[idx]
        box_out = outs[idx + 6]
        if cls_out is None or cls_out.shape[1] == 0:
            continue
        scores = cls_out[0, :, 0]
        bboxes = box_out[0]
        valid = scores > 0.6
        if not valid.any():
            continue
        s2, b2 = scores[valid], bboxes[valid]
        gw = 640 // stride
        gh = 640 // stride
        xv, yv = np.meshgrid(np.arange(gw), np.arange(gh))
        cx = xv.flatten()
        cy = yv.flatten()
        vi = np.where(valid)[0]
        cx = cx[vi]
        cy = cy[vi]
        xc = ((cx + b2[:, 0]) * stride) / 640 * w
        yc = ((cy + b2[:, 1]) * stride) / 640 * h
        bw = np.exp(b2[:, 2]) * stride / 640 * w
        bh = np.exp(b2[:, 3]) * stride / 640 * h
        boxes = np.stack([xc - bw/2, yc - bh/2, xc + bw/2, yc + bh/2], axis=1)
        all_boxes.append(boxes)
        all_scores.append(s2)

    if not all_boxes:
        return []
    all_boxes = np.vstack(all_boxes)
    all_scores = np.hstack(all_scores)
    keep = nms(all_boxes, all_scores, 0.3)
    return [all_boxes[i].tolist() for i in keep[:1]]


# ============ 正方形带边距裁剪 ============
def crop_face(img, bbox):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    m = int(max(bw, bh) * 0.35)
    x1, y1 = max(0, x1 - m), max(0, y1 - m)
    x2, y2 = min(w, x2 + m), min(h, y2 + m)
    side = max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    nx1 = max(0, cx - side // 2)
    ny1 = max(0, cy - side // 2)
    return img[ny1:ny1 + side, nx1:nx1 + side]


# ============ HSEmotion 表情识别 ============
def recognize_emotion(face_bgr, sess):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face_rgb, (INPUT_SIZE, INPUT_SIZE))
    face = face.astype(np.float32) / 255.0
    face = (face - MEAN) / STD
    inp = np.transpose(face, (2, 0, 1))[np.newaxis, ...]
    logits = sess.run(None, {sess.get_inputs()[0].name: inp})[0][0]
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    ranked = sorted(
        [(EMOTIONS_EN[i], EMOTIONS_CN[i], float(probs[i])) for i in range(7)],
        key=lambda x: x[2], reverse=True
    )
    return ranked, logits


# ============================================================
#  run_once() — 可被外部 import 调用的核心函数
#  返回结构化 dict，同时写 result.txt / 截图 / 裁剪图
# ============================================================

def run_once(cap, face_sess, emo_sess):
    """
    执行一次完整的拍照→检测→识别流程。

    参数:
        cap:        cv2.VideoCapture 对象（已打开、已预热）
        face_sess:  YuNet ONNX 推理会话
        emo_sess:   HSEmotion ONNX 推理会话

    返回:
        {
            "timestamp":    "2026-07-08 17:54:37",
            "run_id":       "20260708_175433_212723",
            "face_count":   1,
            "faces": [
                {
                    "bbox":        [521, 74, 543, 96],
                    "emotion_en":  "Sadness",
                    "emotion_cn":  "悲伤",
                    "confidence":  0.30,
                    "top7":        [("悲伤",0.30), ("惊讶",0.207), ...],
                    "logits":      [-1.25, -1.0, 0.09, ...]
                }
            ],
            "screenshot_path": "/home/.../screenshot.jpg",
            "crop_paths":      ["/home/.../crop_1.jpg"]
        }

    副作用:
        - 写入 output/result.txt
        - 保存 output/screenshots/ 全景标注图
        - 保存 output/crops/ 人脸裁剪图
    """

    # 确保输出目录存在
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    os.makedirs(CROP_DIR, exist_ok=True)

    # 拍照
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("摄像头拍照失败")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 检测人脸
    faces = detect_faces(frame, face_sess)

    # ---- 构建返回值 ----
    result = {
        "timestamp": timestamp,
        "run_id": run_id,
        "face_count": len(faces),
        "faces": [],
        "screenshot_path": "",
        "crop_paths": [],
    }

    # ---- 写入 result.txt 的内容（与原来完全一致）----
    result_lines = []
    result_lines.append(f"运行时间: {timestamp}")
    result_lines.append(f"运行编号: {run_id}")
    result_lines.append("")

    if faces:
        result_lines.append(f"检测到 {len(faces)} 张人脸")
        result_lines.append("")

        for i, bbox in enumerate(faces):
            x1, y1, x2, y2 = [int(v) for v in bbox]

            # 正方形裁剪
            crop = crop_face(frame, bbox)
            if crop.size == 0:
                continue

            # 表情识别
            ranked, raw_logits = recognize_emotion(crop, emo_sess)
            top_en, top_cn, top_prob = ranked[0]

            # ---- 结构化数据 ----
            face_data = {
                "bbox":       [x1, y1, x2, y2],
                "emotion_en": top_en,
                "emotion_cn": top_cn,
                "confidence": round(top_prob, 4),
                "top7":       [(cn, round(p, 4)) for en, cn, p in ranked],
                "logits":     [round(v, 4) for v in raw_logits.tolist()],
            }
            result["faces"].append(face_data)

            # ---- result.txt（中文标签）----
            result_lines.append(f"--- 人脸 {i+1} ---")
            result_lines.append(f"检测框: x1={x1} y1={y1} x2={x2} y2={y2}")
            result_lines.append(f"主情绪: {top_cn} ({top_en}) {top_prob*100:.1f}%")
            result_lines.append(f"TOP7: " + "  ".join(
                f"{cn}={p*100:.1f}%" for en, cn, p in ranked
            ))
            result_lines.append(f"原始logits: {[round(v,2) for v in raw_logits.tolist()]}")
            result_lines.append("")

            # 画绿框（人脸检测框）
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 图片标注（英文，OpenCV 中文会乱码）
            label = f"{top_en} {top_prob*100:.0f}%"
            cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 保存人脸裁剪图
            crop_path = os.path.join(CROP_DIR, f"face_crop_{run_id}_{i+1}.jpg")
            cv2.imwrite(crop_path, crop)
            result["crop_paths"].append(crop_path)

    else:
        result_lines.append("未检测到人脸。")
        result_lines.append("")

    # 保存标注全景图
    screenshot_path = os.path.join(SCREENSHOT_DIR, f"emotion_detect_{run_id}.jpg")
    cv2.imwrite(screenshot_path, frame)
    result["screenshot_path"] = screenshot_path

    # 写入 result.txt
    result_lines.append(f"全景截图: {screenshot_path}")
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(result_lines))

    return result


# ============================================================
#  独立运行入口 — 行为与原来完全一致
#  python3 modules/vision_pipeline.py  依然能跑，输出不变
# ============================================================

if __name__ == "__main__":
    # 加载模型
    print("Loading YuNet...", flush=True)
    face_sess = ort.InferenceSession(FACE_MODEL, providers=['CPUExecutionProvider'])
    print("Loading HSEmotion...", flush=True)
    emo_sess = ort.InferenceSession(EMO_MODEL, providers=['CPUExecutionProvider'])

    # 打开摄像头
    cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Camera failed!")
        exit(1)

    # 预热10帧
    for _ in range(10):
        ret, frame = cap.read()

    if not ret:
        print("拍照失败")
        cap.release()
        exit(1)

    # 一行调用
    result = run_once(cap, face_sess, emo_sess)

    # 释放摄像头
    cap.release()

    # 终端简短摘要
    if result["face_count"] > 0:
        f0 = result["faces"][0]
        print(f"完成！检测到 {result['face_count']} 张人脸，"
              f"主情绪: {f0['emotion_cn']} ({f0['confidence']*100:.0f}%)，"
              f"结果已写入 {RESULT_FILE}")
    else:
        print(f"完成！未检测到人脸，结果已写入 {RESULT_FILE}")
