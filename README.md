# SpacemiT K1 MUSE Pi Pro AI Emotional Companion Robot

## Overview

This project is an intelligent emotional companion robot for children, developed on the **SpacemiT K1 MUSE Pi Pro RISC-V Linux single-board computer**.

The system combines computer vision, emotion recognition, speech interaction, and large language model technologies to create a friendly AI companion that can perceive children's emotions and provide active emotional support.

The robot continuously monitors facial expressions through a camera, recognizes emotional states, and initiates conversations when negative emotions persist. It can also interact with children through voice conversation, storytelling, and local music playback.

---

## Features

### 1. Real-time Facial Emotion Recognition

* Camera-based face detection and emotion analysis
* YuNet ONNX face detection model
* HSEmotion EfficientNet-B2 emotion classification model
* Seven emotion categories:

  * Anger
  * Disgust
  * Fear
  * Happiness
  * Sadness
  * Surprise
  * Neutral

The vision pipeline runs locally on the SpacemiT K1 MUSE Pi Pro using RISC-V Linux and ONNX Runtime.

---

### 2. Emotion-aware Companion Logic

The system uses an emotion state machine with a sliding window mechanism.

Workflow:

```
Camera
  |
  v
Face Detection
  |
  v
Emotion Recognition
  |
  v
Emotion State Machine
  |
  +---- Normal emotion
  |        |
  |        v
  |     Continue monitoring
  |
  +---- Persistent negative emotion
           |
           v
       Start conversation
```

When negative emotions are detected continuously, the robot actively starts a conversation to comfort the child.

---

### 3. AI Voice Interaction

The voice pipeline provides a complete conversation loop:

```
Recording
    |
    v
Automatic Speech Recognition (ASR)
    |
    v
Large Language Model (LLM)
    |
    v
Text To Speech (TTS)
    |
    v
Audio Playback
```

Supported functions:

* Natural voice conversation
* Emotion-aware responses
* Storytelling mode
* Child-friendly dialogue

---

### 4. Local Music Playback

The robot can play local music files according to children's requests.

Supported formats:

* MP3
* WAV

---

## Hardware Platform

* **SpacemiT K1 MUSE Pi Pro**
* RISC-V Linux operating system
* USB Camera
* USB Microphone
* Speaker / USB Audio Device

---

## Software Architecture

```
SpacemiT K1 MUSE Pi Pro
          |
          |
      main.py
          |
  -------------------
  |        |        |
Vision   Emotion   Voice
Module   State     Module
         Machine
          |
          |
      Audio Player
```

Project structure:

```
emotion_companion/

├── main.py

├── modules/
│   ├── vision_pipeline.py
│   ├── emotion_state.py
│   ├── voice_pipeline.py
│   └── audio_player.py

├── models/
│   ├── face_detection_yunet_2023mar.onnx
│   └── enet_b2_7.onnx

├── audio/
│   └── music/

└── output/
```

---

## Running

Install required Python dependencies first.

Run the complete system:

```bash
python3 main.py
```

Individual modules can also be tested independently:

```bash
python3 modules/vision_pipeline.py

python3 modules/emotion_state.py

python3 modules/audio_player.py

python3 modules/voice_pipeline.py
```

---

## Technical Highlights

* RISC-V Linux embedded AI deployment
* ONNX Runtime inference without GPU acceleration
* Multi-threaded vision processing
* Emotion-driven interaction state machine
* Cloud ASR + LLM + TTS integration
* Modular Python architecture

---

# 中文说明

# 基于 SpacemiT K1 MUSE Pi Pro 的儿童智能情绪陪伴机器人

## 项目简介

本项目是一款基于 **SpacemiT K1 MUSE Pi Pro（RISC-V Linux 单板计算机）** 开发的儿童智能情绪陪伴设备。

系统融合计算机视觉、情绪识别、语音交互以及大语言模型技术，实现了一种能够感知儿童情绪并主动提供陪伴的智能机器人。

设备通过摄像头持续检测儿童面部表情，当检测到持续负面情绪时，会主动发起语音交流，通过 AI 对话、故事讲述以及音乐播放等方式提供情绪陪伴。

---

## 主要功能

### 1. 实时人脸情绪检测

系统通过摄像头采集图像，并完成：

```
摄像头采集
    |
    v
YuNet人脸检测
    |
    v
HSEmotion表情识别
    |
    v
输出情绪类别
```

支持七类基础情绪：

* 愤怒
* 厌恶
* 恐惧
* 开心
* 悲伤
* 惊讶
* 中性

所有视觉推理均部署在 SpacemiT K1 MUSE Pi Pro 本地运行。

---

### 2. 情绪感知陪伴逻辑

系统采用滑动窗口情绪状态机。

工作流程：

```
连续检测儿童情绪

        |
        v

负面情绪持续出现

        |
        v

主动发起语音陪伴

        |
        v

AI对话与安慰
```

相比单次表情判断，该方法可以降低误检测导致的错误触发。

---

### 3. AI语音交互

系统实现完整语音闭环：

```
语音录制
    |
    v
ASR语音识别
    |
    v
LLM智能回复
    |
    v
TTS语音合成
    |
    v
扬声器播放
```

支持：

* 儿童自然对话
* 情绪感知回复
* 故事模式
* 主动安慰

---

### 4. 本地音乐播放

设备支持根据儿童需求播放本地音乐。

支持格式：

* MP3
* WAV

---

## 硬件平台

* SpacemiT K1 MUSE Pi Pro
* RISC-V Linux系统
* USB摄像头
* USB麦克风
* 扬声器

---

## 软件结构

```
main.py

负责整体调度：

视觉线程
    |
情绪状态机
    |
语音交互
    |
音乐播放
```

主要模块：

* `vision_pipeline.py`

  * 人脸检测
  * 表情识别

* `emotion_state.py`

  * 情绪状态判断
  * 触发陪伴逻辑

* `voice_pipeline.py`

  * ASR
  * LLM
  * TTS

* `audio_player.py`

  * 本地音乐播放

---

## 运行方式

启动完整系统：

```bash
python3 main.py
```

也可以单独测试模块：

```bash
python3 modules/vision_pipeline.py

python3 modules/emotion_state.py

python3 modules/audio_player.py

python3 modules/voice_pipeline.py
```

---

## 技术特点

* 基于 RISC-V Linux 的边缘 AI 部署
* ONNX Runtime 本地推理
* 多线程视觉处理架构
* 情绪状态机决策
* ASR + LLM + TTS 智能语音交互
* 模块化 Python 工程设计
