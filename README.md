# 🏀 Basketball Scoring & Analysis System

An end-to-end AI-powered basketball analytics system that automatically analyzes basketball games from broadcast video.

The system detects players, referees, the basketball, and court keypoints, then combines computer vision with custom game logic to automatically generate scores, statistics, court visualizations, and advanced analytics.

---

# Features

* 🎯 Player detection
* 🏀 Basketball detection
* 👨‍⚖️ Referee detection
* 📍 Court keypoint detection
* 👕 Team classification using jersey color
* 🗺️ Dynamic court mapping using homography
* 🔥 Team heatmaps
* 📌 Shot maps
* 🤝 Assist detection
* 🏀 Automatic scoring (2PT / 3PT)
* 💥 Block detection
* 🔄 Rebound detection
* ↔️ Pass detection
* ❌ Turnover detection
* 📈 Game momentum analysis
* ✨ Real-time OpenCV visualizations and animations

---

# System Pipeline

```
Input Video
      │
      ▼
Object Detection
(Players • Ball • Referees)
      │
      ▼
Court Keypoint Detection
      │
      ▼
Homography Estimation
      │
      ▼
Team Classification
      │
      ▼
Event Detection
      │
      ▼
Game Logic
      │
      ▼
Statistics + Visualizations
```

---

# Models & Datasets

The project combines several pretrained models from different sources.

## Ultralytics

Used for detecting:

* Players
* Basketball
* Referees

---

## Court Detection Model

A second pretrained model is used to detect basketball court keypoints.

These keypoints are later used to estimate the homography between the broadcast camera and a template basketball court.

---

## Hugging Face

A pretrained one-shot image classification model is used to classify each player into the correct team based on jersey color.

---

## Roboflow

Annotated datasets from Roboflow were used for training and evaluation.

---

# Event Detection

Several custom Python algorithms were developed to detect basketball events, including:

* Player possession
* Shot attempts
* Made baskets
* Blocks

These detected events become the foundation for the scoring engine and all game statistics.

---

# Homography & Court Mapping

Using OpenCV homography, the broadcast camera is transformed into a top-down basketball court.

This enables several analytics:

## Dynamic Court Map

Displays the live positions of every player during the game.

## Static Shot Map

Records the location of every successful shot.

## Assist Map

Displays where assists originated on the court.

## Heat Maps

Generates heatmaps showing court usage for each team.

---

# Automatic Scoring Logic

The project does not simply detect objects—it understands basketball events.

## Score Detection

When the basketball passes through the rim, the system:

1. Identifies the shooter.
2. Determines the shooter's team.
3. Determines whether the shot is worth **2** or **3** points based on the player's court position.
4. Updates the scoreboard automatically.

---

## Assist Detection

The possession history is tracked continuously.

When a basket is scored, the player who last passed the ball to the shooter is credited with an assist.

---

## Rebound Detection

If a shot misses and another player gains possession before the ball enters the basket:

* Same team → Offensive Rebound
* Opposite team → Defensive Rebound

---

## Passes & Turnovers

Ball possession is continuously monitored.

* Consecutive possessions between teammates are counted as passes.
* Possession changes between opposing teams are counted as turnovers.

---

# Momentum Analysis

Using player positioning together with detected game events, the system estimates which team currently has momentum throughout the game.

---

# Visualizations

The system generates multiple real-time OpenCV visualizations, including:

* Bounding boxes
* Court keypoints
* Dynamic court map
* Shot map
* Team heatmaps
* Event animations
* Live scoreboard
* Game statistics overlay
* Momentum visualization

---

# Future Improvements

The main missing feature is player identification.

A lightweight open-source OCR model capable of reading jersey numbers would enable:

* Individual player statistics
* Player tracking across the game
* Automatic box score generation
* Per-player heatmaps
* Advanced player analytics

---

# Tech Stack

* Python
* OpenCV
* NumPy
* Ultralytics YOLO
* Hugging Face Transformers
* Roboflow
* Computer Vision
* Homography
* Object Detection
* Image Classification

---

# Demo

<video src="out_5.mp4" controls width="100%"></video> 

---

# Repository

**GitHub:**
https://github.com/Ahmed-M0hamed/Basketball-Scoring-Analysis

---

# License

This project is released under the MIT License.
