# Futuristic AI Real-Time Object Detection and Tracking Web App

## Project Overview

This project is a premium AI surveillance-style Streamlit web app that performs live object detection and tracking from a browser webcam feed. It uses YOLOv8 for detection, a SORT-style tracker for persistent IDs, and WebRTC so each viewer can open the app and use their own camera directly in the browser.

This is the key difference from `cv2.VideoCapture(0)`: that approach only opens the webcam on the machine running the Python server. This app is built so you can share a public link and your friends can allow browser camera access on their own devices.

## Features

- live webcam detection in the browser
- shareable architecture using WebRTC
- YOLOv8 nano real-time object detection
- SORT-style tracking with persistent IDs
- fixed unique colors for each object class
- futuristic glassmorphism dashboard
- glowing CCTV-style bounding boxes and overlays
- live FPS counter
- detected object count
- active tracking count
- detection and tracking toggle controls
- screenshot capture
- browser camera permission support
- error handling for missing model and processing failures

## Technologies Used

- Python
- Streamlit
- streamlit-webrtc
- OpenCV
- Ultralytics YOLOv8
- NumPy
- SciPy
- FilterPy
- LAP
- PyAV

## Installation

1. Create and activate a virtual environment if you want an isolated setup.
2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

If you still see an import error mentioning `tornado`, install it into the same Python environment used by Streamlit:

```powershell
python -m pip install tornado
```

3. Make sure `yolov8n.pt` is present in the project folder.

## How to Run Locally

```powershell
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

Press the `Start` control shown by the webcam component and allow browser camera permission.

## How to Share With Friends

`localhost` only works on your own computer. To let other people access the app, you need a public HTTPS URL.

Good options:

- Streamlit Community Cloud
- `ngrok http 8501`
- Cloudflare Tunnel
- any HTTPS reverse proxy or hosted VM

Why HTTPS matters:

- browser webcam access generally requires a secure context
- WebRTC works best with a public secure URL

When your friends open the shared public link:

- their browser asks for webcam permission
- their own live camera feed is used
- detection and tracking run on that browser stream

## Controls

- `Start` inside the webcam component: start browser camera stream
- `Stop` inside the webcam component: stop browser camera stream
- `Toggle Detection`: enable or disable YOLO detection
- `Toggle Tracking`: enable or disable tracking
- `Save Screenshot`: save the latest annotated frame

## Default Focused Classes

- person
- chair
- laptop
- cell phone
- bottle
- keyboard
- monitor
- mouse
- book

## Project Files

- `app.py`: Streamlit WebRTC webcam app
- `main.py`: desktop OpenCV webcam version
- `requirements.txt`: dependencies
- `README.md`: project guide
- `yolov8n.pt`: YOLOv8 nano model weights
