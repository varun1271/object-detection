from __future__ import annotations

import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

try:
    import av  # type: ignore[import-not-found]
except ImportError:
    av = None

WEBRTC_IMPORT_ERROR: Optional[str] = None
try:
    from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
except ImportError as error:
    VideoProcessorBase = object  # type: ignore[assignment]
    WebRtcMode = None  # type: ignore[assignment]
    webrtc_streamer = None  # type: ignore[assignment]
    WEBRTC_IMPORT_ERROR = str(error)


Box = tuple[int, int, int, int]
MODEL_PATH = "yolov8n.pt"
DEFAULT_CLASSES = (
    "person",
    "chair",
    "laptop",
    "cell phone",
    "bottle",
    "keyboard",
    "tv",
    "mouse",
    "book",
)
CLASS_LABELS = {
    "person": "Person",
    "chair": "Chair",
    "laptop": "Laptop",
    "cell phone": "Mobile Phone",
    "bottle": "Bottle",
    "keyboard": "Keyboard",
    "tv": "Monitor",
    "mouse": "Mouse",
    "book": "Book",
}
CLASS_COLORS = {
    "person": (70, 220, 120),
    "chair": (255, 130, 70),
    "laptop": (185, 95, 255),
    "cell phone": (255, 235, 90),
    "bottle": (70, 190, 255),
    "keyboard": (255, 115, 210),
    "tv": (80, 90, 255),
    "mouse": (255, 205, 70),
    "book": (70, 220, 200),
}
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


@dataclass
class Detection:
    box: Box
    confidence: float
    class_id: int
    class_name: str


@dataclass
class Track:
    track_id: int
    box: Box
    class_id: int
    class_name: str
    confidence: float
    hits: int = 1
    missed_frames: int = 0
    velocity: tuple[int, int] = (0, 0)
    history: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=30))

    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def predict_box(self) -> Box:
        dx, dy = self.velocity
        x1, y1, x2, y2 = self.box
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


class SortLikeTracker:
    def __init__(self, iou_threshold: float = 0.25, max_age: int = 18, min_hits: int = 2) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.next_track_id = 1
        self.tracks: list[Track] = []

    def reset(self) -> None:
        self.next_track_id = 1
        self.tracks.clear()

    def update(self, detections: Sequence[Detection]) -> list[Track]:
        predicted_boxes: list[Box] = []
        for track in self.tracks:
            track.missed_frames += 1
            predicted_boxes.append(track.predict_box())

        candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self.tracks):
            for detection_index, detection in enumerate(detections):
                if track.class_id != detection.class_id:
                    continue
                iou_score = compute_iou(predicted_boxes[track_index], detection.box)
                if iou_score < self.iou_threshold:
                    continue
                previous_center = track.center()
                current_center = box_center(detection.box)
                distance = np.hypot(
                    current_center[0] - previous_center[0],
                    current_center[1] - previous_center[1],
                )
                candidates.append((iou_score - distance * 0.0005, track_index, detection_index))

        candidates.sort(reverse=True, key=lambda item: item[0])

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for _, track_index, detection_index in candidates:
            if track_index in matched_tracks or detection_index in matched_detections:
                continue

            detection = detections[detection_index]
            track = self.tracks[track_index]
            previous_center = track.center()
            current_center = box_center(detection.box)

            track.velocity = (
                current_center[0] - previous_center[0],
                current_center[1] - previous_center[1],
            )
            track.box = detection.box
            track.class_id = detection.class_id
            track.class_name = detection.class_name
            track.confidence = detection.confidence
            track.hits += 1
            track.missed_frames = 0
            track.history.append(current_center)

            matched_tracks.add(track_index)
            matched_detections.add(detection_index)

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            track = Track(
                track_id=self.next_track_id,
                box=detection.box,
                class_id=detection.class_id,
                class_name=detection.class_name,
                confidence=detection.confidence,
            )
            track.history.append(track.center())
            self.tracks.append(track)
            self.next_track_id += 1

        self.tracks = [track for track in self.tracks if track.missed_frames <= self.max_age]
        return [
            track
            for track in self.tracks
            if track.missed_frames == 0 and track.hits >= self.min_hits
        ]


def normalize_class_name(name: str) -> str:
    normalized = name.strip().lower()
    aliases = {
        "mobile": "cell phone",
        "mobile phone": "cell phone",
        "phone": "cell phone",
        "monitor": "tv",
        "screen": "tv",
    }
    return aliases.get(normalized, normalized)


def display_class_name(name: str) -> str:
    return CLASS_LABELS.get(normalize_class_name(name), name.title())


def class_color(name: str) -> tuple[int, int, int]:
    return CLASS_COLORS.get(normalize_class_name(name), (180, 180, 180))


def box_center(box: Box) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def compute_iou(first_box: Box, second_box: Box) -> float:
    x1 = max(first_box[0], second_box[0])
    y1 = max(first_box[1], second_box[1])
    x2 = min(first_box[2], second_box[2])
    y2 = min(first_box[3], second_box[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first_box[2] - first_box[0]) * max(0, first_box[3] - first_box[1])
    second_area = max(0, second_box[2] - second_box[0]) * max(0, second_box[3] - second_box[1])
    union = first_area + second_area - intersection
    return 0.0 if union <= 0 else intersection / union


@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> YOLO:
    # yolov8n.pt is excluded from git (too large).
    # ultralytics will auto-download it on first run if it is missing.
    return YOLO(model_path)


def detect_objects(
    model: YOLO,
    frame,
    confidence: float,
    iou: float,
    image_size: int,
    selected_classes: Sequence[str],
) -> list[Detection]:
    results = model.predict(
        source=frame,
        conf=confidence,
        iou=iou,
        imgsz=image_size,
        agnostic_nms=False,
        verbose=False,
    )

    allowed = {normalize_class_name(name) for name in selected_classes}
    detections: list[Detection] = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = normalize_class_name(str(model.names[class_id]))
            if allowed and class_name not in allowed:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append(
                Detection(
                    box=(x1, y1, x2, y2),
                    confidence=float(box.conf[0]),
                    class_id=class_id,
                    class_name=class_name,
                )
            )
    return detections


def alpha_panel(
    frame,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    fill: tuple[int, int, int],
    alpha: float,
    border: Optional[tuple[int, int, int]] = None,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, fill, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    if border is not None:
        cv2.rectangle(frame, top_left, bottom_right, border, 1, cv2.LINE_AA)


def glow_box(frame, box: Box, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = box
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), color, 6)
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y1), (x1 + 20, y1 + 20), color, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (x2 - 20, y2 - 20), (x2, y2), color, 2, cv2.LINE_AA)


def draw_label_chip(frame, box: Box, label: str, color: tuple[int, int, int]) -> None:
    x1, y1, _, _ = box
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.54
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
    top = max(8, y1 - text_height - baseline - 16)
    bottom = top + text_height + baseline + 12
    alpha_panel(frame, (x1, top), (x1 + text_width + 16, bottom), color, 0.90)
    cv2.putText(
        frame,
        label,
        (x1 + 7, bottom - baseline - 4),
        font,
        scale,
        (18, 20, 25),
        thickness,
        cv2.LINE_AA,
    )


def draw_detections(frame, detections: Sequence[Detection]) -> Counter:
    counts: Counter = Counter()
    for detection in detections:
        counts[detection.class_name] += 1
        color = class_color(detection.class_name)
        glow_box(frame, detection.box, color)
        label = f"{display_class_name(detection.class_name)}  {detection.confidence:.2f}"
        draw_label_chip(frame, detection.box, label, color)
    return counts


def draw_tracks(frame, tracks: Sequence[Track]) -> Counter:
    counts: Counter = Counter()
    for track in tracks:
        counts[track.class_name] += 1
        color = class_color(track.class_name)
        glow_box(frame, track.box, color)
        label = (
            f"{display_class_name(track.class_name)} "
            f"ID:{track.track_id}  {track.confidence:.2f}"
        )
        draw_label_chip(frame, track.box, label, color)

        points = list(track.history)
        for index in range(1, len(points)):
            fade = max(0.2, index / len(points))
            trail_color = tuple(int(channel * fade) for channel in color)
            cv2.line(frame, points[index - 1], points[index], trail_color, 2, cv2.LINE_AA)
        if points:
            cv2.circle(frame, points[-1], 3, color, -1, cv2.LINE_AA)
    return counts


def draw_status_hud(
    frame,
    fps: float,
    total_objects: int,
    active_tracks: int,
    detection_enabled: bool,
    tracking_enabled: bool,
    webcam_status: str,
) -> None:
    height, width = frame.shape[:2]
    alpha_panel(frame, (14, 14), (348, 150), (9, 15, 28), 0.72, (58, 92, 135))
    cv2.putText(
        frame,
        "AI SURVEILLANCE GRID",
        (28, 40),
        cv2.FONT_HERSHEY_DUPLEX,
        0.72,
        (240, 247, 252),
        1,
        cv2.LINE_AA,
    )
    lines = [
        f"Objects: {total_objects}",
        f"Tracks: {active_tracks}",
        f"Detection: {'ON' if detection_enabled else 'OFF'}",
        f"Tracking: {'ON' if tracking_enabled else 'OFF'}",
    ]
    y = 66
    for line in lines:
        cv2.putText(
            frame,
            line,
            (28, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (220, 228, 234),
            1,
            cv2.LINE_AA,
        )
        y += 24

    fps_label = f"FPS {fps:5.1f}"
    label_width = cv2.getTextSize(fps_label, cv2.FONT_HERSHEY_DUPLEX, 0.84, 1)[0][0]
    alpha_panel(
        frame,
        (width - label_width - 44, 14),
        (width - 14, 52),
        (8, 28, 34),
        0.78,
        (70, 220, 218),
    )
    cv2.putText(
        frame,
        fps_label,
        (width - label_width - 28, 40),
        cv2.FONT_HERSHEY_DUPLEX,
        0.84,
        (92, 255, 230),
        1,
        cv2.LINE_AA,
    )

    status_color = (70, 215, 130) if webcam_status.startswith("Live browser camera connected") else (70, 140, 255)
    alpha_panel(frame, (14, height - 72), (min(width - 14, 520), height - 18), (10, 16, 28), 0.72, status_color)
    cv2.putText(
        frame,
        webcam_status[:72],
        (28, height - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 240, 244),
        1,
        cv2.LINE_AA,
    )


def add_scanlines(frame) -> None:
    overlay = frame.copy()
    for y in range(0, frame.shape[0], 6):
        cv2.line(overlay, (0, y), (frame.shape[1], y), (0, 0, 0), 1)
    cv2.addWeighted(overlay, 0.06, frame, 0.94, 0, frame)


def save_screenshot(frame_rgb) -> Optional[Path]:
    if frame_rgb is None:
        return None
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = Path(f"screenshot_{timestamp}.jpg")
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), frame_bgr)
    return path


def process_browser_snapshot(
    image_bytes: bytes,
    selected_classes: Sequence[str],
    confidence: float,
    iou: float,
    image_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    model = load_model(MODEL_PATH)
    frame_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Unable to decode browser camera image.")

    started = time.perf_counter()
    detections: list[Detection] = []
    tracks: list[Track] = []
    if st.session_state.detection_enabled:
        detections = detect_objects(
            model,
            frame,
            confidence,
            iou,
            image_size,
            selected_classes,
        )
    if st.session_state.detection_enabled and st.session_state.tracking_enabled:
        tracks = st.session_state.snapshot_tracker.update(detections)
    elif not st.session_state.tracking_enabled:
        st.session_state.snapshot_tracker.reset()

    output = frame.copy()
    if (
        st.session_state.detection_enabled
        and st.session_state.tracking_enabled
        and tracks
    ):
        counts = draw_tracks(output, tracks)
    elif st.session_state.detection_enabled:
        counts = draw_detections(output, detections)
    else:
        counts = Counter()

    fps = 1.0 / max(time.perf_counter() - started, 1e-6)
    draw_status_hud(
        output,
        fps=fps,
        total_objects=sum(counts.values()),
        active_tracks=len(tracks),
        detection_enabled=st.session_state.detection_enabled,
        tracking_enabled=st.session_state.tracking_enabled,
        webcam_status="Browser snapshot camera connected",
    )
    add_scanlines(output)
    frame_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    stats = {
        "fps": fps,
        "total_objects": sum(counts.values()),
        "active_tracks": len(tracks),
        "last_frame_rgb": frame_rgb,
        "last_error": "",
    }
    return frame_rgb, stats


class StatsStore:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self.lock:
            self.fps = 0.0
            self.total_objects = 0
            self.active_tracks = 0
            self.last_frame_rgb: Optional[np.ndarray] = None
            self.last_error = ""

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return {
                "fps": self.fps,
                "total_objects": self.total_objects,
                "active_tracks": self.active_tracks,
                "last_frame_rgb": None if self.last_frame_rgb is None else self.last_frame_rgb.copy(),
                "last_error": self.last_error,
            }


VideoFrameLike = Any


class DetectionVideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.model = load_model(MODEL_PATH)
        self.tracker = SortLikeTracker()
        self.selected_classes = list(DEFAULT_CLASSES)
        self.confidence = 0.25
        self.iou = 0.45
        self.image_size = 640
        self.detection_enabled = True
        self.tracking_enabled = True
        self.stats = StatsStore()
        self.fps_history: deque[float] = deque(maxlen=30)
        self.status_message = "Awaiting browser camera permission."

    def recv(self, frame: VideoFrameLike) -> VideoFrameLike:
        started = time.perf_counter()
        image = frame.to_ndarray(format="bgr24")

        try:
            detections: list[Detection] = []
            tracks: list[Track] = []
            if self.detection_enabled:
                detections = detect_objects(
                    self.model,
                    image,
                    self.confidence,
                    self.iou,
                    self.image_size,
                    self.selected_classes,
                )
            if self.detection_enabled and self.tracking_enabled:
                tracks = self.tracker.update(detections)
            elif not self.tracking_enabled:
                self.tracker.reset()

            output = image.copy()
            if self.detection_enabled and self.tracking_enabled:
                counts = draw_tracks(output, tracks)
            elif self.detection_enabled:
                counts = draw_detections(output, detections)
            else:
                counts = Counter()

            frame_fps = 1.0 / max(time.perf_counter() - started, 1e-6)
            self.fps_history.append(frame_fps)
            average_fps = sum(self.fps_history) / len(self.fps_history)

            draw_status_hud(
                output,
                fps=average_fps,
                total_objects=sum(counts.values()),
                active_tracks=len(tracks),
                detection_enabled=self.detection_enabled,
                tracking_enabled=self.tracking_enabled,
                webcam_status="Live browser camera connected",
            )
            add_scanlines(output)

            with self.stats.lock:
                self.stats.fps = average_fps
                self.stats.total_objects = sum(counts.values())
                self.stats.active_tracks = len(tracks)
                self.stats.last_frame_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                self.stats.last_error = ""

            if av is None:
                raise RuntimeError("PyAV is not available for WebRTC frame output.")
            return av.VideoFrame.from_ndarray(output, format="bgr24")
        except Exception as error:
            fallback = image.copy()
            alpha_panel(fallback, (20, 20), (min(fallback.shape[1] - 20, 560), 92), (18, 32, 60), 0.82, (80, 150, 255))
            cv2.putText(
                fallback,
                "Processing error",
                (36, 50),
                cv2.FONT_HERSHEY_DUPLEX,
                0.72,
                (240, 246, 252),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                fallback,
                str(error)[:68],
                (36, 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (225, 232, 240),
                1,
                cv2.LINE_AA,
            )
            with self.stats.lock:
                self.stats.last_error = str(error)
                self.stats.last_frame_rgb = cv2.cvtColor(fallback, cv2.COLOR_BGR2RGB)
            if av is None:
                return frame
            return av.VideoFrame.from_ndarray(fallback, format="bgr24")


def init_state() -> None:
    defaults = {
        "last_screenshot": "",
        "detection_enabled": True,
        "tracking_enabled": True,
        "snapshot_tracker": SortLikeTracker(),
        "snapshot_frame_rgb": None,
        "snapshot_stats": {
            "fps": 0.0,
            "total_objects": 0,
            "active_tracks": 0,
            "last_frame_rgb": None,
            "last_error": "",
        },
        "last_capture_hash": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(72, 184, 255, 0.14), transparent 30%),
                radial-gradient(circle at top right, rgba(139, 92, 246, 0.14), transparent 28%),
                linear-gradient(135deg, #04070d 0%, #071320 52%, #050910 100%);
            color: #f2f6fb;
        }
        [data-testid="stSidebar"] {
            background: rgba(8, 15, 28, 0.88);
            border-right: 1px solid rgba(85, 136, 255, 0.22);
        }
        .hero-card, .metric-card, .panel-card {
            background: rgba(10, 18, 30, 0.58);
            border: 1px solid rgba(102, 199, 255, 0.22);
            box-shadow: 0 0 30px rgba(0, 195, 255, 0.08);
            backdrop-filter: blur(16px);
            border-radius: 20px;
        }
        .hero-card {
            padding: 1.5rem 1.6rem;
            margin-bottom: 1rem;
        }
        .hero-kicker {
            color: #63e6ff;
            font-size: 0.82rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin-bottom: 0.6rem;
        }
        .hero-title {
            font-size: 2.3rem;
            line-height: 1.05;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .hero-copy {
            color: #c9d7e6;
            font-size: 1rem;
            max-width: 880px;
        }
        .metric-card {
            padding: 1rem 1.1rem;
            min-height: 118px;
            margin-bottom: 0.7rem;
        }
        .metric-title {
            color: #9eb0c7;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.75rem;
        }
        .metric-value {
            font-size: 1.9rem;
            font-weight: 700;
        }
        .panel-card {
            padding: 1rem 1.2rem;
            margin-top: 1rem;
        }
        .status-pill {
            display: inline-block;
            padding: 0.4rem 0.75rem;
            border-radius: 999px;
            border: 1px solid rgba(104, 230, 255, 0.25);
            background: rgba(10, 18, 30, 0.58);
            color: #eaf5ff;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .stButton > button {
            width: 100%;
            border-radius: 14px;
            border: 1px solid rgba(102, 199, 255, 0.26);
            background: linear-gradient(135deg, rgba(13, 24, 41, 0.95), rgba(18, 44, 69, 0.95));
            color: #f2f6fb;
            box-shadow: 0 0 24px rgba(0, 195, 255, 0.08);
        }
        .stMultiSelect [data-baseweb="tag"] {
            background: rgba(74, 196, 255, 0.18);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, accent: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value" style="color:{accent};">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_snapshot_fallback(
    selected_classes: Sequence[str],
    confidence: float,
    iou: float,
    image_size: int,
) -> dict[str, object]:
    camera_file = st.camera_input("Browser Camera Fallback")
    if camera_file is not None:
        image_bytes = camera_file.getvalue()
        capture_hash = str(hash(image_bytes))
        if capture_hash != st.session_state.last_capture_hash:
            try:
                frame_rgb, snapshot_stats = process_browser_snapshot(
                    image_bytes=image_bytes,
                    selected_classes=selected_classes,
                    confidence=confidence,
                    iou=iou,
                    image_size=image_size,
                )
                st.session_state.snapshot_frame_rgb = frame_rgb
                st.session_state.snapshot_stats = snapshot_stats
                st.session_state.last_capture_hash = capture_hash
            except Exception as error:
                st.session_state.snapshot_stats["last_error"] = str(error)

    if st.session_state.snapshot_frame_rgb is not None:
        st.image(
            st.session_state.snapshot_frame_rgb,
            channels="RGB",
            use_container_width=True,
            caption="Annotated browser camera frame",
        )

    return dict(st.session_state.snapshot_stats)


def runtime_dependency_error() -> Optional[str]:
    missing: list[str] = []
    if av is None:
        missing.append("av")
    if webrtc_streamer is None or WebRtcMode is None:
        missing.append("streamlit-webrtc")
    if not missing:
        return None

    packages = " ".join(missing)
    message = (
        "Missing required packages for browser webcam streaming: "
        f"{', '.join(missing)}. Install them with "
        f"`python -m pip install {packages}` and restart Streamlit."
    )
    if WEBRTC_IMPORT_ERROR:
        message += f" Import error detail: {WEBRTC_IMPORT_ERROR}"
    return message


def main() -> None:
    st.set_page_config(page_title="AI Surveillance Web App", layout="wide")
    init_state()
    inject_styles()

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">Shareable Browser Camera Experience</div>
            <div class="hero-title">Futuristic AI CCTV Detection and Tracking</div>
            <div class="hero-copy">
                This version uses browser webcam access through WebRTC, so each person opening the shared link
                can allow camera permission and run live object detection on their own device.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## Control Center")
        selected_classes = st.multiselect(
            "Focus classes",
            options=list(DEFAULT_CLASSES),
            default=list(DEFAULT_CLASSES),
            format_func=display_class_name,
        )
        confidence = st.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05)
        iou = st.slider("IoU threshold", 0.10, 0.90, 0.45, 0.05)
        image_size = st.select_slider("Inference size", options=[320, 480, 640, 768], value=640)
        if st.button("Toggle Detection", use_container_width=True):
            st.session_state.detection_enabled = not st.session_state.detection_enabled
            if not st.session_state.detection_enabled:
                st.session_state.snapshot_tracker.reset()
            st.rerun()
        if st.button("Toggle Tracking", use_container_width=True):
            st.session_state.tracking_enabled = not st.session_state.tracking_enabled
            if not st.session_state.tracking_enabled:
                st.session_state.snapshot_tracker.reset()
            st.rerun()
        st.markdown("---")
        st.caption("Use a public HTTPS URL for friends. `localhost` works only on your computer.")

    dependency_error = runtime_dependency_error()
    model_error = ""
    if dependency_error is None:
        try:
            load_model(MODEL_PATH)
        except Exception as error:
            model_error = str(error)

    ctx: Any = None
    if dependency_error is None:
        ctx = webrtc_streamer(
            key="ai-surveillance",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1280},
                    "height": {"ideal": 720},
                    "frameRate": {"ideal": 24},
                },
                "audio": False,
            },
            video_processor_factory=DetectionVideoProcessor,
            async_processing=True,
        )

    stats = {
        "fps": 0.0,
        "total_objects": 0,
        "active_tracks": 0,
        "last_frame_rgb": None,
        "last_error": "",
    }
    webcam_health = "WAITING"
    webcam_message = "Press Start above the camera area, then allow browser camera permission."

    if ctx and ctx.video_processor:
        processor = ctx.video_processor
        processor.selected_classes = list(selected_classes)
        processor.confidence = confidence
        processor.iou = iou
        processor.image_size = image_size
        processor.detection_enabled = st.session_state.detection_enabled
        processor.tracking_enabled = st.session_state.tracking_enabled
        stats = processor.stats.snapshot()
        webcam_health = "LIVE" if ctx.state.playing else "READY"
        webcam_message = (
            "Live browser camera connected"
            if ctx.state.playing
            else "Browser camera is ready. Click Start in the streamer control."
        )
    elif dependency_error:
        stats = dict(st.session_state.snapshot_stats)
        webcam_health = "SNAPSHOT"
        webcam_message = "WebRTC is unavailable in this environment. Using browser camera snapshot fallback."

    metrics_placeholder = st.empty()

    def render_metrics(current_stats: dict[str, object], health: str) -> None:
        with metrics_placeholder.container():
            metric_columns = st.columns(4)
            with metric_columns[0]:
                render_metric_card("FPS", f"{float(current_stats['fps']):0.1f}", "#63e6ff")
            with metric_columns[1]:
                render_metric_card("Detected Objects", str(current_stats["total_objects"]), "#7effa9")
            with metric_columns[2]:
                render_metric_card("Active Tracks", str(current_stats["active_tracks"]), "#ff9ef3")
            with metric_columns[3]:
                render_metric_card("Webcam Status", health, "#ffc66d")

    render_metrics(stats, webcam_health)

    left_col, right_col = st.columns([3.25, 1.2])
    with left_col:
        st.markdown("### Live Camera Stream")
        st.caption("Each viewer uses their own browser camera. This is what makes the shared link work.")
        if dependency_error:
            st.warning(dependency_error)
            st.info(
                "This machine cannot load the WebRTC live-stream widget right now, "
                "so the app is using Streamlit's built-in browser camera capture instead."
            )
            stats = render_snapshot_fallback(
                selected_classes=selected_classes,
                confidence=confidence,
                iou=iou,
                image_size=image_size,
            )
        elif ctx and not ctx.state.playing:
            st.info("Use the Start control in the webcam widget below to begin live streaming.")
            st.caption("If the live streamer still does not open, use the fallback camera below.")
            stats = render_snapshot_fallback(
                selected_classes=selected_classes,
                confidence=confidence,
                iou=iou,
                image_size=image_size,
            )

    with right_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("### Live Status")
        st.markdown(
            f'<div class="status-pill">Detection: {"ON" if st.session_state.detection_enabled else "OFF"}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="status-pill">Tracking: {"ON" if st.session_state.tracking_enabled else "OFF"}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="status-pill">Classes: {len(selected_classes)}</div>',
            unsafe_allow_html=True,
        )
        st.write(webcam_message)
        if dependency_error:
            st.warning("Running in snapshot fallback mode because live WebRTC streaming is unavailable.")
        elif model_error:
            st.error(model_error)
        elif stats["last_error"]:
            st.warning(str(stats["last_error"]))
        if st.button("Save Screenshot", use_container_width=True):
            screenshot = save_screenshot(stats["last_frame_rgb"])
            st.session_state.last_screenshot = str(screenshot) if screenshot else "No live frame available yet."
        if st.session_state.last_screenshot:
            st.success(f"Screenshot saved: {st.session_state.last_screenshot}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("### Fixed Class Colors")
        for class_name in selected_classes:
            rgb = class_color(class_name)
            hex_color = "#{:02x}{:02x}{:02x}".format(rgb[2], rgb[1], rgb[0])
            st.markdown(
                f'<div class="status-pill"><span style="display:inline-block;width:10px;height:10px;'
                f'border-radius:50%;background:{hex_color};margin-right:8px;"></span>'
                f'{display_class_name(class_name)}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="panel-card">
            <h3>Sharing This App</h3>
            <p>
                <strong>Important:</strong> <code>http://localhost:8501</code> is only visible on your machine.
                To let friends use this app with their own webcams, share it through a public HTTPS URL
                such as Streamlit Community Cloud, ngrok, or another secure tunnel/reverse proxy.
            </p>
            <p>
                When your friends open that public link, their browser will ask for webcam permission and
                the live detection will run on their camera stream through WebRTC.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if ctx and ctx.state.playing:
        while True:
            if ctx.video_processor:
                try:
                    current_stats = ctx.video_processor.stats.snapshot()
                    render_metrics(current_stats, "LIVE")
                except Exception:
                    pass
            time.sleep(0.1)


if __name__ == "__main__":
    main()
