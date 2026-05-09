from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np
from ultralytics import YOLO


Box = tuple[int, int, int, int]
DEFAULT_CLASSES = (
    "person",
    "chair",
    "laptop",
    "cell phone",
    "bottle",
    "book",
    "keyboard",
    "tv",
    "mouse",
)
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "person": (80, 220, 120),
    "chair": (255, 120, 70),
    "laptop": (180, 90, 255),
    "cell phone": (255, 220, 80),
    "bottle": (70, 180, 255),
    "book": (90, 235, 235),
    "keyboard": (255, 110, 200),
    "tv": (80, 80, 255),
    "mouse": (255, 170, 80),
}
DISPLAY_NAMES = {"tv": "monitor"}


@dataclass
class AppConfig:
    model_path: str
    confidence: float
    iou: float
    image_size: int
    target_classes: tuple[str, ...]
    camera_indices: tuple[int, ...]
    camera_backends: tuple[int, ...]
    camera_width: int
    camera_height: int
    camera_warmup_frames: int
    max_read_retries: int
    enable_alerts: bool
    diagnose_camera: bool
    no_display: bool
    tracker_iou_threshold: float
    tracker_max_age: int
    tracker_min_hits: int


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
    history: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=28))

    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def predict_box(self) -> Box:
        dx, dy = self.velocity
        x1, y1, x2, y2 = self.box
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


class SortLikeTracker:
    def __init__(self, iou_threshold: float, max_age: int, min_hits: int) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.next_track_id = 1
        self.tracks: list[Track] = []

    def update(self, detections: Sequence[Detection]) -> list[Track]:
        predicted_boxes: list[Box] = []
        for track in self.tracks:
            track.missed_frames += 1
            predicted_boxes.append(track.predict_box())

        candidate_matches: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self.tracks):
            for detection_index, detection in enumerate(detections):
                if track.class_id != detection.class_id:
                    continue

                iou_score = compute_iou(predicted_boxes[track_index], detection.box)
                if iou_score < self.iou_threshold:
                    continue

                track_center = track.center()
                detection_center = box_center(detection.box)
                distance = np.hypot(
                    detection_center[0] - track_center[0],
                    detection_center[1] - track_center[1],
                )
                score = iou_score - distance * 0.0005
                candidate_matches.append((score, track_index, detection_index))

        candidate_matches.sort(reverse=True, key=lambda item: item[0])

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for _, track_index, detection_index in candidate_matches:
            if track_index in matched_tracks or detection_index in matched_detections:
                continue

            detection = detections[detection_index]
            track = self.tracks[track_index]
            previous_center = track.center()
            new_center = box_center(detection.box)

            track.velocity = (
                new_center[0] - previous_center[0],
                new_center[1] - previous_center[1],
            )
            track.box = detection.box
            track.class_id = detection.class_id
            track.class_name = detection.class_name
            track.confidence = detection.confidence
            track.hits += 1
            track.missed_frames = 0
            track.history.append(new_center)

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

        self.tracks = [
            track for track in self.tracks if track.missed_frames <= self.max_age
        ]
        return [
            track
            for track in self.tracks
            if track.missed_frames == 0 and track.hits >= self.min_hits
        ]


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(
        description="Live webcam AI CCTV using YOLOv8 and SORT-style tracking."
    )
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLO model.")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument(
        "--classes",
        nargs="*",
        default=list(DEFAULT_CLASSES),
        help="Live focus classes. Defaults to a CCTV-friendly object set.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Optional webcam index. If omitted, the app auto-detects a usable camera.",
    )
    parser.add_argument(
        "--alerts",
        action="store_true",
        help="Show live alert banners for focused object classes.",
    )
    parser.add_argument(
        "--diagnose-camera",
        action="store_true",
        help="Check webcam availability for known indices and backends.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without creating a display window.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Requested webcam width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Requested webcam height.",
    )
    args = parser.parse_args()

    indices = (args.camera,) if args.camera is not None else (0, 1, 2, 3, 4, 5)
    return AppConfig(
        model_path=str(args.model),
        confidence=args.conf,
        iou=args.iou,
        image_size=args.imgsz,
        target_classes=tuple(normalize_class_name(item) for item in args.classes),
        camera_indices=indices,
        camera_backends=(cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY),
        camera_width=args.width,
        camera_height=args.height,
        camera_warmup_frames=14,
        max_read_retries=6,
        enable_alerts=args.alerts,
        diagnose_camera=args.diagnose_camera,
        no_display=args.no_display,
        tracker_iou_threshold=0.25,
        tracker_max_age=18,
        tracker_min_hits=2,
    )


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
    canonical = normalize_class_name(name)
    label = DISPLAY_NAMES.get(canonical, canonical)
    return label.title()


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


def box_center(box: Box) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def backend_name(backend: int) -> str:
    names = {
        cv2.CAP_ANY: "CAP_ANY",
        cv2.CAP_DSHOW: "CAP_DSHOW",
        cv2.CAP_MSMF: "CAP_MSMF",
    }
    return names.get(backend, str(backend))


def configure_opencv_logging() -> None:
    if not hasattr(cv2, "setLogLevel"):
        return

    for level_name in ("LOG_LEVEL_ERROR", "LOG_LEVEL_SILENT"):
        if hasattr(cv2, level_name):
            cv2.setLogLevel(getattr(cv2, level_name))
            return


def load_model(model_path: str) -> YOLO:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"YOLO model file not found: {path}")
    return YOLO(str(path))


def frame_has_content(frame) -> bool:
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(grayscale.mean()) > 3.0


def try_open_camera(index: int, config: AppConfig) -> Optional[cv2.VideoCapture]:
    for backend in config.camera_backends:
        capture = cv2.VideoCapture(index, backend)
        if not capture.isOpened():
            capture.release()
            continue

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        for _ in range(config.camera_warmup_frames):
            ok, frame = capture.read()
            if ok and frame is not None and frame_has_content(frame):
                return capture
            time.sleep(0.05)

        capture.release()
    return None


def diagnose_cameras(config: AppConfig) -> int:
    print("Live webcam diagnostic started")
    print(f"Python: {sys.version.split()[0]}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"Camera indices checked: {', '.join(str(index) for index in config.camera_indices)}")
    print(
        "Backends checked: "
        + ", ".join(backend_name(backend) for backend in config.camera_backends)
    )
    print("")

    found_any = False
    for index in config.camera_indices:
        print(f"[Camera {index}]")
        for backend in config.camera_backends:
            capture = cv2.VideoCapture(index, backend)
            name = backend_name(backend)
            if not capture.isOpened():
                print(f"  - {name}: not available")
                capture.release()
                continue

            ok, frame = capture.read()
            if ok and frame is not None and frame_has_content(frame):
                height, width = frame.shape[:2]
                print(f"  - {name}: OK ({width}x{height})")
                found_any = True
            elif ok and frame is not None:
                print(f"  - {name}: opened but frames were blank")
            else:
                print(f"  - {name}: opened but frame read failed")
            capture.release()
        print("")

    if found_any:
        print("At least one usable live webcam was found.")
        return 0

    print("No usable live webcam was found.")
    print("Close apps like Camera, Zoom, Teams, or browser tabs and retry.")
    return 1


def open_capture(config: AppConfig) -> cv2.VideoCapture:
    for index in config.camera_indices:
        capture = try_open_camera(index, config)
        if capture is None:
            continue

        backend = int(capture.get(cv2.CAP_PROP_BACKEND))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(
            f"[INFO] Live webcam connected on camera {index} using "
            f"{backend_name(backend)} at {width}x{height}"
        )
        return capture

    requested = ", ".join(str(index) for index in config.camera_indices)
    raise RuntimeError(
        "Unable to open a live webcam. Checked camera indices: "
        f"{requested}. Verify camera permissions and that no other app is using it."
    )


def detect_objects(model: YOLO, frame, config: AppConfig) -> list[Detection]:
    results = model.predict(
        source=frame,
        conf=config.confidence,
        iou=config.iou,
        imgsz=config.image_size,
        agnostic_nms=False,
        verbose=False,
    )

    detections: list[Detection] = []
    allowed_classes = set(config.target_classes)
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = normalize_class_name(str(model.names[class_id]))
            if allowed_classes and class_name not in allowed_classes:
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


def class_color(class_name: str) -> tuple[int, int, int]:
    canonical = normalize_class_name(class_name)
    return CLASS_COLORS.get(canonical, (180, 180, 180))


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
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y1), (x1 + 24, y1 + 24), color, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (x2 - 24, y2 - 24), (x2, y2), color, 2, cv2.LINE_AA)


def draw_label_chip(
    frame,
    box: Box,
    label: str,
    color: tuple[int, int, int],
) -> None:
    x1, y1, _, _ = box
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.56
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
    top = max(8, y1 - text_height - baseline - 16)
    bottom = top + text_height + baseline + 12

    alpha_panel(
        frame,
        (x1, top),
        (x1 + text_width + 18, bottom),
        fill=color,
        alpha=0.90,
    )
    cv2.putText(
        frame,
        label,
        (x1 + 8, bottom - baseline - 4),
        font,
        scale,
        (15, 18, 24),
        thickness,
        cv2.LINE_AA,
    )


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
            fade = max(0.20, index / len(points))
            trail_color = tuple(int(channel * fade) for channel in color)
            cv2.line(frame, points[index - 1], points[index], trail_color, 2, cv2.LINE_AA)
        if points:
            cv2.circle(frame, points[-1], 3, color, -1, cv2.LINE_AA)

    return counts


def draw_detections(frame, detections: Sequence[Detection]) -> Counter:
    counts: Counter = Counter()
    for detection in detections:
        counts[detection.class_name] += 1
        color = class_color(detection.class_name)
        glow_box(frame, detection.box, color)
        label = f"{display_class_name(detection.class_name)}  {detection.confidence:.2f}"
        draw_label_chip(frame, detection.box, label, color)
    return counts


def draw_count_table(frame, counts: Counter) -> None:
    if not counts:
        return

    items = counts.most_common(8)
    panel_height = 34 + len(items) * 28
    alpha_panel(frame, (14, 154), (270, 154 + panel_height), (10, 16, 28), 0.70, (58, 90, 120))
    cv2.putText(
        frame,
        "OBJECT COUNT",
        (28, 182),
        cv2.FONT_HERSHEY_DUPLEX,
        0.62,
        (210, 240, 255),
        1,
        cv2.LINE_AA,
    )

    y = 210
    for class_name, count in items:
        color = class_color(class_name)
        cv2.circle(frame, (30, y - 5), 7, color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"{display_class_name(class_name)}: {count}",
            (46, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (235, 240, 244),
            1,
            cv2.LINE_AA,
        )
        y += 28


def draw_detection_legend(frame, focus_classes: Iterable[str]) -> None:
    classes = [normalize_class_name(item) for item in focus_classes]
    if not classes:
        return

    panel_height = 28 + len(classes) * 24
    width = frame.shape[1]
    start_x = max(390, width - 280)
    alpha_panel(
        frame,
        (start_x, 62),
        (width - 14, 62 + panel_height),
        (10, 16, 28),
        0.70,
        (58, 90, 120),
    )
    cv2.putText(
        frame,
        "FOCUSED CLASSES",
        (start_x + 14, 88),
        cv2.FONT_HERSHEY_DUPLEX,
        0.54,
        (210, 240, 255),
        1,
        cv2.LINE_AA,
    )

    y = 112
    for class_name in classes:
        color = class_color(class_name)
        cv2.circle(frame, (start_x + 16, y - 5), 6, color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            display_class_name(class_name),
            (start_x + 30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (234, 239, 242),
            1,
            cv2.LINE_AA,
        )
        y += 24


def make_alert_message(
    counts: Counter, focus_classes: Iterable[str], enabled: bool
) -> Optional[str]:
    if not enabled:
        return None

    for class_name in focus_classes:
        if counts.get(class_name, 0) > 0:
            return f"Live alert: {display_class_name(class_name)} detected"
    return None


def draw_status_panel(
    frame,
    fps: float,
    counts: Counter,
    detection_enabled: bool,
    tracking_enabled: bool,
    focus_classes: Iterable[str],
    alert_message: Optional[str],
) -> None:
    height, width = frame.shape[:2]
    alpha_panel(frame, (14, 14), (360, 138), (10, 16, 28), 0.70, (58, 90, 120))

    lines = [
        "AI CCTV SURVEILLANCE",
        f"Objects: {sum(counts.values())}",
        f"Detection: {'ON' if detection_enabled else 'OFF'}",
        f"Tracking: {'ON' if tracking_enabled else 'OFF'}",
    ]
    y = 40
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (28, y),
            cv2.FONT_HERSHEY_DUPLEX if index == 0 else cv2.FONT_HERSHEY_SIMPLEX,
            0.70 if index == 0 else 0.60,
            (245, 250, 252) if index == 0 else (218, 226, 232),
            1 if index == 0 else 1,
            cv2.LINE_AA,
        )
        y += 24

    fps_label = f"FPS {fps:5.1f}"
    fps_width = cv2.getTextSize(fps_label, cv2.FONT_HERSHEY_DUPLEX, 0.84, 1)[0][0]
    alpha_panel(
        frame,
        (width - fps_width - 42, 14),
        (width - 14, 52),
        (8, 28, 34),
        0.76,
        (70, 210, 210),
    )
    cv2.putText(
        frame,
        fps_label,
        (width - fps_width - 26, 40),
        cv2.FONT_HERSHEY_DUPLEX,
        0.84,
        (90, 255, 222),
        1,
        cv2.LINE_AA,
    )

    draw_count_table(frame, counts)
    draw_detection_legend(frame, focus_classes)

    controls = "Q/ESC Exit   S Save   D Detect   T Track"
    alpha_panel(frame, (14, height - 42), (346, height - 14), (10, 16, 28), 0.70, (58, 90, 120))
    cv2.putText(
        frame,
        controls,
        (24, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (228, 235, 240),
        1,
        cv2.LINE_AA,
    )

    if alert_message:
        alert_width = min(width - 28, 430)
        alpha_panel(
            frame,
            (14, height - 88),
            (14 + alert_width, height - 50),
            (0, 78, 160),
            0.84,
            (70, 190, 255),
        )
        cv2.putText(
            frame,
            alert_message,
            (28, height - 63),
            cv2.FONT_HERSHEY_DUPLEX,
            0.58,
            (245, 248, 252),
            1,
            cv2.LINE_AA,
        )


def add_cctv_scanlines(frame) -> None:
    overlay = frame.copy()
    for y in range(0, frame.shape[0], 6):
        cv2.line(overlay, (0, y), (frame.shape[1], y), (0, 0, 0), 1)
    cv2.addWeighted(overlay, 0.06, frame, 0.94, 0, frame)


def save_screenshot(frame) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = Path(f"screenshot_{timestamp}.jpg")
    cv2.imwrite(str(path), frame)
    return path


def process_webcam(model: YOLO, config: AppConfig) -> int:
    capture = open_capture(config)
    tracker = SortLikeTracker(
        iou_threshold=config.tracker_iou_threshold,
        max_age=config.tracker_max_age,
        min_hits=config.tracker_min_hits,
    )

    window_name = "AI CCTV Surveillance"
    if not config.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, config.camera_width, config.camera_height)

    detection_enabled = True
    tracking_enabled = True
    fps_history: deque[float] = deque(maxlen=30)

    try:
        while True:
            started = time.perf_counter()

            ok = False
            frame = None
            for _ in range(config.max_read_retries):
                ok, frame = capture.read()
                if ok and frame is not None:
                    break
                time.sleep(0.03)

            if not ok or frame is None:
                print("[ERROR] Webcam frame read failed repeatedly. Stopping live monitoring.")
                break

            detections = detect_objects(model, frame, config) if detection_enabled else []
            tracks = tracker.update(detections) if detection_enabled and tracking_enabled else []

            output_frame = frame.copy()
            if detection_enabled and tracking_enabled:
                counts = draw_tracks(output_frame, tracks)
            elif detection_enabled:
                counts = draw_detections(output_frame, detections)
            else:
                counts = Counter()

            fps_history.append(1.0 / max(time.perf_counter() - started, 1e-6))
            average_fps = sum(fps_history) / len(fps_history)

            draw_status_panel(
                output_frame,
                fps=average_fps,
                counts=counts,
                detection_enabled=detection_enabled,
                tracking_enabled=tracking_enabled,
                focus_classes=config.target_classes,
                alert_message=make_alert_message(
                    counts,
                    config.target_classes,
                    config.enable_alerts,
                ),
            )
            add_cctv_scanlines(output_frame)

            key = 255
            if not config.no_display:
                cv2.imshow(window_name, output_frame)
                key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                screenshot_path = save_screenshot(output_frame)
                print(f"[INFO] Screenshot saved to {screenshot_path}")
            if key == ord("d"):
                detection_enabled = not detection_enabled
                print(f"[INFO] Detection {'enabled' if detection_enabled else 'disabled'}")
            if key == ord("t"):
                tracking_enabled = not tracking_enabled
                print(f"[INFO] Tracking {'enabled' if tracking_enabled else 'disabled'}")
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


def run() -> int:
    config = parse_args()
    configure_opencv_logging()

    if config.diagnose_camera:
        return diagnose_cameras(config)

    try:
        model = load_model(config.model_path)
    except Exception as error:
        print(f"[ERROR] {error}")
        return 1

    try:
        return process_webcam(model, config)
    except Exception as error:
        print(f"[ERROR] {error}")
        return 1


if __name__ == "__main__":
    sys.exit(run())
