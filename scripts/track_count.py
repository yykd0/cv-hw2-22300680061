from __future__ import annotations

import argparse
import csv
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np


def side_of_line(point: tuple[float, float], line: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = line
    x, y = point
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def draw_label(frame, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    cv2.rectangle(frame, (x, y - 22), (x + max(80, len(text) * 9), y), color, -1)
    cv2.putText(frame, text, (x + 3, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--line", nargs=4, type=int, default=[320, 120, 320, 620])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    from ultralytics import YOLO

    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    line = tuple(args.line)
    video_path = args.out / "tracking_count.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    tracks = defaultdict(lambda: deque(maxlen=30))
    last_side: dict[int, float] = {}
    counted_ids: set[int] = set()
    events = []
    frame_id = -1
    for result in model.track(
        source=args.source,
        stream=True,
        conf=args.conf,
        iou=args.iou,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    ):
        frame_id += 1
        frame = result.orig_img.copy()
        cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (30, 220, 255), 2)
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            clss = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            for box, track_id, cls_id, conf in zip(boxes, ids, clss, confs):
                x1, y1, x2, y2 = box.astype(int)
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                tracks[track_id].append((frame_id, cx, cy))
                color = (37 * track_id % 255, 17 * track_id % 255, 97 * track_id % 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID {track_id} C{cls_id} {conf:.2f}"
                draw_label(frame, label, x1, max(y1, 24), color)
                pts = [(int(p[1]), int(p[2])) for p in tracks[track_id]]
                for a, b in zip(pts[:-1], pts[1:]):
                    cv2.line(frame, a, b, color, 2)
                side = side_of_line((cx, cy), line)
                if track_id in last_side and track_id not in counted_ids:
                    if last_side[track_id] * side < 0:
                        counted_ids.add(track_id)
                        events.append(
                            {
                                "frame": frame_id,
                                "time_sec": frame_id / fps,
                                "track_id": track_id,
                                "class_id": cls_id,
                                "x": cx,
                                "y": cy,
                            }
                        )
                last_side[track_id] = side
        cv2.putText(
            frame,
            f"Cross count: {len(counted_ids)}",
            (24, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
        if frame_id in {30, 31, 32, 33}:
            cv2.imwrite(str(args.out / f"occlusion_frame_{frame_id:04d}.jpg"), frame)
        writer.write(frame)
    writer.release()
    with open(args.out / "crossing_events.csv", "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=["frame", "time_sec", "track_id", "class_id", "x", "y"])
        writer_csv.writeheader()
        writer_csv.writerows(events)
    with open(args.out / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"cross_count: {len(counted_ids)}\n")
        f.write(f"unique_tracks: {len(tracks)}\n")
        f.write("Use occlusion_frame_0030-0033.jpg for the 3-4 frame ID analysis section.\n")
    print(f"Saved video to {video_path}")
    print(f"Cross count: {len(counted_ids)}")


if __name__ == "__main__":
    main()

