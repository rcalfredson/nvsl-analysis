"""Reconstruct rt-trx displays from saved, uninterpolated tracking samples.

Only NumPy and OpenCV are needed; importing this module does not run analysis.
"""

import argparse
from pathlib import Path
import pickle
import time

import cv2
import numpy as np

from src.utils.heatmap import render_heatmap
from src.utils.video_playback import FrameCommandInput, parse_frame_position


class FlyDetector:
    """Compatibility placeholder for legacy tracker metadata pickles."""


class TrackerUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "FlyDetector":
            return FlyDetector
        return super().find_class(module, name)


def load_pickle(path):
    with path.open("rb") as stream:
        return TrackerUnpickler(stream, encoding="latin1").load()


def round_tracker(value):
    """Python 2 intR rounding used by rt-trx (half away from zero)."""
    value = np.asarray(value)
    return (np.sign(value) * np.floor(np.abs(value) + 0.5)).astype(int)


def sidecar(video, suffix):
    # Both names occur in tracker versions used by this repository.
    candidates = (video.with_suffix(suffix), Path(str(video) + suffix))
    return next((p for p in candidates if p.exists()), candidates[0])


def separators(protocol, shape, layout="auto"):
    ct = protocol.get("ct")
    if ct not in ("regular", "htl", "large"):
        raise ValueError("metadata must specify chamber type (protocol.ct)")
    tm = protocol.get("tm")
    if tm is None:
        raise ValueError("saved template match (protocol.tm) is needed for chamber normalization")
    if ct == "htl":
        xs, ys = [112, 260, 407, 554], [152, 332, 511]
    elif ct == "large":
        if layout == "auto":
            raise ValueError("large chambers require --large-layout original or new (295-pixel floor)")
        sep = 268 if layout == "original" else 321
        xs, ys = [sep], [sep]
    else:
        xs, ys = [98], []
    factor = tm["fctr"] * tm.get("rszFctr", 1.5 if shape[0] == 1080 else 1)
    result = []
    for values, offset, limit in zip((xs, ys), (tm["x"], tm["y"]), shape[1::-1]):
        values = round_tracker((round_tracker(np.array(values) * factor + offset)) / 2).tolist()
        if any(v <= 0 or v >= int(round_tracker(limit / 2)) for v in values):
            raise ValueError("template chamber separators lie outside the video")
        result.append(values)
    return result


class Replay:
    def __init__(self, protocol, tracks, shape, fps, minutes=5, layout="auto"):
        self.protocol, self.shape, self.fps = protocol, shape[:2], fps
        self.tracks = {k: np.asarray(tracks[k], dtype=float) for k in ("x", "y", "w", "h", "theta")}
        sizes = {a.shape for a in self.tracks.values()}
        if len(sizes) != 1 or self.tracks["x"].ndim != 2:
            raise ValueError("tracking arrays must have matching (flies, frames) shapes")
        self.nf = self.tracks["x"].shape[1]
        self.window = max(1, int(round_tracker(minutes * 60 * fps)))
        self.seps = separators(protocol, shape, layout)
        self.running = np.ones(tuple(round_tracker(np.array(self.shape) / 2)), np.float32)
        self.protocol_map = self.running.copy()
        fns, info = protocol["frameNums"], protocol["info"]
        self.sessions = []
        for events, geometry in zip(fns if isinstance(fns, list) else [fns],
                                    info if isinstance(info, list) else [info]):
            sessions = []
            for j, start in enumerate(events.get("startTrain", [])):
                stops = events.get("startPost", [])
                stop = stops[j] if j < len(stops) else self.nf
                center = geometry.get("cPos", [])
                radii = geometry.get("r", [])
                radius = radii[j] if j < len(radii) else protocol.get("circle", protocol.get("area", {})).get("r")
                if not np.isscalar(radius) and radius is not None:
                    radius = radius[j]
                sessions.append((int(start), int(stop), center[j] if j < len(center) else None, radius))
            self.sessions.append(sessions)
        # rt-trx controls the protocol map from its first experimental fly.
        self.reference = next((s for s in self.sessions if s), [])
        self.index = -1
        self.session = None

    def add(self, target, start, stop, sign=1):
        x, y = (self.tracks[k][:, start:stop].ravel() for k in ("x", "y"))
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = round_tracker(x[valid] / 2), round_tracker(y[valid] / 2)
        valid = (x >= 0) & (x < target.shape[1]) & (y >= 0) & (y < target.shape[0])
        np.add.at(target, (y[valid], x[valid]), sign)

    def seek(self, index):
        if not 0 <= index < self.nf:
            raise ValueError("frame outside tracking data")
        session = next((s for s in reversed(self.reference) if s[0] <= index), None)
        if index != self.index + 1:
            self.running.fill(1)
            self.add(self.running, max(0, index - self.window + 1), index + 1)
        else:
            self.add(self.running, index, index + 1)
            if index >= self.window:
                self.add(self.running, index - self.window, index - self.window + 1, -1)
        if session != self.session or index != self.index + 1:
            self.protocol_map.fill(1)
            if session:
                self.add(self.protocol_map, session[0], min(index + 1, session[1]))
        elif session and index < session[1]:
            self.add(self.protocol_map, index, index + 1)
        self.index, self.session = index, session

    def heatmap(self, mode):
        counts = self.running if mode == "running" else self.protocol_map
        img = render_heatmap(counts, xy_seps=self.seps)
        return cv2.resize(img, self.shape[::-1])

    def render(self, frame, mode="running"):
        left = frame.copy()
        for sessions in self.sessions:
            for start, stop, center, radius in sessions:
                if start <= self.index < stop and center is not None and radius is not None:
                    cv2.circle(left, tuple(round_tracker(center)), int(round_tracker(radius)), (255, 255, 255), 1)
        for fly in range(self.tracks["x"].shape[0]):
            x, y, w, h, angle = (self.tracks[k][fly, self.index] for k in ("x", "y", "w", "h", "theta"))
            if np.isfinite([x, y, w, h, angle]).all() and w > 0 and h > 0:
                cv2.ellipse(left, ((x, y), (w, h), angle), (0, 255, 255), 1)
        heat = self.heatmap(mode)
        height, width = self.shape
        # Pad to even dimensions for common MP4 encoders.
        img = np.zeros(((height + 33) // 2 * 2, width * 2, 3), np.uint8)
        img[32:32 + height, :width] = left
        img[32:32 + height, width:] = heat
        seconds = self.index / self.fps
        label = f"{int(seconds // 3600):02d}:{int(seconds // 60 % 60):02d}:{seconds % 60:05.2f}  frame {self.index}"
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        title = f"Running heatmap ({self.window / self.fps / 60:g} min)"
        if mode == "protocol":
            elapsed = 0 if self.session is None else (min(self.index + 1, self.session[1]) - self.session[0]) / self.fps / 60
            title = f"Training heatmap ({elapsed:.1f} min)"
        cv2.putText(img, title, (width + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        return img


class Recorder:
    def __init__(self, path, fps):
        self.path, self.fps, self.writer = Path(path), fps, None
        self.last_frame = None
        self.clip = 0

    def start(self, image):
        path = self.path if self.clip == 0 else self.path.with_name(f"{self.path.stem}_{self.clip + 1:03d}.mp4")
        if path.exists():
            raise ValueError(f"output already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, image.shape[1::-1])
        if not self.writer.isOpened():
            self.writer.release()
            self.writer = None
            raise ValueError(f"could not open MP4 writer: {path}")
        self.last_frame = None
        self.clip += 1
        print(f"Recording {path}", flush=True)

    def write(self, image, index):
        if self.writer is not None and index != self.last_frame:
            self.writer.write(image)
            self.last_frame = index

    def stop(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print("Recording saved.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Replay all flies beside an rt-trx-style live heatmap.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--start", default="0", help="frame or HH:MM:SS (inclusive)")
    parser.add_argument("--end", help="frame or HH:MM:SS (exclusive)")
    parser.add_argument("--heatmap-minutes", type=float, default=5)
    parser.add_argument("--heatmap", choices=("running", "protocol"), default="running")
    parser.add_argument("--large-layout", choices=("auto", "original", "new"), default="auto")
    parser.add_argument("--speed", type=float, default=1, help="preview speed; MP4 always uses source fps")
    parser.add_argument("--output", type=Path, help="MP4 path (default: VIDEO_replay.mp4)")
    parser.add_argument("--record", action="store_true", help="record immediately")
    parser.add_argument("--headless", action="store_true", help="export selected interval without a display")
    args = parser.parse_args(argv)
    if not np.isfinite([args.speed, args.heatmap_minutes]).all() or min(args.speed, args.heatmap_minutes) <= 0:
        parser.error("speed and heatmap minutes must be finite and positive")
    output = args.output or args.video.with_name(args.video.stem + "_replay.mp4")
    if output.suffix.lower() != ".mp4":
        parser.error("output must have an .mp4 extension")
    cap = cv2.VideoCapture(str(args.video))
    recorder = None
    try:
        if not cap.isOpened():
            raise ValueError(f"could not open video: {args.video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("video has no valid frame rate")
        ok, frame = cap.read()
        if not ok:
            raise ValueError("could not read first frame")
        replay = Replay(load_pickle(sidecar(args.video, ".data"))["protocol"],
                        load_pickle(sidecar(args.video, ".trx")), frame.shape, fps,
                        args.heatmap_minutes, args.large_layout)
        nf = min(replay.nf, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        index = parse_frame_position(args.start, fps)
        end = nf if args.end is None else parse_frame_position(args.end, fps)
        if not 0 <= index < end <= nf:
            raise ValueError(f"require 0 <= start < end <= {nf}")
        recorder = Recorder(output, fps)
        mode, paused = args.heatmap, False
        frame_input = FrameCommandInput(fps)
        previous, image = None, None
        auto_record = args.record or args.headless
        deadline = time.monotonic()
        print("Space: pause/play | r: record on/off | m: heatmap mode | ,/.: step | frame or HH:MM:SS then g: seek | q: quit", flush=True)
        while index < end:
            if previous != index:
                if previous is None or index != previous + 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"could not read frame {index}")
                replay.seek(index)
                previous = index
                image = replay.render(frame, mode)
            if auto_record:
                recorder.start(image)
                auto_record = False
            recorder.write(image, index)
            if args.headless:
                index += 1
                continue
            display = image.copy()
            if recorder.writer is not None:
                cv2.putText(display, "REC", (display.shape[1] - 55, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
            cv2.imshow("Real-time replay", display)
            key = cv2.waitKey(1) & 255
            if key == ord("q") or cv2.getWindowProperty("Real-time replay", cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(" "):
                paused = not paused
                deadline = time.monotonic()
            elif key == ord("r"):
                if recorder.writer is None:
                    recorder.start(image)
                    recorder.write(image, index)
                else:
                    recorder.stop()
            elif key == ord("m"):
                mode = "protocol" if mode == "running" else "running"
                image = replay.render(frame, mode)
            elif key in (ord(","), ord("."), ord("g")):
                target = index + (1 if key == ord(".") else -1)
                if key == ord("g"):
                    target = frame_input.consume()
                    if target is None:
                        target = index
                if 0 <= target < end:
                    recorder.stop()
                    index, paused = target, True
                    continue
            else:
                frame_input.feed(chr(key))
            if not paused and time.monotonic() >= deadline + 1 / (fps * args.speed):
                index += 1
                # Fall behind by slowing preview, never skip exported samples.
                deadline = max(deadline + 1 / (fps * args.speed), time.monotonic() - 1 / (fps * args.speed))
    except (ValueError, OSError, KeyError, pickle.UnpicklingError, cv2.error) as exc:
        parser.exit(1, f"replay: {exc}\n")
    finally:
        if recorder:
            recorder.stop()
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
