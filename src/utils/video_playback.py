"""Frame/time command entry shared by OpenCV video players."""

import math
import time


def parse_frame_position(value, fps, *, strict_clock=False, nearest=True):
    """Parse a frame number or clock time.

    The annotated analysis player keeps its historical HH:MM:SS clock syntax
    and truncation. Replay also accepts MM:SS and fractional seconds, rounding
    to the nearest source frame. Integer frame numbers are always exact.
    """
    if value.isdigit():
        return int(value)
    if strict_clock:
        clock = time.strptime(value, "%H:%M:%S")
        seconds = clock.tm_hour * 3600 + clock.tm_min * 60 + clock.tm_sec
    else:
        parts = value.split(":")
        if len(parts) not in (2, 3):
            raise ValueError("use a frame number or HH:MM:SS[.fraction]")
        seconds = 0.
        for part in parts:
            seconds = seconds * 60 + float(part)
    frames = seconds * fps
    if not math.isfinite(frames):
        raise ValueError("frame position must be finite")
    return int(math.copysign(math.floor(abs(frames) + .5), frames)) if nearest else int(frames)


class FrameCommandInput:
    """Accumulate typed digits/colons, then consume them for a go/length command."""

    def __init__(self, fps, *, strict_clock=False, nearest=True):
        self.fps = fps
        self.strict_clock, self.nearest = strict_clock, nearest
        self.text = ""

    def feed(self, key):
        if key in "0123456789:":
            self.text += key

    def consume(self):
        text, self.text = self.text, ""
        try:
            return parse_frame_position(text, self.fps, strict_clock=self.strict_clock,
                                        nearest=self.nearest)
        except ValueError:
            return None
