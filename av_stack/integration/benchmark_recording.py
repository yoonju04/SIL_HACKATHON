from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class SensorFrame:
    number: int
    elapsed: float
    speed: float
    pixels: np.ndarray


class FrontRecorder:
    def __init__(self, folder: Path) -> None:
        self.frames = (folder/'front_frames.jsonl').open('w', encoding='utf-8')
        self.writer = cv2.VideoWriter(str(folder/'front.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20., (800, 450))
        if not self.writer.isOpened():
            self.frames.close()
            raise RuntimeError('Could not open front camera video writer.')

    def capture(self, frame: SensorFrame) -> None:
        pixels = cv2.resize(frame.pixels[:, :, :3], (800, 450))
        self.writer.write(pixels)
        self.frames.write(json.dumps(dict(sensor_frame=frame.number, t=frame.elapsed, speed=frame.speed))+chr(10))
        self.frames.flush()

    def close(self) -> None:
        self.writer.release()
        self.frames.close()
