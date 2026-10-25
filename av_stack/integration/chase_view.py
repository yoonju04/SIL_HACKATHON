"""Demo window: chase camera with a fixed on-screen steering label (and optional view.mp4).

CARLA has no 2D HUD API; 3D debug text drifts and smears with the follow camera. This view
spawns its own camera behind the ego (same pose as integration.carla_io.follow_camera) and
draws the label on the image, so it stays fixed in the top-left corner.
"""
from __future__ import annotations

from pathlib import Path
import threading

import cv2
import numpy as np

WIDTH, HEIGHT = 1280, 720
LABELS = {'mpc': ('STEERING: MPC', (0, 220, 0)),
          'lqr': ('STEERING: LQR (fallback)', (0, 150, 255)),
          'safe_stop': ('STEERING: SAFE STOP', (0, 0, 230)),
          'none': ('STEERING: NONE', (0, 0, 230))}


class ChaseView:
    def __init__(self, carla_module, world, vehicle, video: Path | None = None, fps: float = 20.) -> None:
        self.carla = carla_module
        blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
        for key, value in (('image_size_x', WIDTH), ('image_size_y', HEIGHT), ('fov', 90)):
            blueprint.set_attribute(key, str(value))
        pose = carla_module.Transform(carla_module.Location(x=-8., z=4.), carla_module.Rotation(pitch=-15.))
        self.camera = world.spawn_actor(blueprint, pose, attach_to=vehicle)
        self.lock = threading.Lock()
        self.frame = None
        self.camera.listen(self._store)
        self.writer = None
        if video is not None:
            self.writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), fps, (WIDTH, HEIGHT))
        cv2.namedWindow('av_stack view', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('av_stack view', WIDTH, HEIGHT)

    def _store(self, image) -> None:
        pixels = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)[:, :, :3]
        with self.lock:
            self.frame = pixels.copy()

    def show(self, active: str) -> None:
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
        if frame is None:
            return
        text, color = LABELS.get(active, ('STEERING: ' + active.upper(), (255, 255, 255)))
        (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(frame, (14, 14), (34 + width, 34 + height + 6), (0, 0, 0), -1)
        cv2.putText(frame, text, (24, 30 + height), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        if self.writer is not None:
            self.writer.write(frame)
        cv2.imshow('av_stack view', frame)
        cv2.waitKey(1)

    def close(self) -> None:
        try:
            self.camera.stop()
            self.camera.destroy()
        finally:
            if self.writer is not None:
                self.writer.release()
            cv2.destroyWindow('av_stack view')
