from __future__ import annotations

from pathlib import Path
from threading import Lock

import carla
import cv2
import numpy as np


class UrbanCamera:
    def __init__(self, vehicle: carla.Vehicle, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        self.out = out
        self.caption = "Urban drive"
        self.frames = 0
        self.lock = Lock()
        self.last_image = None
        self.writer = cv2.VideoWriter(str(out / "drive.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (960, 540))
        if not self.writer.isOpened():
            raise RuntimeError("Could not create the driving video.")
        world = vehicle.get_world()
        blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", "960")
        blueprint.set_attribute("image_size_y", "540")
        blueprint.set_attribute("sensor_tick", "0.2")
        blueprint.set_attribute("fov", "100")
        pose = carla.Transform(carla.Location(x=-7.0, z=4.0), carla.Rotation(pitch=-18.0))
        self.sensor = world.spawn_actor(blueprint, pose, attach_to=vehicle)
        self.sensor.listen(self._image)

    def _image(self, image: carla.Image) -> None:
        pixels = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
        frame = np.ascontiguousarray(pixels[:, :, :3])
        cv2.rectangle(frame, (0, 0), (960, 34), (20, 20, 20), -1)
        cv2.putText(frame, self.caption, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
        with self.lock:
            self.writer.write(frame)
            self.last_image = frame
            self.frames += 1

    def snapshot(self, label: str) -> None:
        with self.lock:
            if self.last_image is not None:
                cv2.imwrite(str(self.out / f"{label}.jpg"), self.last_image)

    def close(self) -> None:
        self.sensor.stop()
        self.sensor.destroy()
        with self.lock:
            self.writer.release()
