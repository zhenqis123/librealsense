"""Simple PyQt5 GUI for streaming and recording Intel RealSense frames.

The application shows the RGB stream alongside a colorized depth stream in
real-time. Users can start, pause/resume, and stop a recording session. While
recording, both RGB and raw depth frames are saved to disk together with a CSV
file that keeps the associated timestamps.

Requirements:
    * pyrealsense2
    * PyQt5
    * numpy
    * opencv-python (for writing PNG files)

Usage:
    python pyqt_realsense_recorder.py
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
import pyrealsense2 as rs
from PyQt5 import QtCore, QtGui, QtWidgets


class RealSenseRecorder(QtWidgets.QWidget):
    """Qt widget that displays live color/depth streams and records them."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self.setWindowTitle("RealSense 采集工具")
        self.resize(1280, 520)

        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self._config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self._profile = self._pipeline.start(self._config)
        self._colorizer = rs.colorizer()

        self._frame_timer = QtCore.QTimer(self)
        self._frame_timer.timeout.connect(self._update_frames)
        self._frame_timer.start(30)

        self._recording_dir: Path | None = None
        self._csv_file = None
        self._csv_writer: csv.writer | None = None
        self._frame_index = 0
        self._session_active = False
        self._recording_enabled = False

        self._build_ui()

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        """Create labels, buttons, and layout."""

        main_layout = QtWidgets.QVBoxLayout(self)

        stream_layout = QtWidgets.QHBoxLayout()
        self._color_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self._color_label.setMinimumSize(640, 480)
        self._color_label.setFrameShape(QtWidgets.QFrame.Box)
        self._depth_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self._depth_label.setMinimumSize(640, 480)
        self._depth_label.setFrameShape(QtWidgets.QFrame.Box)

        stream_layout.addWidget(self._color_label)
        stream_layout.addWidget(self._depth_label)
        main_layout.addLayout(stream_layout)

        controls_layout = QtWidgets.QHBoxLayout()
        controls_layout.addStretch()

        self._start_button = QtWidgets.QPushButton("开始录制")
        self._pause_button = QtWidgets.QPushButton("暂停录制")
        self._stop_button = QtWidgets.QPushButton("结束录制")

        self._start_button.clicked.connect(self.start_recording)
        self._pause_button.clicked.connect(self.pause_recording)
        self._stop_button.clicked.connect(self.stop_recording)

        controls_layout.addWidget(self._start_button)
        controls_layout.addWidget(self._pause_button)
        controls_layout.addWidget(self._stop_button)

        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)

        self._status_label = QtWidgets.QLabel("状态：未录制")
        self._status_label.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(self._status_label)

        self._update_button_states()

    # ------------------------------------------------------------------
    # Recording management
    # ------------------------------------------------------------------
    def start_recording(self) -> None:
        """Start or resume a recording session."""

        if not self._session_active:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            recordings_root = Path.cwd() / "recordings"
            self._recording_dir = recordings_root / f"session_{timestamp}"
            self._recording_dir.mkdir(parents=True, exist_ok=True)

            csv_path = self._recording_dir / "timestamps.csv"
            self._csv_file = csv_path.open("w", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(["frame_index", "timestamp_ms", "color_path", "depth_path"])

            self._frame_index = 0
            self._session_active = True

        self._recording_enabled = True
        self._status_label.setText("状态：录制中")
        self._update_button_states()

    def pause_recording(self) -> None:
        """Temporarily pause recording without closing the current session."""

        if not self._session_active:
            return

        self._recording_enabled = False
        self._status_label.setText("状态：已暂停")
        self._update_button_states()

    def stop_recording(self) -> None:
        """Stop the current session and flush all files."""

        if not self._session_active:
            return

        self._recording_enabled = False
        self._session_active = False

        if self._csv_file is not None:
            self._csv_file.close()

        self._csv_file = None
        self._csv_writer = None
        self._recording_dir = None
        self._frame_index = 0
        self._status_label.setText("状态：未录制")
        self._update_button_states()

    def _update_button_states(self) -> None:
        """Enable or disable buttons based on the current state."""

        self._start_button.setEnabled(not self._recording_enabled)
        self._pause_button.setEnabled(self._session_active and self._recording_enabled)
        self._stop_button.setEnabled(self._session_active)

        if self._session_active and not self._recording_enabled:
            self._start_button.setText("继续录制")
        else:
            self._start_button.setText("开始录制")

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------
    def _update_frames(self) -> None:
        """Fetch frames from the camera, update the UI, and persist if needed."""

        frames = self._pipeline.poll_for_frames()
        if not frames:
            return

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return

        # Prepare color image for display (convert BGR -> RGB)
        color_image = np.asanyarray(color_frame.get_data())
        color_image_rgb = color_image[:, :, ::-1]

        color_qimage = QtGui.QImage(
            color_image_rgb.data,
            color_image_rgb.shape[1],
            color_image_rgb.shape[0],
            color_image_rgb.strides[0],
            QtGui.QImage.Format_RGB888,
        ).copy()

        # Colorize depth for visualization
        depth_color_frame = self._colorizer.colorize(depth_frame)
        depth_color_image = np.asanyarray(depth_color_frame.get_data())

        depth_qimage = QtGui.QImage(
            depth_color_image.data,
            depth_color_image.shape[1],
            depth_color_image.shape[0],
            depth_color_image.strides[0],
            QtGui.QImage.Format_RGB888,
        ).copy()

        self._color_label.setPixmap(QtGui.QPixmap.fromImage(color_qimage))
        self._depth_label.setPixmap(QtGui.QPixmap.fromImage(depth_qimage))

        if self._session_active and self._recording_enabled:
            self._save_frames(color_frame, depth_frame, color_image_rgb)

    def _save_frames(
        self,
        color_frame: rs.video_frame,
        depth_frame: rs.depth_frame,
        color_image_rgb: np.ndarray,
    ) -> None:
        """Persist the latest frames and their timestamps to disk."""

        assert self._recording_dir is not None
        assert self._csv_writer is not None

        frame_id = f"{self._frame_index:06d}"

        color_path = self._recording_dir / f"color_{frame_id}.png"
        depth_path = self._recording_dir / f"depth_{frame_id}.png"

        depth_data = np.asanyarray(depth_frame.get_data()).copy()

        # Save RGB as PNG (convert back to BGR for OpenCV)
        cv2.imwrite(str(color_path), cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(depth_path), depth_data)

        timestamp_ms = color_frame.get_timestamp()
        self._csv_writer.writerow([self._frame_index, f"{timestamp_ms:.3f}", color_path.name, depth_path.name])
        if self._csv_file is not None:
            self._csv_file.flush()
        self._frame_index += 1

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt signature)
        self.stop_recording()
        self._frame_timer.stop()
        self._pipeline.stop()
        super().closeEvent(event)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    recorder = RealSenseRecorder()
    recorder.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
