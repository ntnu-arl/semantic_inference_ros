"""Module containing queue-based image processor."""

from semantic_inference_python import Config
from semantic_inference_ros.ros_conversions import Conversions
from dataclasses import dataclass

import rospy
import message_filters
import queue
import threading
import time


@dataclass
class SyncImagesWorkerConfig(Config):
    """Configuration for image worker."""

    queue_size: int = 1
    min_separation_s: float = 0.0

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


class SyncImagesWorker:
    """Class to simplify message processing."""

    def __init__(self, config, topics, types, callback, **kwargs):
        """Register worker with ros."""
        self._config = config
        self._callback = callback

        self._started = False
        self._should_shutdown = False
        self._last_stamp = None

        self._queue = queue.Queue(maxsize=config.queue_size)

        rospy.on_shutdown(self.stop)
        subs = [
            message_filters.Subscriber(topic, type_)
            for topic, type_ in zip(topics, types)
        ]

        self._sync = message_filters.TimeSynchronizer(
            subs, queue_size=config.queue_size
        )
        self._sync.registerCallback(self.add_message)
        self.start()

    def add_message(self, *msgs):
        """Add new message to queue."""
        data = [msg for msg in msgs]
        if not self._queue.full():
            self._queue.put(data, block=False, timeout=False)

    def start(self):
        """Start worker processing queue."""
        if not self._started:
            self._started = True
            self._thread = threading.Thread(target=self._do_work)
            self._thread.start()

    def stop(self):
        """Stop worker from processing queue."""
        if self._started:
            self._should_shutdown = True
            self._thread.join()

        self._started = False
        self._should_shutdown = False

    def spin(self):
        """Wait for ros to shutdown or worker to exit."""
        if not self._started:
            return

        while self._thread.is_alive() and not self._should_shutdown:
            time.sleep(1.0e-2)

        self.stop()

    def _do_work(self):
        while not self._should_shutdown:
            try:
                data = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._last_stamp is not None:
                diff_s = (data[0].header.stamp - self._last_stamp).to_sec()
                if diff_s < self._config.min_separation_s:
                    continue

            self._last_stamp = data[0].header.stamp

            # try:
            imgs = [Conversions.to_image(msg) for msg in data]
            self._callback(data[0].header, *imgs)
            # except Exception as e:
            # rospy.logerr(f"spin failed: {e}")
