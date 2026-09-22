import queue
import threading
import time
from dataclasses import dataclass

import message_filters
import rclpy
from sensor_msgs.msg import Image
from spark_config import Config

from semantic_inference_ros.ros_conversions import Conversions


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
    """Sync images in a single callback worker."""

    def __init__(self, node, config, topics, callback, **kwargs) -> None:
        """Sync image messages constructor.
        :param node: The ROS 2 node instance used for
                     subscriptions and callbacks.
        :param config: SyncImagesWorkerConfig object containing
                       queue size and minimum separation time.
        :param topics: List of ROS 2 topic names to
                       synchronize and listen to.
        :param callback: Callback function to be invoked
                         when synchronized images are received.
        :param kwargs: Additional keyword arguments for
                       message filters synchronization."""

        self._node = node
        self._node.context.on_shutdown(self.stop)

        self._config = config
        self._callback = callback

        self._started = False
        self._should_shutdown = False
        self._last_stamp = None

        self._queue = queue.Queue(maxsize=config.queue_size)

        subs = [
            message_filters.Subscriber(self._node, Image, topic) for topic in topics
        ]
        self._sync = message_filters.TimeSynchronizer(
            subs, queue_size=self._config.queue_size
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

            curr_stamp = rclpy.time.Time.from_msg(data[0].header.stamp)
            if self._last_stamp is not None:
                diff_s = 1.0e-9 * (curr_stamp - self._last_stamp).nanoseconds
                if diff_s < self._config.min_separation_s:
                    continue

            self._last_stamp = curr_stamp

            imgs = [Conversions.to_image(msg) for msg in data]
            self._callback(data[0].header, *imgs)
