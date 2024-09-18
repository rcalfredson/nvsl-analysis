# standard libraries
from typing import Protocol

# third-party libraries
import cv2
import numpy as np


class VideoAnalysisInterface(Protocol):
    """
    An interface for the VideoAnalysis class, abstracting the functionalities for video analysis to
    avoid direct dependencies and circular imports. It specifies the required attributes and methods
    for implementing video analysis functionalities, including frame processing, fly tracking, handling
    video captures, and managing analysis data related to trainings.

    Attributes:
        frame (np.ndarray): The current video frame being processed.
        flies (list): Identifiers or information for flies detected within the video.
        trns (list[Training]): A list of instances representing the trainings of the experiment.
        trx (list[Trajectory]): Tracking data for flies detected in the video.
        fn (str): Filename or path of the video file under analysis.
        cap (cv2.VideoCapture): Video capture object for accessing video frames.
    """

    frame: np.ndarray
    flies: list
    trns: list
    trx: list
    fn: str
    cap: cv2.VideoCapture

    def _syncBucket(self, trn, df: float, skip: int):
        """
        Defines time divisions (buckets) within a training session based on the timing of the first reward event.
        This method synchronizes the buckets for the analysis by identifying the frame index of the first reward
        and optionally adjusting it by skipping a specified number of frames. It's used to segment the training
        session into analyzable parts, facilitating focused analysis around reward events.

        The method calculates the number of buckets by dividing the training session's length by the specified
        frame difference (df), adjusting for the frame index of the first reward event.

        Args:
            trn: The training session to be analyzed, containing details about reward events among other data.
            df (float, optional): The frame difference used to divide the training session into buckets.
                                Represents the length of each bucket in frames. Defaults to np.nan, indicating
                                an undefined default length and requiring specific handling.
            skip (int, optional): The number of frames to skip after the first reward event to define the start
                                of the first bucket. This adjustment allows for more precise analysis by offsetting
                                the bucket start relative to significant events. Defaults to 1.

        Returns:
            tuple: A tuple containing the following elements:
                - fi (int or None): The frame index for the start of the first bucket after adjusting for `skip`.
                                    `None` if no reward events are found.
                - n (float): The calculated number of buckets based on the session length and frame difference.
                - on (list): A list of frame indices for each reward event, used to synchronize the analysis.
        """
        pass

    def _min2f(self, m: float):
        """
        Converts minutes to frames based on the video's frame rate, facilitating time-based analysis
        and synchronization within the video analysis process.

        Args:
            m (float): Time in minutes to be converted into frames.

        Returns:
            int: The equivalent number of frames.
        """
        pass

    def _f2ms(self, m: float):
        """
        Converts frames to milliseconds, enabling precise timing operations and analyses
        based on the video's frame rate.

        Args:
            m (float): The number of frames to be converted.

        Returns:
            float: The equivalent duration in milliseconds.
        """
        pass
