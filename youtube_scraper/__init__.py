"""
YouTube Dataset Scraper for Swin-VALLR
Build lip-reading datasets from trainable YouTube videos.
"""

__version__ = "1.0.0"

from .youtube_video_finder import YouTubeVideoFinder
from .youtube_trainability_checker import YouTubeTrainabilityChecker
from .youtube_dataset_builder import YouTubeDatasetBuilder

__all__ = [
    'YouTubeVideoFinder',
    'YouTubeTrainabilityChecker',
    'YouTubeDatasetBuilder',
]
