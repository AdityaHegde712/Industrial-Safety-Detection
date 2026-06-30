"""
Local File Input Handler
Handles both video files (MP4, AVI) and image directories (JPG, PNG).
"""
import cv2
import os
from typing import Tuple
import numpy as np
from .base import InputHandler

class LocalFileHandler(InputHandler):
    """
    Input handler for local video files or image directories.
    
    Args:
        path: Path to video file or image directory
        
    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If path is neither a video file nor an image directory
    """
    
    def __init__(self, path: str):
        """Initialize the handler with a video file or image directory."""
        self.path = path
        self.is_video = False
        self.is_image_dir = False
        self.cap = None  # Video capture object
        self.image_files = []  # List of image files
        self.image_index = 0  # Current index for image iteration
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
        
        # Check if it's a video file
        if os.path.isfile(path) and path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            self.is_video = True
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video file: {path}")
        
        # Check if it's an image directory
        elif os.path.isdir(path):
            self.is_image_dir = True
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            self.image_files = sorted([
                os.path.join(path, f) for f in os.listdir(path)
                if f.lower().endswith(valid_extensions)
            ])
            if not self.image_files:
                raise ValueError(f"No valid image files found in directory: {path}")
        
        else:
            raise ValueError(f"Path is neither a video file nor an image directory: {path}")
    
    def get_frame(self) -> Tuple[np.ndarray, float]:
        """
        Get the next frame from the input source.
        
        Returns:
            Tuple containing:
                - frame: numpy array (image frame)
                - timestamp: float (timestamp in seconds, or 0.0 for images)
                
        Raises:
            StopIteration: When no more frames are available
        """
        if self.is_video:
            ret, frame = self.cap.read()
            if not ret:
                raise StopIteration("No more frames in video")
            timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            return frame, timestamp
        
        elif self.is_image_dir:
            if self.image_index >= len(self.image_files):
                raise StopIteration("No more images in directory")
            
            img_path = self.image_files[self.image_index]
            frame = cv2.imread(img_path)
            if frame is None:
                self.image_index += 1
                return self.get_frame()  # Skip invalid images
            
            timestamp = 0.0  # Images don't have timestamps
            self.image_index += 1
            return frame, timestamp
        
        else:
            raise StopIteration("Input source not properly initialized")
    
    def release(self) -> None:
        """Release resources (close video file)."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
