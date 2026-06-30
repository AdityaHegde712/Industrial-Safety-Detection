"""
Input Handler Abstract Base Class
Defines the interface for all input handlers (local files, RTSP streams, etc.)
"""
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

class InputHandler(ABC):
    """
    Abstract base class for input handlers.
    
    All input handlers must implement get_frame() and release() methods.
    This allows plug-and-play swapping of input sources (local files <-> RTSP streams).
    """
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def release(self) -> None:
        """
        Release resources (close video file, release camera, etc.)
        """
        pass
