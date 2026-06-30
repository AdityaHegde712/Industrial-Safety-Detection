"""
RTSP Stream Input Handler (STUBBED)
This handler is stubbed for future phase implementation.
Method signatures match LocalFileHandler for plug-and-play replacement.
"""
from typing import Tuple
import numpy as np
from .base import InputHandler

class RTSPHandler(InputHandler):
    """
    Input handler for RTSP video streams.
    
    STATUS: STUBBED - Coming in future phase
    
    This class provides the same interface as LocalFileHandler
    for plug-and-play replacement when RTSP support is added.
    """
    
    def __init__(self, path: str):
        """
        Initialize RTSP stream handler.
        
        Args:
            path: RTSP URL (e.g., rtsp://camera-ip:554/stream)
            
        Raises:
            NotImplementedError: Always, as this is stubbed for future implementation
        """
        raise NotImplementedError(
            "RTSP stream support coming in future phase. "
            "Please use LocalFileHandler for current sprint."
        )
    
    def get_frame(self) -> Tuple[np.ndarray, float]:
        """
        Get the next frame from RTSP stream.
        
        Raises:
            NotImplementedError: Always, as this is stubbed for future implementation
        """
        raise NotImplementedError(
            "RTSP stream support coming in future phase. "
            "Please use LocalFileHandler for current sprint."
        )
    
    def release(self) -> None:
        """
        Release RTSP stream resources.
        
        Raises:
            NotImplementedError: Always, as this is stubbed for future implementation
        """
        raise NotImplementedError(
            "RTSP stream support coming in future phase. "
            "Please use LocalFileHandler for current sprint."
        )
