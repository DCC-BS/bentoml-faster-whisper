from core import Segment as WhisperSegment
from diarization_service import DiarizationSegment
from typing import Iterable, List, Dict, Any, Optional
import numpy as np


def _find_best_speaker(
    diarize_list: List[Dict[str, Any]],
    start_time: float,
    end_time: float,
    window_start: int
) -> tuple[Optional[str], int]:
    """
    Find the best matching speaker for a time segment.
    
    Returns:
        Tuple of (best_speaker, new_window_start)
    """
    best_intersection = 0
    best_speaker = None
    
    for i in range(window_start, len(diarize_list)):
        d = diarize_list[i]
        if d["start"] > end_time:
            break  # No more potential overlaps
        
        # Calculate intersection
        intersection = min(d["end"], end_time) - max(d["start"], start_time)
        if intersection > best_intersection:
            best_intersection = intersection
            best_speaker = d["speaker"]
    
    # Update window to skip segments that end before current segment
    while window_start < len(diarize_list) and diarize_list[window_start]["end"] <= start_time:
        window_start += 1
        
    return best_speaker, window_start


def merge_whipser_diarization(
    whisper_segments: Iterable[WhisperSegment],
    diarization_segments: Iterable[DiarizationSegment],
) -> Iterable[WhisperSegment]:
    # Convert to lists and sort diarization segments by start time
    whisper_list = list(whisper_segments)
    diarize_list = sorted([vars(d) for d in diarization_segments], key=lambda x: x["start"])
    
    # Sliding window index for optimization
    window_start = 0
    
    for seg in whisper_list:
        # Find best speaker for segment
        best_speaker, window_start = _find_best_speaker(
            diarize_list, seg.start, seg.end, window_start
        )
        
        # Assign speaker
        if best_speaker:
            seg.speaker = best_speaker
        
        # Process words with the same approach
        if seg.words is not None:
            word_window = window_start  # Start from the same window position
            
            for word in seg.words:
                best_word_speaker, word_window = _find_best_speaker(
                    diarize_list, word.start, word.end, word_window
                )
                
                # Assign speaker
                if best_word_speaker:
                    word.speaker = best_word_speaker
    
    return whisper_list