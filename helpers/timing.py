import asyncio
import functools
import time
from typing import Callable

from helpers.utils import (
    get_audio_duration,
    input_audio_length_histogram,
    realtime_factor_histogram,
)


def measure_processing_time(func: Callable) -> Callable:
    """
    Decorator that measures audio processing time and records metrics to Prometheus.
    It calculates the real-time factor and tracks audio length.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()

        # Get the request object from either args or kwargs
        audio_file = kwargs.get("file")

        result = func(self, *args, **kwargs)

        if audio_file:
            end_time = time.time()
            duration = end_time - start_time
            audio_duration = get_audio_duration(audio_file)
            input_audio_length_histogram.observe(audio_duration)
            realtime_factor_histogram.observe(audio_duration / duration)

        return result

    # Handle async functions
    @functools.wraps(func)
    async def async_wrapper(self, *args, **kwargs):
        start_time = time.time()

        # Similar logic to extract request/file
        request = kwargs.get("request")
        if not request:
            for key, value in kwargs.items():
                if isinstance(value, dict) and "file" in value:
                    audio_file = value["file"]
                    break
            else:
                audio_file = kwargs.get("file")
        else:
            audio_file = getattr(request, "file", None)

        # For **params unpacking
        if not audio_file and len(kwargs) == 1 and list(kwargs.keys())[0] == "params":
            params = kwargs["params"]
            audio_file = params.get("file")

        result = await func(self, *args, **kwargs)

        if audio_file:
            end_time = time.time()
            duration = end_time - start_time
            audio_duration = get_audio_duration(audio_file)
            input_audio_length_histogram.observe(audio_duration)
            realtime_factor_histogram.observe(audio_duration / duration)

        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return wrapper
