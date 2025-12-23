"""Performance timing utilities for profiling function execution."""

import time
from functools import wraps


def timed(func):
    """Decorator that measures and prints the execution time of a function.

    Args:
        func: The function to be timed.

    Returns:
        A wrapped function that prints execution time after completion.

    Example:
        @timed
        def slow_function():
            time.sleep(1)

        slow_function()  # Prints: slow_function took 1.000123s
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.6f}s")
        return result
    return wrapper