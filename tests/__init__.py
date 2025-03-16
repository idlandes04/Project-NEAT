"""
Test package for the neural architecture integration.
"""
import signal
import unittest
import threading
import time
import platform

# Global test timeout
DEFAULT_TEST_TIMEOUT = 30  # 30 seconds

class TimeoutError(Exception):
    """Exception raised when a test takes too long."""
    pass

def timeout_handler(signum, frame):
    """Handler for timeout signal."""
    raise TimeoutError("Test timed out")

# Check if we're running on Windows (WSL) where signal.SIGALRM doesn't work as expected
IS_WINDOWS = platform.system().startswith(('Windows', 'Microsoft'))

# Apply timeout to any test case
def add_timeout(test_case, timeout=DEFAULT_TEST_TIMEOUT):
    """
    Add timeout to a test case.
    
    Args:
        test_case: Test case to add timeout to
        timeout: Timeout in seconds
    
    Returns:
        Test case with timeout
    """
    old_setUp = test_case.setUp
    old_tearDown = test_case.tearDown
    
    def new_setUp(self):
        if IS_WINDOWS:
            # Use threading timeout for Windows/WSL
            self._timeout_timer = threading.Timer(timeout, lambda: _thread_timeout_handler(self))
            self._timeout_timer.daemon = True
            self._timeout_timer.start()
        else:
            # Use signal-based timeout for Unix systems
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
        
        # Call original setUp
        old_setUp(self)
        
    def new_tearDown(self):
        if IS_WINDOWS:
            # Cancel threading timeout
            if hasattr(self, '_timeout_timer'):
                self._timeout_timer.cancel()
        else:
            # Disable the alarm
            signal.alarm(0)
        
        # Call original tearDown
        old_tearDown(self)
    
    # Replace methods
    test_case.setUp = new_setUp
    test_case.tearDown = new_tearDown
    
    return test_case

def _thread_timeout_handler(test_instance):
    """Thread-based timeout handler for WSL compatibility."""
    # Get the main thread ID (where the test is running)
    main_thread = threading.main_thread()
    
    # Generate a more informative error message
    import inspect
    current_frame = inspect.currentframe()
    calling_frame = inspect.getouterframes(current_frame)[1]
    
    # Raise an informative exception
    test_name = getattr(test_instance, '_testMethodName', 'unknown')
    error_msg = f"Test timed out: {test_name} in {test_instance.__class__.__name__}"
    
    # Print the error directly as we might not be able to catch it properly
    print(f"\n\nTIMEOUT ERROR: {error_msg}\n\n")
    
    # Force the test to fail by raising an exception in the main thread
    # This is a bit of a hack, but it works for our purposes
    import os
    os._exit(1)  # Force exit the process
