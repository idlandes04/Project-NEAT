"""
Test package for the neural architecture integration.
"""
import signal
import unittest

# Global test timeout
DEFAULT_TEST_TIMEOUT = 30  # 30 seconds

class TimeoutError(Exception):
    """Exception raised when a test takes too long."""
    pass

def timeout_handler(signum, frame):
    """Handler for timeout signal."""
    raise TimeoutError("Test timed out")

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
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        old_setUp(self)
        
    def new_tearDown(self):
        signal.alarm(0)  # Disable the alarm
        old_tearDown(self)
        
    test_case.setUp = new_setUp
    test_case.tearDown = new_tearDown
    
    return test_case
