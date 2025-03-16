# Platform Compatibility Notes

## CUDA Streams and Priority Settings

When working with PyTorch CUDA streams across different platforms, there are important differences to consider:

1. **Stream Priority API Differences**:
   - In Linux and macOS, some implementations may use `torch.cuda.Stream.Priority.HIGH/NORMAL`.
   - In Windows/WSL environments, use numeric values directly: `-1` (high priority) and `0` (normal priority).
   - This avoids the `AttributeError: type object 'Stream' has no attribute 'Priority'` error.

2. **Recommended Implementation**:
   ```python
   # Platform-independent approach for CUDA stream priorities
   priority = -1 if high_priority else 0  # -1 is higher priority, 0 is normal
   stream = torch.cuda.Stream(priority=priority)
   ```

3. **Comparing Stream Priorities**:
   ```python
   # Instead of:
   # priority == torch.cuda.Stream.Priority.HIGH
   
   # Use:
   priority < 0  # Negative values are higher priority
   ```

This approach ensures consistent behavior across all platforms including Windows, Linux and macOS.