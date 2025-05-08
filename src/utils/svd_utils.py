# --- START OF FILE src/utils/svd_utils.py ---

"""
Utilities for Singular Value Decomposition (SVD).

Provides functions for efficient SVD computation (full and randomized)
and optional memory/disk caching. Includes matrix reconstruction and
cache clearing utilities.
"""

import torch
import logging # Added import
import os
import pickle
import hashlib
import time
import threading # Added import
from typing import Tuple, Optional, Dict, Any

# Optional imports for efficient SVD
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # logger.debug("Numpy not found, randomized SVD via sklearn will be disabled.") # Logger not defined yet


try:
    # Check for scikit-learn's randomized_svd
    from sklearn.utils.extmath import randomized_svd
    SKLEARN_AVAILABLE = NUMPY_AVAILABLE # Requires numpy
    # if not SKLEARN_AVAILABLE: # Logger not defined yet
        # logger.debug("Scikit-learn not found or numpy missing, randomized SVD disabled.")

except ImportError:
    SKLEARN_AVAILABLE = False
    # logger.debug("Scikit-learn not found, randomized SVD disabled.") # Logger not defined yet


logger = logging.getLogger(__name__) # Added logger definition

# --- Configuration Defaults ---
DEFAULT_SVD_N_OVERSAMPLES = 10
DEFAULT_SVD_N_ITER = 5
DEFAULT_CACHE_DIR = ".svd_cache"
DEFAULT_MAX_MEMORY_CACHE = 100 # Max number of SVD results to keep in memory

# --- Caching ---
# In-memory cache (module-level, shared across calls)
_memory_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_cache_lock = threading.Lock() # Basic lock for memory cache access


def _get_matrix_hash(matrix: torch.Tensor) -> str:
    """
    Generates a reasonably stable hash for a matrix based on shape and content sample.

    Args:
        matrix: The input torch.Tensor.

    Returns:
        A hexadecimal string representing the hash.
    """
    hasher = hashlib.sha256()
    # Include shape and dtype in hash
    hasher.update(str(matrix.shape).encode())
    hasher.update(str(matrix.dtype).encode())

    # Hash a subset of the data for efficiency, handle different tensor sizes
    numel = matrix.numel()
    if numel == 0:
        # Handle empty tensor case
        sample = b''
    elif numel < 1000:
        # Hash the entire tensor if small
        sample_indices = torch.arange(numel, dtype=torch.long)
        sample = matrix.flatten()[sample_indices].cpu().numpy().tobytes()
    else:
        # Hash a fixed number of samples for larger tensors
        sample_indices = torch.linspace(0, numel - 1, steps=1000, dtype=torch.long)
        sample = matrix.flatten()[sample_indices].cpu().numpy().tobytes()

    hasher.update(sample)
    return hasher.hexdigest()[:16] # Truncate hash for brevity


def compute_efficient_svd(
    matrix: torch.Tensor,
    k: int,
    use_randomized: bool = True,
    n_oversamples: int = DEFAULT_SVD_N_OVERSAMPLES,
    n_iter: int = DEFAULT_SVD_N_ITER,
    enable_cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
    force_compute: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the Singular Value Decomposition (SVD) of a matrix efficiently.

    Uses randomized SVD for potentially large speedups on suitable matrices if
    numpy and scikit-learn are available and `use_randomized` is True.
    Otherwise, falls back to `torch.linalg.svd`. Includes optional disk
    and memory caching for computed results.

    Args:
        matrix: The input matrix [M, N] to decompose.
        k: The number of singular values/vectors to compute and keep.
        use_randomized: Whether to attempt using randomized SVD.
        n_oversamples: Oversampling parameter for randomized SVD.
        n_iter: Number of power iterations for randomized SVD.
        enable_cache: Whether to use memory and disk caching.
        cache_dir: The directory for disk caching. Will be created if it doesn't exist.
        force_compute: If True, bypasses cache lookup and forces recomputation.
                       The result will still be cached.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: U, S, Vh tensors, where
            U is [M, k_actual], S is [k_actual], Vh is [k_actual, N].
            k_actual = min(k, matrix.shape[0], matrix.shape[1]).
            Returns empty tensors with appropriate dimension hints if k_actual <= 0
            or if computation fails.
    """
    global _memory_cache
    device = matrix.device
    original_dtype = matrix.dtype
    m, n = matrix.shape
    k_actual = min(k, m, n) # Effective number of components

    if k_actual <= 0:
         logger.warning(f"Requested k={k} results in k_actual={k_actual} <= 0 for matrix shape {matrix.shape}. Returning empty tensors.")
         return torch.empty((m, 0), device=device, dtype=original_dtype), \
                torch.empty((0,), device=device, dtype=original_dtype), \
                torch.empty((0, n), device=device, dtype=original_dtype)

    matrix_hash = _get_matrix_hash(matrix)
    cache_key = f"svd_hash_{matrix_hash}_k_{k_actual}"

    # 1. Check Cache (if enabled and not forced)
    if enable_cache and not force_compute:
        # Check memory cache (thread-safe read)
        with _cache_lock:
            cached_result = _memory_cache.get(cache_key)
        if cached_result is not None:
            U_cpu, S_cpu, Vh_cpu = cached_result
            # logger.debug(f"SVD cache hit (memory) for key: {cache_key}")
            return U_cpu.to(device=device, dtype=original_dtype), \
                   S_cpu.to(device=device, dtype=original_dtype), \
                   Vh_cpu.to(device=device, dtype=original_dtype)

        # Check disk cache
        cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    U_cpu, S_cpu, Vh_cpu = pickle.load(f)
                # logger.debug(f"SVD cache hit (disk) for key: {cache_key}")
                # Add to memory cache (thread-safe write)
                with _cache_lock:
                    if len(_memory_cache) >= DEFAULT_MAX_MEMORY_CACHE:
                         # Simple LRU: remove the first item added (oldest)
                         try:
                              _memory_cache.pop(next(iter(_memory_cache)))
                         except StopIteration: # Should not happen if cache size > 0
                              pass
                    _memory_cache[cache_key] = (U_cpu, S_cpu, Vh_cpu)
                return U_cpu.to(device=device, dtype=original_dtype), \
                       S_cpu.to(device=device, dtype=original_dtype), \
                       Vh_cpu.to(device=device, dtype=original_dtype)
            except (pickle.UnpicklingError, EOFError, TypeError, Exception) as e:
                logger.warning(f"Failed to load or parse SVD cache file {cache_file}: {e}. Recomputing.")
                try:
                    os.remove(cache_file) # Remove corrupted cache file
                except OSError:
                    pass

    # 2. Compute SVD
    logger.debug(f"SVD cache miss for key: {cache_key}. Computing SVD (k_actual={k_actual})...")
    start_time = time.time()

    # Determine if randomized SVD should be used based on heuristics
    should_use_randomized = (
        use_randomized
        and SKLEARN_AVAILABLE # Checks if numpy and sklearn are imported
        and m > k_actual * 1.5 and n > k_actual * 1.5 # More efficient when k << min(M, N)
        and k_actual < min(m, n) * 0.8 # Avoid if k is close to full rank
    )

    try:
        if should_use_randomized:
            logger.debug(f"Attempting randomized SVD for shape {matrix.shape}, k={k_actual}")
            # Ensure matrix is float32 or float64 for numpy/sklearn
            compute_dtype = torch.float32 if original_dtype not in [torch.float64] else torch.float64
            matrix_compute = matrix.to(compute_dtype)

            weight_np = matrix_compute.detach().cpu().numpy()
            U_np, S_np, Vh_np = randomized_svd(
                weight_np,
                n_components=k_actual,
                n_oversamples=n_oversamples,
                n_iter=n_iter,
                random_state=None # Use default numpy random state
            )
            # Convert back to torch tensors on the correct device and original dtype
            U = torch.from_numpy(U_np.copy()).to(device=device, dtype=original_dtype)
            S = torch.from_numpy(S_np.copy()).to(device=device, dtype=original_dtype)
            Vh = torch.from_numpy(Vh_np.copy()).to(device=device, dtype=original_dtype)
            svd_method = "randomized"
        else:
            logger.debug(f"Using full SVD (torch.linalg.svd) for shape {matrix.shape}, k={k_actual}")
            # Use torch.linalg.svd
            U_full, S_full, Vh_full = torch.linalg.svd(matrix, full_matrices=False)
            # Truncate results to k_actual
            U = U_full[:, :k_actual]
            S = S_full[:k_actual]
            Vh = Vh_full[:k_actual, :]
            svd_method = "full"

        end_time = time.time()
        logger.debug(f"SVD computation ({svd_method}) took {end_time - start_time:.4f} seconds.")

    except Exception as e:
        logger.error(f"SVD computation failed for matrix shape {matrix.shape}, k={k_actual}: {e}. Returning empty tensors.", exc_info=True)
        return torch.empty((m, 0), device=device, dtype=original_dtype), \
               torch.empty((0,), device=device, dtype=original_dtype), \
               torch.empty((0, n), device=device, dtype=original_dtype)

    # 3. Cache results (if enabled)
    if enable_cache:
        # Store CPU tensors in cache for portability
        U_cpu, S_cpu, Vh_cpu = U.detach().cpu(), S.detach().cpu(), Vh.detach().cpu()

        # Add to memory cache (thread-safe write)
        with _cache_lock:
            if len(_memory_cache) >= DEFAULT_MAX_MEMORY_CACHE:
                 try:
                      _memory_cache.pop(next(iter(_memory_cache))) # Remove oldest
                 except StopIteration: pass
            _memory_cache[cache_key] = (U_cpu, S_cpu, Vh_cpu)

        # Save to disk cache
        try:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True) # Ensure directory exists
            cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump((U_cpu, S_cpu, Vh_cpu), f)
            # logger.debug(f"Saved SVD result to disk cache: {cache_file}")
        except (OSError, pickle.PicklingError, Exception) as e:
            logger.warning(f"Failed to save SVD cache file {cache_file}: {e}")

    return U, S, Vh


def reconstruct_from_svd(U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor) -> torch.Tensor:
    """
    Reconstructs a matrix from its SVD components (U, S, Vh).

    Args:
        U: Left singular vectors [M, k].
        S: Singular values [k].
        Vh: Right singular vectors [k, N].

    Returns:
        Reconstructed matrix [M, N].
    """
    if S.numel() == 0: # Handle empty SVD case
         m = U.shape[0] if U.ndim == 2 else 0
         n = Vh.shape[1] if Vh.ndim == 2 else 0
         return torch.zeros((m,n), device=U.device, dtype=U.dtype)

    # Ensure S is diagonal
    # Use torch.diag_embed for batch compatibility if needed, but here S is 1D
    diag_S = torch.diag(S)

    # Reconstruct: M = U @ diag(S) @ Vh
    try:
        reconstructed = torch.matmul(U, torch.matmul(diag_S, Vh))
        return reconstructed
    except RuntimeError as e:
         logger.error(f"Error during SVD reconstruction: {e}. Shapes: U={U.shape}, S={S.shape}, Vh={Vh.shape}", exc_info=True)
         # Return zero matrix of expected shape as fallback
         m = U.shape[0]
         n = Vh.shape[1]
         return torch.zeros((m,n), device=U.device, dtype=U.dtype)


def clear_svd_cache(memory: bool = True, disk: bool = False, cache_dir: str = DEFAULT_CACHE_DIR):
    """
    Clears the SVD cache(s).

    Args:
        memory: If True, clears the in-memory cache.
        disk: If True, attempts to remove the disk cache directory.
        cache_dir: The directory path of the disk cache.
    """
    global _memory_cache
    cleared_memory = False
    cleared_disk = False

    if memory:
        with _cache_lock:
            _memory_cache.clear()
        cleared_memory = True
        logger.info("Cleared in-memory SVD cache.")

    if disk:
        if os.path.exists(cache_dir):
            try:
                import shutil
                shutil.rmtree(cache_dir)
                # Optionally recreate the directory immediately
                # os.makedirs(cache_dir, exist_ok=True)
                cleared_disk = True
                logger.info(f"Removed disk SVD cache directory: {cache_dir}")
            except (OSError, Exception) as e:
                logger.error(f"Failed to remove disk SVD cache directory {cache_dir}: {e}")
        else:
            logger.info(f"Disk SVD cache directory not found (already clear or never created): {cache_dir}")

    return cleared_memory, cleared_disk

# --- Example Usage ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')

    print("Testing SVD Utilities...")
    # Create a sample matrix
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    matrix = torch.randn(100, 50, device=device, dtype=torch.float32) * 10
    k = 10

    # --- Test Computation & Caching ---
    print(f"\nComputing SVD for matrix shape {matrix.shape} with k={k}...")
    clear_svd_cache(memory=True, disk=True) # Clear cache before first run

    start = time.time()
    U, S, Vh = compute_efficient_svd(matrix, k, enable_cache=True, use_randomized=True)
    end = time.time()
    print(f"First SVD computation took {end - start:.4f}s")
    print(f"Result shapes: U={U.shape}, S={S.shape}, Vh={Vh.shape}")

    start = time.time()
    U_cached, S_cached, Vh_cached = compute_efficient_svd(matrix, k, enable_cache=True, use_randomized=True)
    end = time.time()
    print(f"Second SVD computation (should hit cache) took {end - start:.6f}s")
    assert torch.allclose(S, S_cached), "Cached S differs from computed S"

    # --- Test Reconstruction ---
    print("\nTesting Reconstruction...")
    reconstructed_matrix = reconstruct_from_svd(U, S, Vh)
    print(f"Reconstructed matrix shape: {reconstructed_matrix.shape}")
    # Compare reconstruction error to original matrix (for low rank approximation)
    original_reconstruction_full = reconstruct_from_svd(*torch.linalg.svd(matrix, full_matrices=False))
    low_rank_error = torch.norm(matrix - reconstructed_matrix)
    full_rank_error = torch.norm(matrix - original_reconstruction_full) # Should be near zero
    print(f"Low rank (k={k}) reconstruction error (Frobenius norm): {low_rank_error:.4f}")
    print(f"Full rank reconstruction error (Frobenius norm): {full_rank_error:.4e}") # Expect very small

    # --- Test Cache Clearing ---
    print("\nTesting Cache Clearing...")
    clear_svd_cache(memory=True, disk=False)
    with _cache_lock:
        assert len(_memory_cache) == 0, "Memory cache not cleared"
    print("Memory cache cleared.")
    # Recompute to check disk cache was potentially created
    _ = compute_efficient_svd(matrix, k, enable_cache=True)
    if os.path.exists(DEFAULT_CACHE_DIR) and len(os.listdir(DEFAULT_CACHE_DIR)) > 0:
        print(f"Disk cache directory '{DEFAULT_CACHE_DIR}' exists and is not empty.")
        clear_svd_cache(memory=False, disk=True)
        assert not os.path.exists(DEFAULT_CACHE_DIR) or len(os.listdir(DEFAULT_CACHE_DIR)) == 0, "Disk cache not cleared"
        print("Disk cache cleared.")
    else:
        print("Disk cache directory not found or empty, skipping disk clear test.")


# --- END OF FILE src/utils/svd_utils.py ---