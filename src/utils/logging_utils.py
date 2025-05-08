# --- START OF FILE src/utils/logging_utils.py ---

"""
Logging utilities for Project NEAT.

Provides a standardized way to configure logging across the project,
including console and rotating file output.
"""

import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

# --- Configuration ---
LOG_FORMAT = "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_LEVEL = logging.INFO
CONSOLE_LOG_LEVEL = logging.INFO # Default level for console output
FILE_LOG_LEVEL = logging.DEBUG   # Default level for file output (more verbose)
LOG_FILE_NAME = "project_neat.log"
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024 # 10 MB
LOG_FILE_BACKUP_COUNT = 5

# Flag to prevent multiple setups
_logging_configured = False

def setup_logging(
    log_dir: str = "output/logs",
    log_level: int = DEFAULT_LOG_LEVEL,
    console_level: Optional[int] = None,
    file_level: Optional[int] = None,
    log_filename: str = LOG_FILE_NAME
) -> None:
    """
    Configures logging for the entire application.

    Sets up logging to both the console (stdout) and a rotating file.
    Ensures handlers are not added multiple times if called repeatedly.

    Args:
        log_dir (str): The directory to store log files. Will be created if it doesn't exist.
        log_level (int): The base logging level for the root logger. Messages below this
                         level will be ignored entirely. Defaults to logging.INFO.
        console_level (Optional[int]): Specific level for console output. If None, uses
                                       CONSOLE_LOG_LEVEL constant.
        file_level (Optional[int]): Specific level for file output. If None, uses
                                    FILE_LOG_LEVEL constant.
        log_filename (str): The name for the log file within the log_dir.
    """
    global _logging_configured
    if _logging_configured:
        # Optional: Could log a debug message here if needed
        # logging.getLogger(__name__).debug("Logging setup already called. Skipping reconfiguration.")
        return

    # Determine effective levels, using constants as defaults
    eff_console_level = console_level if console_level is not None else CONSOLE_LOG_LEVEL
    eff_file_level = file_level if file_level is not None else FILE_LOG_LEVEL

    # --- Root Logger Configuration ---
    # Get the root logger instance
    root_logger = logging.getLogger()

    # Set the root logger level to the lowest (most verbose) of the requested levels.
    # This ensures messages intended for any handler are not filtered out prematurely.
    effective_root_level = min(eff_console_level, eff_file_level, log_level)
    root_logger.setLevel(effective_root_level)

    # Remove existing handlers attached *directly* to the root logger
    # to prevent duplicates if this function were somehow called again
    # without the guard flag working (e.g., in multithreading without locks).
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close() # Close handler to release resources

    # --- Formatter ---
    # Create a shared formatter for consistency
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # --- Console Handler (stdout) ---
    try:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(eff_console_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    except Exception as e:
        # Fallback basic config if stream handler fails (highly unlikely)
        logging.basicConfig(level=eff_console_level, format=LOG_FORMAT, datefmt=DATE_FORMAT)
        logging.error(f"Failed to set up console handler: {e}. Using basicConfig fallback.", exc_info=True)


    # --- Rotating File Handler ---
    try:
        # Ensure log directory exists; create if necessary
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, log_filename)

        # Create the rotating file handler
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=LOG_FILE_MAX_BYTES,
            backupCount=LOG_FILE_BACKUP_COUNT,
            encoding='utf-8' # Use UTF-8 for broad compatibility
        )
        file_handler.setLevel(eff_file_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Log confirmation message *after* handlers are set up
        logging.getLogger(__name__).info(
            f"Logging configured. Root Level: {logging.getLevelName(effective_root_level)}, "
            f"Console Level: {logging.getLevelName(eff_console_level)}, "
            f"File Level: {logging.getLevelName(eff_file_level)}, "
            f"File Path: {log_file_path}"
        )

    except Exception as e:
        # Log error to console if file handler setup fails
        logging.error(f"Failed to set up file logging to {log_dir}/{log_filename}: {e}", exc_info=True)
        # Execution continues with console logging enabled


    # Set the flag to indicate configuration is complete
    _logging_configured = True

# --- Example Usage ---
if __name__ == "__main__":
    print("Setting up logging example (defaults)...")
    # Example: Setup logging with default settings, storing logs in './output/logs'
    # (relative to where the script is run)
    setup_logging()

    # Get loggers for different hypothetical modules/parts of the application
    logger_main = logging.getLogger("project_neat.main")
    logger_data = logging.getLogger("project_neat.data")
    logger_model_core = logging.getLogger("project_neat.model.core")
    logger_model_memory = logging.getLogger("project_neat.model.memory")

    # Log messages at different levels to demonstrate filtering
    logger_main.debug("This is a main debug message (should only go to file by default).")
    logger_main.info("Starting main application process.")
    logger_data.info("Processing data batch #1...")
    logger_model_core.debug("Performing attention calculation.")
    logger_model_memory.info("Memory query executed.")
    logger_model_core.warning("Potential numerical instability detected in layer 5.")
    logger_data.error("Failed to load data file 'nonexistent.txt'.")
    logger_main.critical("Critical system component failed!")

    print(f"\nLog messages written. Check console output and the log file in './output/logs/{LOG_FILE_NAME}'.")
    print(f"File logging level is DEBUG, Console logging level is INFO by default.")

    # Demonstrate that calling setup again has no effect due to the guard flag
    print("\nAttempting to reconfigure logging (should have no effect)...")
    initial_handlers = list(logging.getLogger().handlers)
    setup_logging(log_level=logging.WARNING) # Try changing level
    final_handlers = list(logging.getLogger().handlers)
    if len(initial_handlers) == len(final_handlers) and all(h1 is h2 for h1, h2 in zip(initial_handlers, final_handlers)):
        print("Confirmed: setup_logging did not add duplicate handlers.")
    else:
        print("Warning: Handler list changed after second setup_logging call.")

    logger_main.info("This INFO message should still appear on console (handler level unchanged).")
    logger_main.debug("This DEBUG message should still only appear in file (handler level unchanged).")

# --- END OF FILE src/utils/logging_utils.py ---