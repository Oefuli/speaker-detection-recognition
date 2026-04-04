import logging
import sys

def setup_logger() -> None:
    """
    Configures the root logger for the entire application.
    Sets the log level and defines the output format for the console.
    """
    # Defines the format of the log messages
    # Example: 2026-04-04 19:38:25 - paths_files - ERROR - File doesn't exist.
    log_format = "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Configure the basic logging setup
    logging.basicConfig(
        level=logging.INFO, # Change to logging.DEBUG for more verbosity
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout) # Outputs logs to the console
            # To log into a file, you could add: logging.FileHandler("app.log")
        ]
    )