import logging
import os


# Function to create directory if it doesn't exist and handle permissions error
def create_log_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except PermissionError:
        print(f"Permission denied: cannot create log directory at {path}")
        return False
    return True


# Try to create the log directory in the current working directory
log_dir = "logs"
if not create_log_dir(log_dir):
    # Fall back to user's home directory if permission is denied
    log_dir = os.path.join(os.path.expanduser('~'), "logs")
    create_log_dir(log_dir)


# Configure the logger
def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup a logger with the specified name and log file."""

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
