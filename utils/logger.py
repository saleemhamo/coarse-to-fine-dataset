import logging
import os
import re
from utils.constants import LOGS_DIR

# Ensure the logs directory exists
if not os.path.exists(LOGS_DIR):
    print(f"Creating log directory: {LOGS_DIR}")
    os.makedirs(LOGS_DIR)


class PrintHandler(logging.StreamHandler):
    def emit(self, record):
        print(self.format(record))


class SingletonLogger:
    _instance = None

    def __new__(cls, name, log_file):
        if cls._instance is None:
            cls._instance = super(SingletonLogger, cls).__new__(cls)
            cls._instance._initialize(name, log_file)
        return cls._instance

    def _initialize(self, name, log_file):
        print(f"Initializing logger: {name}, log_file: {log_file}")
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Use asctime for preformatted date and time string
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Ensure the directory for the log file exists
        log_dir = os.path.dirname(log_file)
        if not os.path.exists(log_dir):
            print(f"Creating log file directory: {log_dir}")
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        print(f"File handler added: {log_file}")

        stream_handler = PrintHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
        print(f"Stream handler added")

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)


# Configure the logger
def setup_logger(name) -> SingletonLogger:
    """Function to setup a logger with the specified name and log file."""
    docker_image_name = os.getenv('DOCKER_IMAGE_NAME', 'default_image')
    job_id = os.getenv('JOB_ID', 'default_job')

    # Replace invalid characters in the log file name
    job_id = re.sub(r'[^a-zA-Z0-9]', '_', job_id)
    docker_image_name = re.sub(r'[^a-zA-Z0-9]', '_', docker_image_name)

    log_file_name = f"{docker_image_name}_{job_id}.log"
    log_file_path = os.path.join(LOGS_DIR, log_file_name)
    print(f"In setup_logger {name}: log_file_path={log_file_path}")
    return SingletonLogger(name, log_file_path)
