import logging
import os
from utils.constants import LOGS_DIR

# Ensure the logs directory exists
if not os.path.exists(LOGS_DIR):
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
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%d-%m-%Y %H:%M:%S - %(name)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        stream_handler = PrintHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)


# Configure the logger
def setup_logger(name) -> SingletonLogger:
    """Function to setup a logger with the specified name and log file."""
    docker_image_name = os.getenv('DOCKER_IMAGE_NAME', 'default_image')
    job_id = os.getenv('JOB_ID', 'default_job')
    log_file_name = f"{docker_image_name}_{job_id}.log"
    log_file_path = os.path.join(LOGS_DIR, log_file_name)
    return SingletonLogger(name, log_file_path)
