import logging
import os
import time


# Create a directory for logs if it doesn't exist
# log_dir = "logs"
# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)


class Logger:

    def __init__(self, name):
        self.name = name
        self.level = logging.INFO

        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # file_handler = logging.FileHandler(os.path.join(log_dir, 'log.log'))
        # file_handler.setFormatter(formatter)
        # self.logger.addHandler(file_handler)

    def info(self, message):
        print(f"{self.name} - [INFO] - {time.time()} - {message}")


# Configure the logger
def setup_logger(name, log_file, level=logging.INFO) -> Logger:
    """Function to setup a logger with the specified name and log file."""
    logger = Logger(name)
    # class PrintHandler(logging.StreamHandler):
    #     def emit(self, record):
    #         print(self.format(record))

    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #
    # file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
    # file_handler.setFormatter(formatter)
    #
    # stream_handler = PrintHandler()
    # stream_handler.setFormatter(formatter)
    #
    # logger = logging.getLogger(name)
    # logger.setLevel(level)
    # logger.addHandler(file_handler)
    # logger.addHandler(stream_handler)

    return logger
