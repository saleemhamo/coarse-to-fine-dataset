import time


class Logger:

    def __init__(self, name):
        self.name = name

    def info(self, message):
        print(f"{self.name} - [INFO] - {time.time()} - {message}")


# Configure the logger
def setup_logger(name, log_file) -> Logger:
    """Function to setup a logger with the specified name and log file."""
    logger = Logger(name)

    return logger
