import platform
import sys
import subprocess
import torch

from utils.logger import setup_logger

logger = setup_logger('system_env_logger')


def get_python_info():
    return {
        "Python Version": sys.version,
        "Executable": sys.executable,
    }


def get_installed_packages():
    try:
        packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
        packages = packages.decode("utf-8").split('\n')
        return packages
    except Exception as e:
        return str(e)


def get_cuda_info():
    cuda_available = torch.cuda.is_available()
    cuda_info = {
        "CUDA Available": cuda_available,
    }
    if cuda_available:
        cuda_info["CUDA Version"] = torch.version.cuda
        cuda_info["Number of GPUs"] = torch.cuda.device_count()
        cuda_info["GPUs"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    return cuda_info


def get_os_info():
    return {
        "System": platform.system(),
        "Node Name": platform.node(),
        "Release": platform.release(),
        "Version": platform.version(),
        "Machine": platform.machine(),
        "Processor": platform.processor(),
    }


def print_environment_info():
    logger.info("=== Python Information ===")
    python_info = get_python_info()
    for key, value in python_info.items():
        logger.info(f"{key}: {value}")

    logger.info("\n=== Installed Packages ===")
    packages = get_installed_packages()
    if isinstance(packages, list):
        for package in packages:
            logger.info(package)
    else:
        logger.info(f"Error retrieving packages: {packages}")

    logger.info("\n=== CUDA Information ===")
    cuda_info = get_cuda_info()
    for key, value in cuda_info.items():
        if isinstance(value, list):
            logger.info(f"{key}:")
            for item in value:
                logger.info(f"  - {item}")
        else:
            logger.info(f"{key}: {value}")

    logger.info("\n=== OS Information ===")
    os_info = get_os_info()
    for key, value in os_info.items():
        logger.info(f"{key}: {value}")


if __name__ == "__main__":
    print_environment_info()
