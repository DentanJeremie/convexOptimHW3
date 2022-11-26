import logging
import sys

from src.utils.pathtools import project

logger = logging.getLogger('root')
logFormatter = logging.Formatter('{relativeCreated:8.0f}ms {levelname:5s} [{filename}] {message:s}', style='{')
logger.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler(project.log)
fileHandler.setFormatter(logFormatter)
fileHandler.setLevel(logging.DEBUG)
logger.addHandler(fileHandler)

sum_fileHandler = logging.FileHandler(project.sum_log)
sum_fileHandler.setFormatter(logFormatter)
sum_fileHandler.setLevel(logging.INFO)
logger.addHandler(sum_fileHandler)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
consoleHandler.setLevel(logging.DEBUG)
logger.addHandler(consoleHandler)

def init_logging(script_name:str) -> None:
    logger.info(f'RUNNING {str(script_name).upper()}')
    logger.info(f'SUMMARY LOG AT INFO LEVEL STORED AT: {project.as_relative(project.sum_log)}')
    logger.info(f'COMPLETE LOG AT DEBUG LEVEL STORED AT: {project.as_relative(project.log)}')
    
# Silent unuseful log
from matplotlib import pyplot as plt
plt.set_loglevel("warning")

