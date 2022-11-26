from pathlib import Path
import sys

from src.mse import MSE
from src.utils.logging import logger
from src.utils.pathtools import project

if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent.parent))

logger.info(f'RUNNING FULL COMPUTATION FOR HOMEWORK 3')
logger.info(f'SUMMARY LOG AT INFO LEVEL STORED AT: {project.as_relative(project.sum_log)}')
logger.info(f'COMPLETE LOG AT DEBUG LEVEL STORED AT: {project.as_relative(project.log)}')

n = 50
d = 1000
lambda_lasso = 10
mu_list = [2, 15, 50, 100, 500, 1000]
solver = MSE(n, d, lambda_lasso)

logger.info(f'Values of mu tested : {mu_list}')
solver.log_config()

objective_iterates = solver.compute_figures_mu(mu_list)