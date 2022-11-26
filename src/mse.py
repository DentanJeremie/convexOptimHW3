import typing as t

import matplotlib.pyplot as plt
import numpy as np

from src.barrier import Barrier
from src.utils.logging import logger
from src.utils.pathtools import project

class MSE(object):

    def __init__(
        self,
        n: int,
        d: int,
        lambda_lasso: float,
        *,
        eps_barrier: float = 10e-8,
        eps_newton: float = 10e-8,
        eps_figures: float = 10e-16,
        alpha_bls: float = .3,
        beta_bls: float = .9,
        t0_barrier: float = 1.0,
        mu_barrier: float = 2,
        max_step_newton: int = 100,
        max_step_barrier: int = 100,
    ):
        self.n: int = n
        self.d: int = d
        self.lambda_lasso: float = lambda_lasso

        self.eps_barrier: float = eps_barrier
        self.eps_newton: float = eps_newton
        self.eps_figures: float = eps_figures
        self.alpha_bls: float = alpha_bls
        self.beta_bls: float = beta_bls
        self.t0_barrier: float = t0_barrier
        self.mu_barrier: float = mu_barrier
        self.max_step_newton: int = max_step_newton
        self.max_step_barrier: int = max_step_barrier

        # Initiating data
        self.X: np.ndarray = np.random.normal(size = (self.n, self.d))
        self.y: np.ndarray = np.random.normal(size = self.n)

        # Initializing the patameters for the solvers
        # Parameters
        self.Q = 0.5 * np.eye(self.n)
        self.p = 1.0 * self.y
        self.A = np.concatenate(
            (
                self.X.T,
                -self.X.T,
            ),
            axis = 0,
        )
        self.b = self.lambda_lasso * np.ones(2 * self.d)
        self.v0 = np.zeros(self.n)

        # Debug
        logger.debug(f'shape X: {self.X.shape}')
        logger.debug(f'shape y: {self.y.shape}')
        logger.debug(f'shape Q: {self.Q.shape}')
        logger.debug(f'shape p: {self.p.shape}')
        logger.debug(f'shape A: {self.A.shape}')
        logger.debug(f'shape b: {self.b.shape}')
        logger.debug(f'shape v0: {self.v0.shape}')
        
    def log_config(self) -> None:
        """Logs the configuration of the solver in info.
        """
        for param in [
            "n",
            "d",
            "lambda_lasso",
            "eps_barrier",
            "eps_newton",
            "eps_figures",
            "alpha_bls",
            "beta_bls",
            "t0_barrier",
            "mu_barrier",
            "max_step_newton",
            "max_step_barrier",
        ]:
            logger.info(f'Config: {param}: {self.__getattribute__(param)}')

    def optimize_mse(self):

        # Barrier solver
        barrier_solver = Barrier(
            Q = self.Q,
            p = self.p,
            A = self.A,
            b = self.b,
            v0 = self.v0,
            eps = self.eps_barrier,
            eps_newton = self.eps_newton,
            alpha_bls = self.alpha_bls,
            beta_bls = self.beta_bls,
            t0_barrier = self.t0_barrier,
            mu_barrier = self.mu_barrier,
            max_step_newton = self.max_step_newton,
            max_step_barrier = self.max_step_barrier,
        )

        optimized_objectives = barrier_solver.barrier_optimize()[1]

        logger.debug(f'End of QP optimization.')
        logger.debug(f'Best value found : {min(optimized_objectives)}')
        logger.debug(f'Number of barrier steps : {len(optimized_objectives)}')

        return optimized_objectives

    def compute_figures_mu(self, mu_list: t.List[float]) -> None:
        """Computes the figures and stores them.

        :param mu_list: The list of parameter mu to test
        """
        ax = plt.axes(label='histogram_of_degrees')
        ax.set_xlabel('Barrier iterations')
        ax.set_ylabel('log10(objective_value - min_reached)')
        ax.set_title('Performance of barrier method depending on mu')

        for mu in mu_list:
            logger.info(f'Optimizing with mu={mu}')
            self.mu_barrier = mu
            objective_values = np.array(self.optimize_mse())

            min_objective = np.min(objective_values)
            plotable = objective_values[objective_values != min_objective]
            iterations = [k for k in range(len(plotable))]

            ax.step(
                iterations,
                np.log10(plotable - min_objective + self.eps_figures),
                label = f'mu = {mu}',
                where = 'post',
            )

        ax.grid()
        ax.legend()
        plt.savefig(project.mu_figure)
        plt.close()

        logger.info(f'Comparison of mus stored at {project.as_relative(project.mu_figure)}')


if __name__ == '__main__':

    n = 50
    d = 1000
    lambda_lasso = 10
    mu_list = [2, 15, 50, 100, 500, 1000]

    solver = MSE(n, d, lambda_lasso)
    objective_iterates = solver.compute_figures_mu(mu_list)

    

