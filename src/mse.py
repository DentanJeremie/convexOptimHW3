import logging
import torch

from barrier import Barrier
import utils

class MSE(object):

    def __init__(
        self,
        n: int,
        d: int,
        lambda_lasso: float,
        *,
        eps_barrier: float = 10e-6,
        eps_newton: float = 10e-6,
        alpha_bls: float = .3,
        beta_bls: float = .9,
        t0_barrier: float = 1.0,
        mu_barrier: float = 2,
        max_step_newton: int = 100,
        max_step_barrier: int = 100,
        data_noise: float = .3,
    ):
        self.n: int = n
        self.d: int = d
        self.lambda_lasso: float = lambda_lasso

        self.eps_barrier: float = eps_barrier
        self.eps_newton: float = eps_newton
        self.alpha_bls: float = alpha_bls
        self.beta_bls: float = beta_bls
        self.t0_barrier: float = t0_barrier
        self.mu_barrier: float = mu_barrier
        self.max_step_newton: int = max_step_newton
        self.max_step_barrier: int = max_step_barrier

        # Initiating data
        self.X: torch.tensor = torch.randn(self.n, self.d)
        self.target_w: torch.tensor = torch.randn(self.d)
        self.data_noise = data_noise
        self.y: torch.tensor = self.X @ self.target_w + self.data_noise * torch.normal(mean=torch.zeros(self.n))

        # Debug
        logging.debug(f'shape X: {self.X.shape}')
        logging.debug(f'shape target_w: {self.target_w.shape}')
        logging.debug(f'shape y: {self.y.shape}')
        
    def optimize_mse(self):
        # Parameters
        Q = -0.5 * torch.eye(self.n)
        p = -1.0 * self.y
        A = torch.cat(
            (
                self.X,
                -1*self.X,
                -1*torch.eye(self.n)
            ),
            dim=1
        )
        b = torch.cat(
            (
                self.lambda_lasso * torch.ones(self.d),
                self.lambda_lasso * torch.ones(self.d),
                torch.zeros(self.n),
            )
        )
        v0 = self.eps_newton * torch.ones(self.n)

        logging.debug(f'shape Q: {Q.shape}')
        logging.debug(f'shape p: {p.shape}')
        logging.debug(f'shape A: {A.shape}')
        logging.debug(f'shape b: {b.shape}')
        logging.debug(f'shape v0: {v0.shape}')

        # Barrier solver
        barrier_solver = Barrier(
            Q = Q,
            p = p,
            A = A,
            b = b,
            v0 = v0,
            eps = self.eps_barrier,
            eps_newton = self.eps_newton,
            alpha_bls = self.alpha_bls,
            beta_bls = self.beta_bls,
            t0_barrier = self.t0_barrier,
            mu_barrier = self.mu_barrier,
            max_step_newton = self.max_step_newton,
            max_step_barrier = self.max_step_barrier,
        )

        optimized = barrier_solver.barrier_optimize()

        logging.debug(f'End of QP optimization.')
        logging.debug(f'Best value found : {min(optimized[1])}')
        logging.debug(f'Number of barrier steps : {len(optimized[1])}')

        return barrier_solver.barrier_optimize()[1]

if __name__ == '__main__':

    n = 2
    d = 10
    lambda_lasso = 10

    solver = MSE(n, d, lambda_lasso)
    objective_iterates = solver.optimize_mse()
    logging.info(f'Optimization done.')

        


