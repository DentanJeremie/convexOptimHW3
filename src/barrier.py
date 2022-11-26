import typing as t

import numpy as np

from src.centering import Centering
from src.utils.logging import logger

class Barrier(object):
    
    def __init__(
        self,
        Q: np.ndarray,
        p: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        v0: np.ndarray,
        eps: float,
        *,
        eps_newton: float = -1,
        alpha_bls: float = .3,
        beta_bls: float = .9,
        t0_barrier: float = 1.0,
        mu_barrier: float = 2,
        max_step_newton: int = 100,
        max_step_barrier: int = 100,
    ):
        # Debug
        logger.debug('Initializing a new barrier optimizer.')

        self.Q: np.ndarray = Q
        self.p: np.ndarray = p
        self.A: np.ndarray = A
        self.b: np.ndarray = b
        self.v0: np.ndarray = v0
        self.eps: float = eps
        self.eps_newton: float = eps_newton

        # Use same eps for newton and barrier ?
        if eps_newton == -1:
            self.eps_newton = self.eps

        # Backtracking line search
        self.alpha_bls: float = alpha_bls
        self.beta_bls: float = beta_bls

        # Barrier parameters
        self.t0_barrier: float = t0_barrier
        self.mu_barrier: float= mu_barrier
        self.m : int = self.b.shape[-1]

        # Max step
        self.max_step_newton: int = max_step_newton
        self.max_step_barrier: int = max_step_barrier

        # Checking if v0 is in the domain
        if not self.is_in_domain(self.v0):
            logger.warning("Out-of-domain starting point.")

    def is_in_domain(self, v:np.ndarray) -> bool:
        return (self.b - self.A @ v > 0).all()
    
    def evaluate(self, v:np.ndarray) -> np.ndarray:
        return (
            v @ self.Q @ v
            + self.p @ v
        )
    
    def barrier_optimize(self):
        logger.info('Starting a new barrier optimization.')

        v = self.v0
        t = self.t0_barrier
        variables_iterates = [v]
        objective_iterates = [self.evaluate(v)]
        logger.debug(f'Initial objective value : {objective_iterates[-1]}')

        step = 0
        while self.m / t >= self.eps and step <= self.max_step_barrier:
            newton_solver = Centering(
                Q = self.Q,
                p = self.p,
                A = self.A,
                b = self.b,
                t = t,
                v0 = v,
                eps = self.eps_newton,
                max_step=self.max_step_newton,
                alpha_bls=self.alpha_bls,
                beta_bls=self.beta_bls
            )
            variables_iterates, _, _ = newton_solver.newton_optimize()

            v = variables_iterates[-1]
            t *= self.mu_barrier
            variables_iterates.append(v)
            objective_iterates.append(self.evaluate(v))

            # Debug
            logger.debug(f'New barrier step with value t={t}')
            logger.debug(f'Objective value with this v : {objective_iterates[-1]}')

            step += 1

        if step > self.max_step_barrier:
            logger.warning("Max step number reached before gradient norm stopping criterion.")

        min_reached = np.min(np.array(objective_iterates))
        logger.info(f'End of Barrier Optimization. Best value reached: {min_reached}')
        
        return variables_iterates, objective_iterates


def barr_method(
    Q: np.ndarray,
    p: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    v0: np.ndarray,
    eps: float,
) -> t.List[np.ndarray]:

    solver = Barrier(Q, p, A, b, v0, eps)
    return solver.barrier_optimize()[0]


if __name__ == '__main__':

    dim = 20

    q_gen = np.random.random((dim, dim))
    Q = np.matmul(q_gen, q_gen.transpose()) + np.eye(dim)
    p = np.random.random(dim)
    A = np.random.random((5*dim, dim))
    b = 100+np.random.random(5*dim)
    v0 = np.zeros(dim)
    eps = 10e-8

    test = Barrier(Q, p, A, b, v0, eps)
    variables_iterates, objective_iterates = test.barrier_optimize()

    logger.info(f'End of Barrier optimization.')
    logger.info(f'Objective value with this v : {objective_iterates[-1]}')

