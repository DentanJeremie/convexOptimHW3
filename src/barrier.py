import logging
import typing as t

import torch

from centering import Centering

class Barrier(object):
    
    def __init__(
        self,
        Q: torch.tensor,
        p: torch.tensor,
        A: torch.tensor,
        b: torch.tensor,
        v0: torch.tensor,
        eps: float,
        *,
        eps_newton: float = -1,
        alpha_bls: float = .2,
        beta_bls: float = .9,
        t0_barrier: float = 1.0,
        mu_barrier: float = 2,
        max_step_newton: int = 100,
        max_step_barrier: int = 100,
    ):
        self.Q: torch.tensor = Q
        self.p: torch.tensor = p
        self.A: torch.tensor = A
        self.b: torch.tensor = b
        self.v0: torch.tensor = v0
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
        self.m : int = self.b.size(-1)

        # Max step
        self.max_step_newton: int = max_step_newton
        self.max_step_barrier: int = max_step_barrier

        # Checking if v0 is in the domain
        if not self.is_in_domain(self.v0):
            logging.warning("Out-of-domain starting point.")

    def is_in_domain(self, v:torch.tensor) -> bool:
        return (self.b - v@self.A > 0).all().item()
    
    def evaluate(self, v:torch.tensor) -> torch.tensor:
        return (
            v@self.Q@v
            + self.p@v
        )
    
    def barrier_optimize(self):
        v = self.v0
        t = self.t0_barrier
        variables_iterates = [v]
        objective_iterates = []

        step = 0
        while self.m / t >= self.eps and step <= self.max_step_barrier:
            newton_solver = Centering(
                Q = self.Q,
                p = self.p,
                A = self.A,
                b = self.b,
                t = t,
                v0 = v.detach(),
                eps = self.eps_newton,
                max_step=self.max_step_newton,
                alpha_bls=self.alpha_bls,
                beta_bls=self.beta_bls
            )
            optimized = newton_solver.newton_optimize()

            v = optimized[0][-1]
            t *= self.mu_barrier
            variables_iterates.append(v)
            objective_iterates.append(self.evaluate(v).item())

            # Debug
            logging.debug(f'New barrier step with value t={t}')
            logging.debug(f'Newton optimization values : {optimized[1]}')
            logging.debug(f'New value for v : {v}')
            logging.debug(f'Objective value with this v : {objective_iterates[-1]}')

            step += 1

        if step > self.max_step_barrier:
            logging.warning("Max step number reached before gradient norm stopping criterion.")

        logging.debug('End of Barrier Optimization.')
        
        return variables_iterates, objective_iterates


def barr_method(
    Q: torch.tensor,
    p: torch.tensor,
    A: torch.tensor,
    b: torch.tensor,
    v0: torch.tensor,
    eps: float,
) -> t.List[torch.tensor]:

    solver = Barrier(Q, p, A, b, v0, eps)
    return solver.barrier_optimize()[0]


if __name__ == '__main__':

    Q = torch.tensor([[1,2],[3,4]], dtype = torch.float64)
    p = torch.tensor([5,6], dtype = torch.float64)
    A = torch.tensor([[0,2],[8,3]], dtype = torch.float64)
    b = torch.tensor([5,2], dtype = torch.float64)
    v0 = torch.tensor([-12,-8], dtype = torch.float64)
    eps = 10e-8

    test = Barrier(Q, p, A, b, v0, eps)
    variables_iterates, objective_iterates = test.barrier_optimize()

    logging.info(f'End of Barrier optimization.')
    logging.info(f'Final value of v : {variables_iterates[-1]}')
    logging.info(f'Objective value with this v : {objective_iterates[-1]}')

