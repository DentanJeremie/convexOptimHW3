import logging
import typing as t

import torch

import utils

class Centering(object):

    def __init__(
        self,
        Q: torch.tensor,
        p: torch.tensor,
        A: torch.tensor,
        b: torch.tensor,
        t: float,
        v0: torch.tensor,
        eps: float,
        *,
        alpha_bls: float = .3,
        beta_bls: float = .9,
        max_step = 1000,
    ):
        # Debug
        logging.debug('Initializing a new Newton optimizer.')

        self.Q: torch.tensor = Q
        self.p: torch.tensor = p
        self.A: torch.tensor = A
        self.b: torch.tensor = b
        self.t: float = t
        self.v0: torch.tensor = v0
        self.eps: float = eps

        # Backtracking line search
        self.alpha_bls: float = alpha_bls
        self.beta_bls: float = beta_bls

        # Max step
        self.max_step: int = max_step

        # grad / no grad on the parameters
        self.Q.requires_grad = False
        self.p.requires_grad = False
        self.A.requires_grad = False
        self.b.requires_grad = False
        self.v0.requires_grad = False

        # Not in domain warning
        if not self.is_in_domain(self.v0):
            logging.warning("Out-of-domain starting point.")

    def is_in_domain(self, v:torch.tensor) -> bool:
        return (self.b - v@self.A > 0).all().item()
    
    def evaluate(self, v:torch.tensor) -> torch.tensor:
        return (
            v@self.Q@v
            + self.p@v
            - (1/self.t) * torch.log(self.b - v@self.A).sum()
        )

    def grad(self, v: torch.tensor) -> torch.tensor:
        u = v.detach()
        u.requires_grad = True
        return torch.autograd.grad(outputs = self.evaluate(u), inputs = u)[0]

    def hessian(self, v:torch.tensor) -> torch.tensor:
        return torch.autograd.functional.hessian(self.evaluate, v.detach())

    def newton_direction(self, v:torch.tensor) -> torch.tensor:
        hess = self.hessian(v)
        grad = self.grad(v)
        return -1 * torch.inverse(hess) @ grad

    def sgd_direction(self, v):
        return -1*self.grad(v)

    def backtrack_line_search(self, v:torch.tensor, direction:torch.tensor):
        r = 1
        while (
            (
                not self.is_in_domain(v + r*direction)
            )
            or
            (
                self.evaluate(v + r*direction)
                > self.evaluate(v) + self.alpha_bls * r * self.grad(v) @ direction
            )
        ):
            r *= self.beta_bls

            # Checking convexity inequality
            if (
                self.is_in_domain(v + r*direction)
                and 
                not self.evaluate(v + r*direction) >= self.evaluate(v) + r * self.grad(v) @ direction
            ):
                logging.warning('Unsatisfied convexity inequality.')

        logging.debug(f'Backtraching line search final parameter : {r}')
        return v + r*direction

    def newton_optimize(self):
        logging.debug('Starting a new Newton optimization.')
        v = self.v0
        variables_iterates = [v]
        objective_iterates = [self.evaluate(v).item()]
        gradients_norm_iterates = [torch.linalg.vector_norm(self.grad(v)).item()]

        step = 0
        while torch.linalg.vector_norm(self.grad(v)).item() > self.eps and step <= self.max_step:
            #direction = self.newton_direction(v)
            direction = self.sgd_direction(v)
            v = self.backtrack_line_search(v, direction)

            variables_iterates.append(v)
            objective_iterates.append(self.evaluate(v).item())
            gradients_norm_iterates.append(torch.linalg.vector_norm(self.grad(v)).item())

            logging.debug('New step for Newton optimization')
            logging.debug(f'New value for v : {v}')
            logging.debug(f'Objective value : {objective_iterates[-1]}')
            logging.debug(f'Gradient norm : {gradients_norm_iterates[-1]}')

            step += 1

        if step > self.max_step:
            logging.warning("Max step number reached before gradient norm stopping criterion.")

        logging.debug('End of Newton optimization.')

        return variables_iterates, objective_iterates, gradients_norm_iterates


def centering_step(
    Q: torch.tensor,
    p: torch.tensor,
    A: torch.tensor,
    b: torch.tensor,
    t: float,
    v0: torch.tensor,
    eps: float,
) -> t.List[torch.tensor]:

    solver = Centering(Q, p, A, b, t, v0, eps)
    return solver.newton_optimize()[0]


if __name__ == '__main__':

    dim = 20

    q_gen = torch.randn(dim, dim)
    Q = torch.mm(q_gen, q_gen.t()) + torch.eye(dim)
    p = torch.randn(dim)
    A = torch.randn(dim, 5*dim)
    b = 100+torch.randn(5*dim)
    t = 10
    v0 = torch.zeros(dim)
    eps = 10e-6

    test = Centering(Q, p, A, b, t, v0, eps)
    variables_iterates, objective_iterates, gradients_norm_iterates = test.newton_optimize()

    logging.info(f'Objective last iterates: {objective_iterates[-1]}')
    logging.info(f'Variables last iterates: {variables_iterates[-1]}')
    logging.info(f'Gradients norm last iterates: {gradients_norm_iterates[-1]}')