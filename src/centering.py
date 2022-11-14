import logging
import typing as t

import torch

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
        alpha_bls: float = .2,
        beta_bls: float = .9,
        max_step = 100,
    ):
        self.Q: torch.tensor = Q
        self.p: torch.tensor = p
        self.A: torch.tensor = A
        self.b: torch.tensor = b
        self.t: float = t
        self.v0: torch.tensor = v0
        self.eps: float = eps

        # Backtracking line search
        self.alpha_bls = alpha_bls
        self.beta_bls = beta_bls

        # Max step
        self.max_step = max_step

        # Checking if v0 is in the domain
        if not (self.A @ self.v0 <= self.b).all().item():
            logging.warning("Out-of-domain starting point.")

        # grad / no grad on the parameters
        self.Q.requires_grad = False
        self.p.requires_grad = False
        self.A.requires_grad = False
        self.b.requires_grad = False
        self.v0.requires_grad = True

        # Dimension of our problem
        self.d = self.v0.shape[0]

    def evaluate(self, v:torch.tensor) -> torch.tensor:
        return (
            v@self.Q@v
            + self.p@v
            + torch.log(self.b - v@self.A).sum()
        )

    def grad_old(self, v: torch.tensor) -> torch.tensor:
        v.requires_grad = True
        if v.grad is not None:
            v.grad.zero_()
        output = self.evaluate(v)
        output.backward()
        return v.grad

    def grad(self, v: torch.tensor) -> torch.tensor:
        return torch.autograd.grad(outputs = self.evaluate(v), inputs = v)[0]

    def hessian(self, v:torch.tensor) -> torch.tensor:
        return torch.autograd.functional.hessian(self.evaluate, v)

    def newton_direction(self, v:torch.tensor) -> torch.tensor:
        hess = self.hessian(v)
        grad = self.grad(v)
        return -1 * grad @ torch.inverse(hess)

    def backtrack_line_search(self, v:torch.tensor, direction:torch.tensor):
        t = 1
        while (
            self.evaluate(v + t*direction)
            >= self.evaluate(v) + self.alpha_bls * t * self.grad(v) @ direction
        ):
            t *= self.beta_bls

        return v + t*direction

    def newton_optimize(self):
        v = self.v0
        variables_iterates = [v]
        objective_iterates = [self.evaluate(v).item()]
        gradients_norm_iterates = [torch.linalg.vector_norm(self.grad(v)).item()]

        step = 0
        while torch.linalg.vector_norm(self.grad(v)).item() > self.eps and step < self.max_step:
            direction = self.newton_direction(v)
            v = self.backtrack_line_search(v, direction)

            variables_iterates.append(v)
            objective_iterates.append(self.evaluate(v).item())
            gradients_norm_iterates.append(torch.linalg.vector_norm(self.grad(v)).item())

            step += 1

        if step == self.max_step:
            logging.warning("Max step number reached before gradient norm stopping criterion.")

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

    Q = torch.tensor([[1,2],[3,4]], dtype = torch.float64)
    p = torch.tensor([5,6], dtype = torch.float64)
    A = torch.tensor([[0,2],[8,3]], dtype = torch.float64)
    b = torch.tensor([5,2], dtype = torch.float64)
    t = .7
    v0 = torch.tensor([-12,-8], dtype = torch.float64)
    eps = 10e-8

    test = Centering(Q, p, A, b, t, v0, eps)
    variables_iterates, objective_iterates, gradients_norm_iterates = test.newton_optimize()

    logging.info(f'Objective iterates: {objective_iterates}')
    logging.info(f'Variables iterates: {variables_iterates}')
    logging.info(f'Gradients norm iterates: {gradients_norm_iterates}')