import typing as t

import numpy as np

from src.utils.pathtools import project
from src.utils.logging import logger

class Centering(object):

    def __init__(
        self,
        Q: np.ndarray,
        p: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        t: float,
        v0: np.ndarray,
        eps: float,
        *,
        alpha_bls: float = .3,
        beta_bls: float = .9,
        max_step = 1000,
    ):
        # Debug
        logger.debug('Initializing a new Newton optimizer.')

        self.Q: np.ndarray = Q
        self.p: np.ndarray = p
        self.A: np.ndarray = A
        self.b: np.ndarray = b
        self.t: float = t
        self.v0: np.ndarray = v0
        self.eps: float = eps

        # Backtracking line search
        self.alpha_bls: float = alpha_bls
        self.beta_bls: float = beta_bls

        # Max step
        self.max_step: int = max_step

        # Not in domain warning
        if not self.is_in_domain(self.v0):
            logger.warning("Out-of-domain starting point.")

    def is_in_domain(self, v:np.ndarray) -> bool:
        return (self.g(v) > 0).all()

    def g(self, v: np.ndarray) -> np.ndarray:
        return self.b - self.A @ v

    def H(self, v: np.ndarray) -> np.ndarray:
        return self.A.T / self.g(v)

    def evaluate(self, v: np.ndarray) -> np.ndarray:
        return self.t * (v @ self.Q @ v + self.p @ v) - np.sum(np.log(self.g(v)))

    def gradient(self, v: np.ndarray) -> np.ndarray:
        return 2 * self.t * self.Q @ v + self.t * self.p + np.sum(self.H(v), axis=1)

    def hessian(self, v: np.ndarray) -> np.ndarray:
        return 2 * self.t * self.Q + self.H(v) @ self.H(v).T
    
    def evaluate_derivatives(self, v:np.ndarray) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes the value of the function, of its gradient, and of its Hessian

        :param v: The point on which to compute the function and its derivatives
        :returns: A tuple (f(v), grad_f(v), hessian_f(v))
        """

        g: np.ndarray = self.b - self.A @ v
        H: np.ndarray = self.A.transpose() / g

        function_value = self.t * (v @ self.Q @ v + self.p @ v) - np.sum(np.log(g))
        gradient_value = 2 * self.t * self.Q @ v + self.t * self.p + np.sum(H, axis=1)
        hessian_value = 2 * self.t * self.Q + H @ H.T

        return (function_value, gradient_value, hessian_value)

    def newton_decrement_squared(
        self,
        v: np.ndarray,
        direction: np.ndarray,
    ) -> float:
        return (-self.gradient(v) @ direction) / 2

    def newton_direction(self, v:np.ndarray) -> np.ndarray:
        grad = self.gradient(v)
        hess = self.hessian(v)
        return -1 * np.linalg.inv(hess) @ grad

    def backtrack_line_search(self, v:np.ndarray, direction:np.ndarray):
        r = 1 / self.beta_bls
        while (
            (
                not self.is_in_domain(v + r * direction)
            )
            or
            (
                self.evaluate(v + r * direction)
                > self.evaluate(v) + self.alpha_bls * r * self.gradient(v) @ direction
            )
        ):
            r *= self.beta_bls

        logger.debug(f'Backtraching line search final parameter : {r}')
        return v + r * direction

    def newton_optimize(self):
        logger.debug('Starting a new Newton optimization.')
        v = self.v0
        func = self.evaluate(v)

        variables_iterates = [v]
        objective_iterates = [func]
        newton_decrement_iterates = []
        logger.debug(f'Initial objective value: {func}')

        step = 0
        while True and step <= self.max_step:
        
            direction = self.newton_direction(v)
            decrement = self.newton_decrement_squared(v, direction)
            newton_decrement_iterates.append(decrement)
            logger.debug(f'Newton decrement : {newton_decrement_iterates[-1]}')

            if decrement < self.eps:
                logger.debug(f"Newton's decrement criterion met : {decrement:4.2e} < {self.eps:4.2e}")
                break

            logger.debug('Newton decrement not met, updating v')
            v = self.backtrack_line_search(v, direction)
            variables_iterates.append(v)
            objective_iterates.append(self.evaluate(v).item())
            logger.debug(f'New objective value : {objective_iterates[-1]}')

            step += 1

        if step > self.max_step:
            logger.warning("Max step number reached before gradient norm stopping criterion.")

        logger.debug('End of Newton optimization.')

        return variables_iterates, objective_iterates, newton_decrement_iterates


def centering_step(
    Q: np.ndarray,
    p: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    t: float,
    v0: np.ndarray,
    eps: float,
) -> t.List[np.ndarray]:

    solver = Centering(Q, p, A, b, t, v0, eps)
    return solver.newton_optimize()[0]


if __name__ == '__main__':

    np.random.seed()
    dim = 20

    q_gen = np.random.random((dim, dim))
    Q = np.matmul(q_gen, q_gen.transpose()) + np.eye(dim)
    p = np.random.random(dim)
    A = np.random.random((5*dim, dim))
    b = 100+np.random.random(5*dim)
    t_ = 10
    v0 = np.zeros(dim)
    eps = 10e-8

    test = Centering(Q, p, A, b, t_, v0, eps)
    variables_iterates, objective_iterates, newton_decrement_iterates = test.newton_optimize()

    logger.info(f'Objective last iterate: {objective_iterates[-1]}')
    logger.info(f'Newton decrement last iterate: {newton_decrement_iterates[-1]}')