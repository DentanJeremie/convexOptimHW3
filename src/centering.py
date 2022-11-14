import autograd.numpy as np
from autograd import grad

from utils.pathtools import project

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
    ):
        self.Q: np.ndarray = Q
        self.p: np.ndarray = p
        self.A: np.ndarray = A
        self.b: np.ndarray = b
        self.t: float = t
        self.v0: np.ndarray = v0
        self.eps: float = eps

        self.d = self.v0.shape[0]

        def evaluate(v: np.ndarray) -> float:
            return (
                v.T @ self.Q @ v
                + self.p.T @ v
                - np.log(b - self.A @ v)
            )

        self.f = evaluate
        self.D_f = grad(self.f)
        self.D2_f = grad(self.D_f)

    def evaluate(self, v:np.ndarray) -> None:
        return (
            v.T @ self.Q @ v
            + self.p.T @ v
            - np.log(b - self.A @ v)
        )

    def __call__(self, v:np.ndarray) -> float:
        return self.f(v)
    
    def autograd(self, v:np.ndarray) -> np.ndarray:
        return self.D_f(v)

    def autohessian(self, v:np.ndarray) -> np.ndarray:
        return self.D2_f(v)

    def grad(self, v:np.ndarray) -> np.ndarray:
        part_function = self.t*(2*self.Q@v + self.p)

        factors = 1 / (self.b - self.A@v)
        part_barrier = self.A.T @ factors

        return part_function + part_barrier


if __name__ == '__main__':
    pass

Q = np.array([[1,2],[3,4]])
p = np.array([5,6])
A = np.array([[0,2],[8,3]])
b = np.array([5,2])
t = .7
v0 = np.array([2,8])
eps = 10e-6

test = Centering(Q, p, A, b, t, v0, eps)
v0 = np.array([2,8])