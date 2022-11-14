import torch

from utils.pathtools import project

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
    ):
        self.Q: torch.tensor = Q
        self.p: torch.tensor = p
        self.A: torch.tensor = A
        self.b: torch.tensor = b
        self.t: float = t
        self.v0: torch.tensor = v0
        self.eps: float = eps

        # No grad on those parameters
        self.Q.requires_grad = False
        self.p.requires_grad = False
        self.A.requires_grad = False
        self.b.requires_grad = False

        # Dimension of our problem
        self.d = self.v0.shape[0]

    def evaluate(self, v:torch.tensor) -> torch.tensor:
        return (
            v@self.Q@v
            + self.p@v
            + torch.log(self.b - v@self.A).sum()
        )

    def grad(self, v: torch.tensor) -> torch.tensor:
        v.requires_grad = True
        if v.grad is not None:
            v.grad.zero_()
        output = self.evaluate(v)
        output.backward()
        return v.grad

    def hessian(self, v:torch.tensor) -> torch.tensor:
        return torch.autograd.functional.hessian(self.evaluate, v)

    def __call__(self, v:torch.tensor) -> float:
        return self.evaluate(v)


if __name__ == '__main__':
    pass

Q = torch.tensor([[1,2],[3,4]], dtype = torch.float64)
p = torch.tensor([5,6], dtype = torch.float64)
A = torch.tensor([[0,2],[8,3]], dtype = torch.float64)
b = torch.tensor([5,2], dtype = torch.float64)
t = .7
v0 = torch.tensor([2,8], dtype = torch.float64)
eps = 10e-6

test = Centering(Q, p, A, b, t, v0, eps)
v = torch.tensor([-10,-30], dtype = torch.float64)