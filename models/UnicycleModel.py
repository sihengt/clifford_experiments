import torch

class UnicycleModel:
    def __init__(self):
        print("Unicycle")

    # State transition function
    def f(X, U, dt):
        # X (state) should be (x, y, theta)^T
        # U (action) should be (v, omega)
        assert(X.shape == 3)
        assert(X.shape == 2)
        theta_k = X[2]
        v_k, omega_k = U

        return X + dt * torch.tensor([
            [v_k * torch.cos(theta_k)],
            [v_k * torch.sin(theta_k)],
            [omega_k]
        ])

    
