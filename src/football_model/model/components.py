import pymc as pm
import pytensor.tensor as pt
from pytensor import scan

def ar1_team_process(
    name,
    n_time,
    n_teams,
    sigma,
    rho,
    init_scale=0.2,
):
    # Non-centered parameterization for better sampling when sigma is small
    z = pm.Normal(f"{name}_std", 0, 1, shape=(n_time, n_teams))
    
    # Use scan instead of Python loop for efficiency with many time steps
    def step(z_t, x_prev, rho, sigma):
        return rho * x_prev + sigma * z_t
    
    # Initialize with first time step
    x_init = init_scale * z[0]
    
    # Scan over remaining time steps (t=1 to n_time-1)
    x_rest, _ = scan(
        fn=step,
        sequences=[z[1:]],
        outputs_info=[x_init],
        non_sequences=[rho, sigma],
        n_steps=n_time - 1,
        strict=True
    )
    
    # Concatenate initial and scanned results
    result = pt.concatenate([x_init[None, :], x_rest], axis=0)
    
    return pm.Deterministic(name, result)


def centered_over_teams(x, name):
    return pm.Deterministic(
        name,
        x - x.mean(axis=1, keepdims=True)
    )


def soft_clip(x, limit=2.0):
    return limit * pm.math.tanh(x / limit)
