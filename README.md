# pyNMPC
Nonlinear Model Predictive Control based on CVXPY and jax

## Demo
### Unicycle trajectory tracking
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaoanlu/pyNMPC/blob/main/examples/unicycle.ipynb)

### Quadrotor suspension point-to-point control
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaoanlu/pyNMPC/blob/main/examples/quadrotor_suspension.ipynb)

## Usage
### 1. Define a jittable dynamics function
The function should take input args `x`, `u`, and `dt` and return a new state vector.
```python
def dynamics(x: jnp.ndarray, u: jnp.ndarray, dt: float) -> jnp.ndarray:
    """
      Unicycle dynamics model.
        x_next = x + u[0] * cos(theta) * dt
        y_next = y + u[0] * sin(theta) * dt
        theta_next = theta + u[1] * dt
    """
    theta = x[2]
    return x.at[0].set(x[0] + u[0] * jnp.cos(theta) * dt) \
            .at[1].set(x[1] + u[0] * jnp.sin(theta) * dt) \
            .at[2].set(x[2] + u[1] * dt)
```

### Set `MPCParams` values
```python
from nmpc import MPCParams
params = MPCParams(
    dt=0.1,                                # Time step
    N=20,                                  # Horizon length
    n_states=3,                                  # State vector dimension
    n_controls=2,                                  # Control input dimension
    Q=jnp.diag(jnp.array([10.0, 10.0, 1.0])),  # State weights
    QN=jnp.diag(jnp.array([10.0, 10.0, 1.0])),  # Terminal state weights
    R=jnp.diag(jnp.array([1.0, 0.1])),         # Control input weights
    x_ref=jnp.array([0.0, 0.0, 0.0]),          # Will be overridden
    u_min=jnp.array([0.0, -1.0]),                  # Lower buond of control input
    u_max=jnp.array([1.0, 1.0]),             # Upper buond of control input
    max_sqp_iter=5,                       # Max SQP iterations
    sqp_tol=1e-4,                         # SQP convergence tolerance
    verbose=False                         # Verbosity flag
)
```

### Instantiate `NMPC` and solve
```python
from nmpc import NMPC
mpc = NMPC(dynamics_fn=dynamics, params=params, solver_opts={...})  # pass dynamics and parameter
current_state = jax.numpy.array([...])
current_reference = jax.numpy.array([...])
mpc_result = mpc.solve(x0=current_state, x_ref=current_reference)

# optimized control
mpc_result.u_traj
# predicted state
mpc_result.x_traj
```

## Requirements
- Python 3.10+
- cvxpy
- jax
