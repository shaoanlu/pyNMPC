# pyNMPC
Nonlinear Model Predictive Control based on CVXPY and jax

```math
\begin{aligned}
\min_{\mathbf{x}, \mathbf{u}, \mathbf{s}} \quad & \sum_{k=0}^{N-1} \left( \|\mathbf{x}_k - \mathbf{x}_k^{\text{ref}}\|_Q^2 + \|\mathbf{u}_k\|_R^2 + \rho\|\mathbf{s}_k\|^2\right) + \|\mathbf{x}_N - \mathbf{x}_N^{\text{ref}}\|_{Q_N}^2 \\
\text{subject to} \quad & \mathbf{x}_{k+1} = f(x_k, u_k, \Delta t), \quad k = 0, \dots, N-1 \\
& \mathbf{x}_0 = \mathbf{x}_{\text{init}} \\
& \mathbf{x}_{\text{min}}-\mathbf{s}_k \le \mathbf{x}_k \le \mathbf{x}_{\text{max}}+\mathbf{s}_k, \quad k = 0, \dots, N \\
& \mathbf{u}_{\text{min}} \le \mathbf{u}_k \le \mathbf{u}_{\text{max}}, \quad k = 0, \dots, N-1 \\
& \mathbf{s}_k \ge 0, \quad k = 0, \dots, N
\end{aligned}
```

## Demo
### Unicycle trajectory tracking
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaoanlu/pyNMPC/blob/main/examples/unicycle.ipynb)

### Quadrotor suspension point-to-point control
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaoanlu/pyNMPC/blob/main/examples/quadrotor_suspension.ipynb)

## Usage
### 1. Define a jittable dynamics function
The function should take input args `x`, `u`, and `dt` and return a new state vector.
```python
import jax.numpy as jnp

def dynamics(x: jnp.ndarray, u: jnp.ndarray, dt: float) -> jnp.ndarray:
    """
      Unicycle dynamics model.
        x_next = x + u[0] * cos(theta) * dt
        y_next = y + u[0] * sin(theta) * dt
        theta_next = theta + u[1] * dt
    """
    theta = x[2]
    return x + jnp.array([u[0] * jnp.cos(theta), u[0] * jnp.sin(theta), u[1]]) * dt
```

### 2. Set `MPCParams` values
```python
import jax.numpy as jnp
from nmpc import MPCParams

params = MPCParams(
    dt=0.1,                                    # Time step
    N=20,                                      # Horizon length
    n_states=3,                                # State vector dimension
    n_controls=2,                              # Control input dimension
    Q=jnp.diag(jnp.array([10.0, 10.0, 1.0])),  # State weights
    QN=jnp.diag(jnp.array([10.0, 10.0, 1.0])), # Terminal state weights
    R=jnp.diag(jnp.array([1.0, 0.1])),         # Control input weights
    x_ref=jnp.array([0.0, 0.0, 0.0]),          # Will be overridden
    x_min=jnp.array([-3, -3, -float("inf")]),  # Lower buond of state
    x_max=jnp.array([3, 3, float("inf")]),     # Upper buond of state
    u_min=jnp.array([0.0, -1.0]),              # Lower buond of control input
    u_max=jnp.array([1.0, 1.0]),               # Upper buond of control input
    slack_weight=1e4,                          # Slack penalty for soft state constraints
    max_sqp_iter=5,                            # Max SQP iterations
    sqp_tol=1e-4,                              # SQP convergence tolerance
    verbose=False                              # Verbosity flag
)
```

### 3. Instantiate `NMPC` and solve
```python
import jax.numpy as jnp
from nmpc import NMPC

mpc = NMPC(dynamics_fn=dynamics, params=params, solver_opts={...})  # pass dynamics and parameter
current_state = jnp.array([...])
current_reference = jnp.array([...])
mpc_result = mpc.solve(x0=current_state, x_ref=current_reference)

# Optimized control
mpc_result.u_traj
# Predicted state
mpc_result.x_traj
```

## Requirements
- Python 3.10+
- `cvxpy[osqp]`
- `jax`
