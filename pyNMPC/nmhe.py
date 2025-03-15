import jax
import jax.numpy as jnp
import numpy as np
import cvxpy as cp
import time
from functools import partial
from typing import Tuple, Callable, Any, Dict
from dataclasses import dataclass


@dataclass(kw_only=True)
class MHEParams:
    """Parameters for Nonlinear Moving Horizon Estimation (NMHE)."""

    dt: float
    N: int
    n_states: int
    n_controls: int
    n_est_params: int
    n_outputs: int
    Q: jnp.ndarray  # Process noise weighting
    R: jnp.ndarray  # Measurement noise weighting
    Px: jnp.ndarray  # Initial state uncertainty weighting
    Pp: jnp.ndarray  # Initial parameter uncertainty weighting
    x_min: jnp.ndarray | None = None
    x_max: jnp.ndarray | None = None
    u_min: jnp.ndarray | None = None
    u_max: jnp.ndarray | None = None
    p_min: jnp.ndarray | None = None
    p_max: jnp.ndarray | None = None
    max_sqp_iter: int = 5
    sqp_tol: float = 1e-4
    verbose: bool = False


@dataclass(frozen=True)
class MHEResult:
    """NMHE Result."""

    x_est: jnp.ndarray
    w_est: jnp.ndarray
    v_est: jnp.ndarray
    p_est: jnp.ndarray
    sqp_iters: int
    solve_time: float
    converged: bool
    cost: float


class NMHE:
    def __init__(
        self,
        dynamics_fn: Callable,
        output_fn: Callable,
        params: MHEParams,
        solver_opts: Dict[str, Any] | None = None,
    ):
        self.dynamics = jax.jit(dynamics_fn)
        self.output_fn = jax.jit(output_fn)
        self.params = params
        self.n_states = params.n_states
        self.n_outputs = params.n_outputs
        self.n_controls = params.n_controls

        if solver_opts is None:
            self.solver_opts = {
                "solver": cp.PIQP,
                "eps_abs": 1e-3,
                "eps_rel": 1e-3,
                "max_iter": 1000,
                "warm_start": True,
                "verbose": False,
            }
        else:
            self.solver_opts = solver_opts

        self.problem, self.x_var, self.w_var, self.v_var, self.p_var, self.parameter_dict = (
            self._create_parameterized_qp(self.params)
        )

    @partial(jax.jit, static_argnums=(0, 3))
    def _linearize_dynamics(self, x_nom, u_nom, dt):
        dyn_fixed_dt = lambda x, u: self.dynamics(x, u, dt)
        A = jax.jacfwd(lambda x: dyn_fixed_dt(x, u_nom))(x_nom)
        B = jax.jacfwd(lambda u: dyn_fixed_dt(x_nom, u))(u_nom)
        c = dyn_fixed_dt(x_nom, u_nom) - A @ x_nom - B @ u_nom  # residual
        return A, B, c

    @partial(jax.jit, static_argnums=(0,))
    def _linearize_output(self, x_nom):
        H = jax.jacfwd(self.output_fn)(x_nom)
        h_x = self.output_fn(x_nom) - H @ x_nom
        return H, h_x

    def _create_parameterized_qp(self, params: MHEParams):
        N, n_states, n_controls, n_outputs, n_est_params = (
            params.N,
            params.n_states,
            params.n_controls,
            params.n_outputs,
            params.n_est_params,
        )
        # Variables
        x_var = [cp.Variable(n_states) for _ in range(N + 1)]
        p_var = cp.Variable(n_est_params) if n_est_params > 0 else None
        w_var = [cp.Variable(n_states) for _ in range(N)]
        v_var = [cp.Variable(n_outputs) for _ in range(N)]
        # Parameters
        x0_param = cp.Parameter(n_states)
        u_seq_param = [cp.Parameter(params.n_controls) for _ in range(N)]
        y_meas_seq_param = [cp.Parameter(n_outputs) for _ in range(N)]
        A_params = [cp.Parameter((n_states, n_states)) for _ in range(N)]
        B_params = [cp.Parameter((n_states, n_controls)) for _ in range(N)]
        c_params = [cp.Parameter(n_states) for _ in range(N)]
        H_params = [cp.Parameter((n_outputs, n_states)) for _ in range(N)]
        h_params = [cp.Parameter(n_outputs) for _ in range(N)]
        est_p0_param = cp.Parameter(n_est_params) if p_var else None

        # NOTE
        """
        Ref:
        https://ftp.esat.kuleuven.be/pub/stadius/ida/reports/11-25.pdf
        https://www.do-mpc.com/en/latest/theory_mhe.html
        
        given process noise w and measurement noise v

        quad(x_mea[0] - x_var[0]) + quad(p_mea-p_var) + SUM(quad(w[k]) + quad(v[k]))
        s.t.
            x[k+1] = A[k] x[k] + B[k] u[k] + c[k] + w[k]
            y[k] = H[k] x[k] + h[k] + v[k]
            x_min <= x_var <= x_max
            u_min <= u_var <= u_max
        """

        # Initialize constraints and cost
        constraints = []
        cost = cp.quad_form(x_var[0] - x0_param, params.Px)
        if p_var:
            cost += cp.quad_form(p_var[0] - est_p0_param, params.Pp)

        for k in range(N):
            # Process dynamics
            constraints.append(
                x_var[k + 1] == A_params[k] @ x_var[k] + B_params[k] @ u_seq_param[k] + c_params[k] + w_var[k]
            )
            cost += cp.quad_form(w_var[k], params.Q)

            # Output measurement
            constraints.append(y_meas_seq_param[k] == H_params[k] @ x_var[k] + h_params[k] + v_var[k])
            cost += cp.quad_form(v_var[k], params.R)

            # State constraints
            if params.x_min is not None:
                constraints.append(x_var[k] >= params.x_min)
            if params.x_max is not None:
                constraints.append(x_var[k] <= params.x_max)

            # Estimated parameter constraints
            if p_var:
                if params.p_min is not None:
                    constraints.append(p_var[k] >= params.p_min)
                if params.p_max is not None:
                    constraints.append(p_var[k] <= params.p_max)

        # Problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        parameter_dict = {
            "x0": x0_param,
            "u_seq": u_seq_param,
            "y_meas_seq": y_meas_seq_param,
            "A_list": A_params,
            "B_list": B_params,
            "c_list": c_params,
            "H_list": H_params,
            "h_list": h_params,
        }

        return problem, x_var, w_var, v_var, p_var, parameter_dict

    def _solve_qp(self, x0, u_seq, y_seq, A_list, B_list, c_list, H_list, h_list) -> Tuple[jnp.ndarray, float]:
        # Convert to numpy for CVXPY
        self.parameter_dict["x0"].value = np.array(x0)
        for k in range(self.params.N):
            self.parameter_dict["u_seq"][k].value = np.array(u_seq[k])
            self.parameter_dict["y_meas_seq"][k].value = np.array(y_seq[k])
            self.parameter_dict["A_list"][k].value = np.array(A_list[k])
            self.parameter_dict["B_list"][k].value = np.array(B_list[k])
            self.parameter_dict["c_list"][k].value = np.array(c_list[k])
            self.parameter_dict["H_list"][k].value = np.array(H_list[k])
            self.parameter_dict["h_list"][k].value = np.array(h_list[k])

        # Solve
        self.problem.solve(**self.solver_opts)

        if self.problem.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError("NMHE QP failed")

        x_est = jnp.array([self.x_var[k].value for k in range(self.params.N + 1)])
        w_est = jnp.array([self.w_var[k].value for k in range(self.params.N)])
        v_est = jnp.array([self.v_var[k].value for k in range(self.params.N)])
        p_est = self.p_var.value if self.p_var else None
        return x_est, w_est, v_est, p_est, self.problem.value

    def estimate(
        self,
        x0: jnp.ndarray,
        u_seq: jnp.ndarray,
        y_seq: jnp.ndarray,
        prev_est: jnp.ndarray | None = None,
    ) -> MHEResult:
        """Run NMHE for state estimation."""
        start_time = time.time()
        N = self.params.N

        # Initialize trajectory
        if prev_est is None:
            x_est = jnp.tile(x0, (N + 1, 1))
        else:
            x_est = prev_est

        converged = False
        prev_cost = float("inf")

        for sqp_iter in range(self.params.max_sqp_iter):
            if self.params.verbose:
                print(f"NMHE SQP Iteration {sqp_iter + 1}")

            # Linearize dynamics and output
            A_list, B_list, c_list = jax.vmap(self._linearize_dynamics, in_axes=(0, 0, None))(
                x_est[:-1], u_seq, self.params.dt
            )
            H_list, h_list = jax.vmap(self._linearize_output)(x_est[:-1])

            # Solve QP
            x_new, w_new, v_new, p_new, cost = self._solve_qp(x0, u_seq, y_seq, A_list, B_list, c_list, H_list, h_list)

            # Update estimates
            x_est, w_est, v_est, p_est = x_new, w_new, v_new, p_new

            # Convergence check
            if prev_cost == float("inf"):
                prev_cost = cost
            if abs(prev_cost - cost) / (abs(prev_cost) + 1e-9) < self.params.sqp_tol:
                converged = True
                if self.params.verbose:
                    print(f"NMHE converged in {sqp_iter + 1} iterations")
                break
            prev_cost = cost

        return MHEResult(
            x_est=x_est,
            w_est=w_est,
            v_est=v_est,
            p_est=p_est,
            sqp_iters=sqp_iter + 1,
            solve_time=time.time() - start_time,
            converged=converged,
            cost=prev_cost,
        )
