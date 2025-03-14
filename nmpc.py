import jax
import jax.numpy as jnp
import numpy as np
import cvxpy as cp
import time
from functools import partial
from typing import Tuple, List, Callable, Any, Dict
from dataclasses import dataclass, field


@dataclass(kw_only=True)
class MPCParams:
    """Parameters for the Model Predictive Control (MPC) problem.

    This dataclass contains all configuration parameters needed for the nonlinear MPC solver.

    Attributes:
        dt (float): Time step for discretization (seconds).
        N (int): Prediction horizon length (number of steps).
        n_states (int): Dimension of the state vector.
        n_controls (int): Dimension of the control input vector.
        Q (jnp.ndarray): State cost matrix of shape (n_states, n_states).
        QN (jnp.ndarray): Terminal state cost matrix of shape (n_states, n_states).
        R (jnp.ndarray): Control cost matrix of shape (n_controls, n_controls).
        x_ref (jnp.ndarray): Reference/target state of shape (n_states,).
        x_min (jnp.ndarray): Lower bounds for state of shape (n_states,).
        x_max (jnp.ndarray): Upper bounds for state of shape (n_states,).
        u_min (jnp.ndarray): Lower bounds for control input of shape (n_controls,).
        u_max (jnp.ndarray): Upper bounds for control input of shape (n_controls,).
        slack_weight (float): slack penalty for soft state constraints.
        max_sqp_iter (int): Maximum number of SQP iterations.
        sqp_tol (float): Tolerance for SQP convergence.
        verbose (bool): Whether to print verbose output during optimization.
    """

    dt: float
    N: int
    n_states: int
    n_controls: int
    Q: jnp.ndarray
    QN: jnp.ndarray
    R: jnp.ndarray
    x_ref: jnp.ndarray
    x_min: jnp.ndarray | None = None
    x_max: jnp.ndarray | None = None
    u_min: jnp.ndarray | None = None
    u_max: jnp.ndarray | None = None
    slack_weight: float = 1e4
    use_soft_constraint: bool = True
    max_sqp_iter: int = 5
    sqp_tol: float = 1e-4
    verbose: bool = False


@dataclass(frozen=True)
class MPCResult:
    """Results from the MPC optimization.

    This dataclass stores the output of the NMPC solver, including the optimal trajectories
    and metadata about the optimization process.

    Attributes:
        x_traj (jnp.ndarray): Optimal state trajectory of shape (N+1, n_states).
        u_traj (jnp.ndarray): Optimal control input trajectory of shape (N, n_controls).
        sqp_iters (int): Number of SQP iterations performed.
        solve_time (float): Total time taken to solve the MPC problem (seconds).
        converged (bool): Whether the SQP algorithm converged within the maximum iterations.
        cost (float): Final cost value of the optimal solution.
    """

    x_traj: jnp.ndarray
    u_traj: jnp.ndarray
    slack: jnp.ndarray | None
    sqp_iters: int
    solve_time: float
    converged: bool
    cost: float


class NMPC:
    """Nonlinear Model Predictive Control (NMPC) solver using Sequential Quadratic Programming (SQP).

    This class implements a nonlinear MPC solver that iteratively linearizes the system dynamics
    and solves a sequence of quadratic programming (QP) subproblems. It uses JAX for efficient
    automatic differentiation and CVXPY for solving the QP subproblems.

    Args:
        dynamics_fn (Callable): System dynamics function with signature f(x, u) -> x_next,
                               where x is the state and u is the control input.
        params (MPCParams): Configuration parameters for the MPC problem.
        params_dict (Dict[str, Any], optional): Additional parameters for extensions like
                                              Control Barrier Functions (CBFs).

    Attributes:
        dynamics (Callable): JIT-compiled system dynamics function.
            The function shuold accept args as func(x, u, dt) and return a new state vector.
        params (MPCParams): MPC configuration parameters.
        params_dict (Dict[str, Any]): Additional parameters for extensions.
        n_states (int): Dimension of the state vector.
        n_controls (int): Dimension of the control input vector.
        problem (cp.Problem): Parameterized QP problem instance.
        x_var (List[cp.Variable]): State variables for the QP.
        u_var (List[cp.Variable]): Control variables for the QP.
        parameter_dict (Dict): Dictionary of parameters for the QP.
    """

    def __init__(
        self,
        dynamics_fn: Callable,
        params: MPCParams,
        params_dict: Dict[str, Any] | None = None,
        solver_opts: Dict[str, Any] | None = None,
    ):
        self.dynamics = jax.jit(dynamics_fn, static_argnums=(2,))
        self.params = params
        self.params_dict = params_dict if params_dict else {}
        self.n_states = params.n_states
        self.n_controls = params.n_controls

        if solver_opts is None:
            self.solver_opts = {
                "solver": cp.OSQP,
                "adaptive_rho": True,
                "eps_abs": 1e-3,
                "eps_rel": 1e-3,
                "max_iter": 1000,
                "warm_start": True,
                "verbose": False,
            }
        else:
            self.solver_opts = solver_opts

        self.problem, self.x_var, self.u_var, self.s_var, self.parameter_dict = self._create_parameterized_qp(params)

    @partial(jax.jit, static_argnums=(0,))
    def _stage_cost(self, x: jnp.ndarray, u: jnp.ndarray, x_ref: jnp.ndarray) -> float:
        """Compute the quadratic stage cost at a single time step.

        Args:
            x (jnp.ndarray): Current state of shape (n_states,).
            u (jnp.ndarray): Current control input of shape (n_controls,).
            x_ref (jnp.ndarray): Reference state of shape (n_states,).

        Returns:
            float: Quadratic cost value combining state tracking error and control effort.
        """
        state_error = x - x_ref
        return jnp.dot(state_error, self.params.Q @ state_error) + jnp.dot(u, self.params.R @ u)

    @partial(jax.jit, static_argnums=(0, 3))
    def _linearize_dynamics(
        self, x_nom: jnp.ndarray, u_nom: jnp.ndarray, dt: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Linearize the system dynamics around a nominal state and control input.

        Uses JAX's automatic differentiation to compute the Jacobians of the dynamics
        with respect to the state and control input.

        Args:
            x_nom (jnp.ndarray): Nominal state of shape (n_states,).
            u_nom (jnp.ndarray): Nominal control input of shape (n_controls,).

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
                - A: State Jacobian matrix of shape (n_states, n_states).
                - B: Control Jacobian matrix of shape (n_states, n_controls).
                - c: Affine term of shape (n_states,) for the linearized dynamics.
        """
        dyn_fixed_dt = lambda x, u: self.dynamics(x, u, dt)

        A = jax.jacfwd(lambda x: dyn_fixed_dt(x, u_nom))(x_nom)
        B = jax.jacfwd(lambda u: dyn_fixed_dt(x_nom, u))(u_nom)
        c = dyn_fixed_dt(x_nom, u_nom) - A @ x_nom - B @ u_nom  # residual

        return A, B, c

    @partial(jax.jit, static_argnums=(0, 3))
    def _linearize_trajectory(
        self, x_traj: jnp.ndarray, u_traj: jnp.ndarray, dt: float
    ) -> Tuple[List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray]]:
        """Linearize system dynamics at all points along a trajectory in parallel.

        Uses JAX's vectorized mapping (vmap) to efficiently compute linearizations
        at multiple points simultaneously.

        Args:
            x_traj (jnp.ndarray): State trajectory of shape (N+1, n_states).
            u_traj (jnp.ndarray): Control input trajectory of shape (N, n_controls).

        Returns:
            Tuple[List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray]]:
                - A_list: List of state Jacobian matrices.
                - B_list: List of control Jacobian matrices.
                - c_list: List of affine terms.
        """
        return jax.vmap(lambda x, u: self._linearize_dynamics(x, u, dt))(x_traj[:-1], u_traj)

    def _create_parameterized_qp(
        self, params: MPCParams
    ) -> Tuple[cp.Problem, List[cp.Variable], List[cp.Variable], Dict]:
        """Create a parameterized QP problem for reuse across SQP iterations.

        Builds the CVXPY optimization problem with all constraints and cost functions.
        The problem is parameterized to allow efficient warm-starting in subsequent SQP iterations.

        Returns:
            Tuple[cp.Problem, List[cp.Variable], List[cp.Variable], Dict]:
                - problem: The CVXPY optimization problem.
                - x_var: List of state variables.
                - u_var: List of control variables.
                - parameter_dict: Dictionary of parameters for the QP.
        """
        N, n_states, n_controls = params.N, params.n_states, params.n_controls

        # Decision variables
        x_var = [cp.Variable(n_states) for _ in range(N + 1)]
        u_var = [cp.Variable(n_controls) for _ in range(N)]
        if params.use_soft_constraint:
            slack_var = [cp.Variable(n_states, nonneg=True) for _ in range(N + 1)]

        # Parameters
        x0_param = cp.Parameter(n_states)
        x_ref_param = cp.Parameter(n_states)
        A_params = [cp.Parameter((n_states, n_states)) for _ in range(N)]
        B_params = [cp.Parameter((n_states, n_controls)) for _ in range(N)]
        c_params = [cp.Parameter(n_states) for _ in range(N)]

        # Initial state constraint
        constraints = [x_var[0] == x0_param]
        cost = 0

        # Process constraints and cost
        for k in range(N):
            # Stage cost
            error_k = x_var[k] - x_ref_param
            cost += cp.quad_form(error_k, params.Q) + cp.quad_form(u_var[k], params.R)

            # Dynamics constraint
            constraints.append(x_var[k + 1] == A_params[k] @ x_var[k] + B_params[k] @ u_var[k] + c_params[k])

            # State constraints
            if params.use_soft_constraint:
                if params.x_min is not None:
                    constraints.append(x_var[k] >= params.x_min - slack_var[k])
                if params.x_max is not None:
                    constraints.append(x_var[k] <= params.x_max + slack_var[k])
                cost += params.slack_weight * cp.sum_squares(slack_var[k])
            else:
                if params.x_min is not None:
                    constraints.append(x_var[k] >= params.x_min)
                if params.x_max is not None:
                    constraints.append(x_var[k] <= params.x_max)

            # Input constraints
            if params.u_max is not None:
                constraints.append(u_var[k] <= params.u_max)
            if params.u_min is not None:
                constraints.append(u_var[k] >= params.u_min)

        # Terminal cost
        cost += cp.quad_form(x_var[N] - x_ref_param, params.QN)

        # Create problem
        problem = cp.Problem(cp.Minimize(cost), constraints)

        # Parameter dictionary
        parameter_dict = {
            "x0": x0_param,
            "A_list": A_params,
            "B_list": B_params,
            "c_list": c_params,
            "x_ref": x_ref_param,
        }

        return problem, x_var, u_var, slack_var, parameter_dict

    def _solve_qp_subproblem(
        self,
        x0: jnp.ndarray,
        x_traj: jnp.ndarray,
        u_traj: jnp.ndarray,
        A_list: List[jnp.ndarray],
        B_list: List[jnp.ndarray],
        c_list: List[jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """Solve a single QP subproblem within the SQP algorithm.

        Updates the parameters of the parameterized QP problem and solves it
        using CVXPY.

        Args:
            x0 (jnp.ndarray): Initial state of shape (n_states,).
            x_traj (jnp.ndarray): Current state trajectory of shape (N+1, n_states).
            u_traj (jnp.ndarray): Current control trajectory of shape (N, n_controls).
            A_list (List[jnp.ndarray]): List of state Jacobian matrices.
            B_list (List[jnp.ndarray]): List of control Jacobian matrices.
            c_list (List[jnp.ndarray]): List of affine terms.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray, float]:
                - New state trajectory.
                - New control trajectory.
                - Final cost value.
        """
        # Convert to numpy
        x0_np = np.array(x0)
        A_list_np = [np.array(A) for A in A_list]
        B_list_np = [np.array(B) for B in B_list]
        c_list_np = [np.array(c) for c in c_list]

        # Update parameters
        self.parameter_dict["x0"].value = x0_np
        self.parameter_dict["x_ref"].value = np.array(self.params.x_ref)
        for k in range(self.params.N):
            self.parameter_dict["A_list"][k].value = A_list_np[k]
            self.parameter_dict["B_list"][k].value = B_list_np[k]
            self.parameter_dict["c_list"][k].value = c_list_np[k]

        # Solve QP
        self.problem.solve(**self.solver_opts)

        # Check solution
        if self.problem.status not in ["optimal", "optimal_inaccurate"]:
            return x_traj, u_traj, float("inf")

        # Extract solution
        x_new = jnp.array([self.x_var[k].value for k in range(self.params.N + 1)])
        u_new = jnp.array([self.u_var[k].value for k in range(self.params.N)])
        s_new = None
        if hasattr(self, "slack_var"):
            s_new = jnp.array([self.s_var[k].value for k in range(self.params.N + 1)])

        return x_new, u_new, s_new, self.problem.value

    def solve(
        self, x0: jnp.ndarray, x_ref: jnp.ndarray | None = None, mpc_result: MPCResult | None = None
    ) -> MPCResult:
        """Solve the NMPC problem using Sequential Quadratic Programming.

        This is the main entry point for solving the nonlinear MPC problem.
        It iteratively linearizes the dynamics and solves QP subproblems until
        convergence or the maximum number of iterations is reached.

        Args:
            x0 (jnp.ndarray): Initial state of shape (n_states,).
            mpc_result (MPCResult, optional): Previous MPC result for warm-starting.
                If None, a simple heuristic initialization is used.

        Returns:
            MPCResult: Object containing the optimal trajectories and solver metadata.
                - x_traj: Optimal state trajectory.
                - u_traj: Optimal control input trajectory.
                - sqp_iters: Number of SQP iterations performed.
                - solve_time: Total solve time in seconds.
                - converged: Whether the algorithm converged.
                - cost: Final cost value.
        """
        start_time = time.time()
        N = self.params.N

        if x_ref is not None:
            self.params.x_ref = x_ref

        # Initialize trajectories
        if mpc_result is None:
            x_traj = jnp.zeros((N + 1, self.n_states))
            for i in range(N + 1):
                alpha = min(i / N, 1.0)
                x_traj = x_traj.at[i].set((1 - alpha) * x0 + alpha * self.params.x_ref)
            u_traj = jnp.zeros((N, self.n_controls))
        else:
            x_traj, u_traj = mpc_result.x_traj, mpc_result.u_traj

        # SQP iterations
        prev_cost = float("inf")
        converged = False

        for sqp_iter in range(self.params.max_sqp_iter):
            if self.params.verbose:
                print(f"SQP Iteration {sqp_iter + 1}")

            # Linearize dynamics
            A_list, B_list, c_list = self._linearize_trajectory(x_traj, u_traj, self.params.dt)

            # Solve QP subproblem
            x_new, u_new, s_new, cost = self._solve_qp_subproblem(x0, x_traj, u_traj, A_list, B_list, c_list)

            # Check convergence
            if prev_cost == float("inf"):
                prev_cost = cost
            cost_change = abs(prev_cost - cost) / (abs(prev_cost) + 1e-9)
            if cost_change < self.params.sqp_tol and sqp_iter > 0:
                converged = True
                if self.params.verbose:
                    print(f"SQP converged after {sqp_iter + 1} iterations")
                break

            # Update trajectories
            x_traj, u_traj, slack, prev_cost = x_new, u_new, s_new, cost

        return MPCResult(
            x_traj=x_traj,
            u_traj=u_traj,
            slack=slack,
            sqp_iters=sqp_iter + 1,
            solve_time=time.time() - start_time,
            converged=converged,
            cost=prev_cost,
        )
