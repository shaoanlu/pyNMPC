import unittest

import jax.numpy as jnp
import numpy as np

from pyNMPC.nmpc import NMPC, MPCParams, MPCResult


class TestMPCParams(unittest.TestCase):
    def setUp(self):
        self.params = MPCParams(
            dt=0.1,
            N=10,
            n_states=2,
            n_controls=1,
            x_ref=jnp.array([1.0, 0.0]),
            Q=jnp.eye(2),
            QN=10 * jnp.eye(2),
            R=jnp.eye(1),
            R_delta=0.1 * jnp.eye(1),
            x_min=jnp.array([-5.0, -5.0]),
            x_max=jnp.array([5.0, 5.0]),
            u_min=jnp.array([-1.0]),
            u_max=jnp.array([1.0]),
        )

    def test_initialization(self):
        self.assertEqual(self.params.dt, 0.1)
        self.assertEqual(self.params.N, 10)
        self.assertEqual(self.params.n_states, 2)
        self.assertEqual(self.params.n_controls, 1)
        self.assertTrue(jnp.allclose(self.params.u_prev, jnp.zeros(1)))


class TestNMPC(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test case."""
        # Define a simple linear system for testing: x_{k+1} = A*x_k + B*u_k
        self.n_states = 2
        self.n_controls = 1
        self.dt = 0.1
        self.N = 5

        # Simple linear dynamics function
        def linear_dynamics(x, u, dt):
            A = jnp.array([[1.0, dt], [0.0, 1.0]])
            B = jnp.array([[0.0], [dt]])
            return A @ x + B @ u

        self.dynamics_fn = linear_dynamics

        # Define MPC parameters
        self.params = MPCParams(
            dt=self.dt,
            N=self.N,
            n_states=self.n_states,
            n_controls=self.n_controls,
            x_ref=jnp.array([1.0, 0.0]),
            Q=jnp.eye(self.n_states),
            QN=jnp.eye(self.n_states) * 10.0,
            R=jnp.eye(self.n_controls) * 0.1,
            R_delta=jnp.eye(self.n_controls) * 0.05,
            x_min=jnp.array([-5.0, -5.0]),
            x_max=jnp.array([5.0, 5.0]),
            u_min=jnp.array([-2.0]),
            u_max=jnp.array([2.0]),
            verbose=False,
        )

        # Create NMPC instance
        self.nmpc = NMPC(self.dynamics_fn, self.params)

    def test_initialization(self):
        """Test that NMPC initializes correctly."""
        self.assertEqual(self.nmpc.n_states, self.n_states)
        self.assertEqual(self.nmpc.n_controls, self.n_controls)
        self.assertEqual(self.nmpc.params.N, self.N)
        self.assertEqual(self.nmpc.params.dt, self.dt)

    def test_qp_problem_construction(self):
        # Check if CVXPY problem is formed correctly and variables initialized
        problem = self.nmpc.problem
        self.assertIsNotNone(problem, "QP problem should be constructed")
        self.assertEqual(len(self.nmpc.x_var), self.params.N + 1, "Incorrect number of state variables")
        self.assertEqual(len(self.nmpc.u_var), self.params.N, "Incorrect number of control variables")

    def test_dynamics_jit_compilation(self):
        x0 = jnp.array([0.0, 0.0])
        u0 = jnp.array([1.0])
        result = self.nmpc.dynamics(x0, u0, self.params.dt)
        expected = x0 + jnp.concatenate([jnp.zeros(1), u0 * self.params.dt])
        self.assertTrue(jnp.allclose(result, expected), f"Dynamics output incorrect: {result}")

    def test_linearize_dynamics(self):
        """Test dynamics linearization."""
        x_nom = jnp.array([0.5, 1.0])
        u_nom = jnp.array([1.0])

        A, B, c = self.nmpc._linearize_dynamics(x_nom, u_nom, self.dt)

        # Expected values for our simple linear system
        A_expected = jnp.array([[1.0, self.dt], [0.0, 1.0]])
        B_expected = jnp.array([[0.0], [self.dt]])

        # For a linear system, c should be zero (no residual)
        c_expected = jnp.array([0.0, 0.0])

        np.testing.assert_allclose(A, A_expected, rtol=1e-5)
        np.testing.assert_allclose(B, B_expected, rtol=1e-5)
        np.testing.assert_allclose(c, c_expected, rtol=1e-5, atol=1e-5)

    def test_linearize_trajectory(self):
        """Test trajectory linearization."""
        N = self.params.N
        x_traj = jnp.ones((N + 1, self.n_states))
        u_traj = jnp.ones((N, self.n_controls))

        A_list, B_list, c_list = self.nmpc._linearize_trajectory(x_traj, u_traj, self.dt)

        # Check shapes
        self.assertEqual(len(A_list), N)
        self.assertEqual(len(B_list), N)
        self.assertEqual(len(c_list), N)

        # Check values for first element
        A_expected = jnp.array([[1.0, self.dt], [0.0, 1.0]])
        B_expected = jnp.array([[0.0], [self.dt]])

        np.testing.assert_allclose(A_list[0], A_expected, rtol=1e-5)
        np.testing.assert_allclose(B_list[0], B_expected, rtol=1e-5)

    def test_stage_cost(self):
        """Test computation of stage cost."""
        x = jnp.array([0.5, 0.5])
        u = jnp.array([1.0])
        x_ref = jnp.array([1.0, 0.0])

        cost = self.nmpc._stage_cost(x, u, x_ref)

        # Expected cost: (x-x_ref)^T * Q * (x-x_ref) + u^T * R * u
        expected_cost = 0.5**2 + 0.5**2 + 0.1 * 1.0**2

        self.assertAlmostEqual(cost, expected_cost, places=5)

    def test_solve_with_no_constraints(self):
        """Test MPC solving with no state or input constraints."""
        # Create unconstrained parameters
        unconstrained_params = MPCParams(
            dt=1.0,
            N=self.N,
            n_states=self.n_states,
            n_controls=self.n_controls,
            x_ref=jnp.array([0.1, 0.0]),
            Q=jnp.eye(self.n_states),
            QN=jnp.eye(self.n_states) * 10.0,
            R=jnp.eye(self.n_controls) * 0.1,
            verbose=False,
        )

        nmpc_unconstrained = NMPC(self.dynamics_fn, unconstrained_params)

        # Initial state
        x0 = jnp.array([0.0, 0.0])

        # Solve MPC problem
        result = nmpc_unconstrained.solve(x0)

        # Check results
        self.assertIsInstance(result, MPCResult)
        self.assertEqual(result.x_traj.shape, (self.N + 1, self.n_states))
        self.assertEqual(result.u_traj.shape, (self.N, self.n_controls))
        self.assertTrue(result.converged)

        # The solution should drive the state towards the reference
        final_state = result.x_traj[-1]
        np.testing.assert_allclose(
            actual=final_state,
            desired=unconstrained_params.x_ref,  # Expect to get reasonably close to reference
            atol=0.01,
            err_msg=f"Final state: {result.x_traj=}",
        )

    def test_solve_with_constraints(self):
        """Test MPC solving with state and input constraints."""
        # Initial state
        x0 = jnp.array([0.0, 0.0])

        # Solve MPC problem with constraints
        result = self.nmpc.solve(x0)

        # Check results
        self.assertIsInstance(result, MPCResult)
        self.assertTrue(result.converged)

        # Check if state constraints are satisfied
        for x in result.x_traj:
            np.testing.assert_array_less(x, self.params.x_max + 1e-3)
            np.testing.assert_array_less(self.params.x_min - 1e-3, x)

        # Check if input constraints are satisfied
        for u in result.u_traj:
            np.testing.assert_array_less(u, self.params.u_max + 1e-3)
            np.testing.assert_array_less(self.params.u_min - 1e-3, u)

    def test_warm_start(self):
        """Test MPC solving with warm start from previous solution."""
        # Initial state
        x0 = jnp.array([0.0, 0.0])

        # First solve without warm start
        result1 = self.nmpc.solve(x0)

        # Second solve with warm start
        result2 = self.nmpc.solve(x0, mpc_result=result1)

        # Check results
        self.assertTrue(result2.converged)
        self.assertLessEqual(result2.sqp_iters, result1.sqp_iters)  # Should converge in fewer iterations
        self.assertLessEqual(result2.solve_time, result1.solve_time * 1.5)  # Should be faster (with some margin)

    def test_update_reference(self):
        """Test updating the reference state."""
        # Initial state
        x0 = jnp.array([0.0, 0.0])

        # New reference
        x_ref_new = jnp.array([2.0, 1.0])

        # Solve with new reference
        result = self.nmpc.solve(x0, x_ref=x_ref_new)

        # Check if the reference was updated
        np.testing.assert_array_equal(self.nmpc.params.x_ref, x_ref_new)

        # Final state should move towards the new reference
        final_state = result.x_traj[-1]
        # Direction check: should move in positive direction for both states
        self.assertGreater(final_state[0], x0[0])
        self.assertGreater(final_state[1], x0[1])

    def test_nonlinear_dynamics(self):
        """Test with a nonlinear dynamics function."""

        # Define a simple nonlinear system
        def nonlinear_dynamics(x, u, dt):
            # Simple pendulum dynamics x = [theta, theta_dot]
            g = 9.81  # gravity
            l = 1.0  # pendulum length

            theta = x[0]
            theta_dot = x[1]

            # Simplified dynamics (small angle approximation with control)
            theta_ddot = -g / l * jnp.sin(theta) + u[0] / l

            # Euler integration
            theta_new = theta + theta_dot * dt
            theta_dot_new = theta_dot + theta_ddot * dt

            return jnp.array([theta_new, theta_dot_new])

        # Set up parameters for nonlinear system
        nl_params = MPCParams(
            dt=self.dt,
            N=self.N,
            n_states=self.n_states,
            n_controls=self.n_controls,
            x_ref=jnp.array([0.0, 0.0]),  # Stabilize at upright position
            Q=jnp.eye(self.n_states),
            QN=jnp.eye(self.n_states) * 10.0,
            R=jnp.eye(self.n_controls) * 0.1,
            x_min=jnp.array([-jnp.pi / 2, -5.0]),
            x_max=jnp.array([jnp.pi / 2, 5.0]),
            u_min=jnp.array([-5.0]),
            u_max=jnp.array([5.0]),
            verbose=False,
        )

        nl_nmpc = NMPC(nonlinear_dynamics, nl_params)

        # Initial state - pendulum at slight angle
        x0 = jnp.array([0.3, 0.0])

        # Solve MPC problem
        result = nl_nmpc.solve(x0)

        # Check results
        self.assertTrue(result.converged)

        # Final state should move towards upright position
        final_state = result.x_traj[-1]
        self.assertLess(jnp.abs(final_state[0]), jnp.abs(x0[0]))  # Angle should decrease

    def test_soft_constraints(self):
        """Test MPC with soft state constraints."""
        # Set up parameters with soft constraints
        soft_params = MPCParams(
            dt=self.dt,
            N=self.N,
            n_states=self.n_states,
            n_controls=self.n_controls,
            x_ref=jnp.array([1.0, 0.0]),
            Q=jnp.eye(self.n_states),
            QN=jnp.eye(self.n_states) * 10.0,
            R=jnp.eye(self.n_controls) * 0.1,
            x_min=jnp.array([-0.1, -5.0]),  # Very tight x position constraint
            x_max=jnp.array([0.1, 5.0]),
            u_min=jnp.array([-2.0]),
            u_max=jnp.array([2.0]),
            use_soft_constraint=True,
            slack_weight=1e2,
            verbose=False,
        )

        soft_nmpc = NMPC(self.dynamics_fn, soft_params)

        # Initial state
        x0 = jnp.array([0.2, 0.0])

        # Solve MPC problem with soft constraints
        result = soft_nmpc.solve(x0)

        # Check results
        self.assertTrue(result.converged)
        self.assertIsNotNone(result.slack)  # Slack variables should be returned

        # Check if slack variables are used (some should be non-zero due to x0 out of bounds)
        self.assertTrue(jnp.any(jnp.array(result.slack) > 1e-6))

    def test_control_delta_cost(self):
        """Test MPC with control rate-of-change penalty."""
        # Set fixed previous control
        u_prev = jnp.array([1.0])

        # Initial state
        x0 = jnp.array([0.0, 0.0])

        # Solve with specified previous control
        result = self.nmpc.solve(x0, u_prev=u_prev)

        # First control should be closer to u_prev than it would be without R_delta
        first_u = result.u_traj[0]
        self.assertLess(jnp.abs(first_u - u_prev), 1.0)  # Some smoothing should occur


if __name__ == "__main__":
    unittest.main()
