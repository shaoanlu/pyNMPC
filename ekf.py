from typing import Callable
import numpy as np
import jax
from dataclasses import dataclass


@dataclass(kw_only=True)
class EKFParams:
    """Parameters for the Extended Kalman Filter.

    This dataclass contains the configuration parameters needed for
    initializing an Extended Kalman Filter, making the initialization process
    more organized and easier to validate.

    Attributes:
        dt: Time step size in seconds.
        process_noise_cov: Process noise covariance matrix (Q).
            Should be an n_x × n_x matrix where n_x is the state dimension.
        measurement_noise_cov: Measurement noise covariance matrix (R).
            Should be an n_z × n_z matrix where n_z is the measurement dimension.
        initial_state: Initial state estimate vector with length n_x.
        initial_cov: Initial state covariance matrix with shape (n_x, n_x).
    """

    dt: float
    process_noise_cov: np.ndarray  # Q matrix
    measurement_noise_cov: np.ndarray  # R matrix
    initial_state: np.ndarray
    initial_cov: np.ndarray

    def __post_init__(self) -> None:
        """Validate dimensions of the provided matrices."""
        state_dim = len(self.initial_state)

        # Check process noise covariance
        if self.process_noise_cov.shape != (state_dim, state_dim):
            raise ValueError(
                f"Process noise covariance matrix should be {state_dim}x{state_dim}, "
                f"got {self.process_noise_cov.shape}"
            )

        # Check initial covariance
        if self.initial_cov.shape != (state_dim, state_dim):
            raise ValueError(
                f"Initial covariance matrix should be {state_dim}x{state_dim}, " f"got {self.initial_cov.shape}"
            )


class ExtendedKalmanFilter:
    """Extended Kalman Filter  for nonlinear state estimation.

    This class implements the Extended Kalman Filter, which extends the linear Kalman Filter
    to nonlinear systems by linearizing the dynamics and observation models using Jacobians.

    -----
    System model:
        State transition (dynamics):
            x_k = f(x_{k-1}, u_{k-1}) + w_{k-1},   where  w_{k-1} ~ N(0, Q_{k-1})

        Measurement (observation):
            z_k = h(x_k) + v_k,   where  v_k ~ N(0, R_k)

    -----
    Notation:
        x_k : State vector at time k
        u_k : Control input at time k
        z_k : Observation (measurement) at time k
        f(.) : Nonlinear state transition function
        h(.) : Nonlinear measurement function
        w_k : Process noise (zero-mean Gaussian)
        v_k : Measurement noise (zero-mean Gaussian)
        Q_k : Process noise covariance matrix
        R_k : Measurement noise covariance matrix

    -----
    Linearization:
        Jacobian of the dynamics (state transition):
            F_k = ∂f/∂x evaluated at (x̂_{k-1|k-1}, u_{k-1})

        Jacobian of the observation (measurement):
            H_k = ∂h/∂x evaluated at x̂_{k|k-1}

    -----
    EKF algorithm:

    Prediction step:
        x̂_{k|k-1} = f(x̂_{k-1|k-1}, u_{k-1})
        P_{k|k-1} = F_k P_{k-1|k-1} F_kᵀ + Q_{k-1}

    Update step:
        Innovation:
            y_k = z_k - h(x̂_{k|k-1})
        Innovation covariance:
            S_k = H_k P_{k|k-1} H_kᵀ + R_k
        Kalman gain:
            K_k = P_{k|k-1} H_kᵀ S_k⁻¹
        State update:
            x̂_{k|k} = x̂_{k|k-1} + K_k y_k
        Covariance update:
            P_{k|k} = (I - K_k H_k) P_{k|k-1}
    """

    def __init__(self, dynamics_fn: Callable, output_fn: Callable, params: EKFParams) -> None:
        """Initialize the Extended Kalman Filter.

        Args:
            dynamics_fn: State transition function that computes the next state given current state,
                        control input, and time step. Should have signature:
                        f(x, u, dt) -> next_state
            output_fn: Measurement function that maps state to expected measurements.
                      Should have signature: h(x) -> measurement
            params: Configuration parameters for the EKF
        """
        # Store functions and parameters
        self.dynamics_fn = jax.jit(dynamics_fn)
        self.output_fn = jax.jit(output_fn)
        self.dt = params.dt

        self.Q = params.process_noise_cov
        self.R = params.measurement_noise_cov

        # Store state estimates
        self.x_hat = params.initial_state
        self.P = params.initial_cov
        self.I = np.eye(len(self.x_hat))

        # Calculate Jacobians using JAX automatic differentiation
        self.F_jacobian = jax.jit(jax.jacfwd(self.dynamics_fn, argnums=0))
        self.H_jacobian = jax.jit(jax.jacfwd(self.output_fn))

    def predict(self, u: np.ndarray) -> None:
        """Perform the prediction step of the EKF.

        Updates the state estimate and covariance based on the system dynamics model.

        Args:
            u: Control input vector
        """
        # Predict state
        self.x_hat = np.array(self.dynamics_fn(self.x_hat, u, self.dt))

        # Calculate state transition matrix (F)
        F = np.array(self.F_jacobian(self.x_hat, u, self.dt))

        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q

        # Ensure the covariance matrix stays symmetric and positive definite
        self.P = (self.P + self.P.T) / 2

        # Add small value to diagonal for numerical stability if needed
        min_eig = np.min(np.real(np.linalg.eigvals(self.P)))
        if min_eig < 1e-10:
            self.P += np.eye(len(self.x_hat)) * 1e-10

    def update(self, z: np.ndarray) -> None:
        """Perform the update step of the EKF.

        Updates the state estimate and covariance based on the measurement.

        Args:
            z: Measurement vector
        """
        # Calculate expected measurement
        y_pred = np.array(self.output_fn(self.x_hat))

        # Validate measurement dimension on first update
        if hasattr(self, "_meas_dim_validated") is False:
            meas_dim = len(y_pred)
            if self.R.shape != (meas_dim, meas_dim):
                raise ValueError(
                    f"Measurement noise covariance matrix should be {meas_dim}x{meas_dim}, " f"got {self.R.shape}"
                )
            if len(z) != meas_dim:
                raise ValueError(f"Measurement vector should have length {meas_dim}, got {len(z)}")
            self._meas_dim_validated = True

        # Calculate measurement matrix (H)
        H = np.array(self.H_jacobian(self.x_hat))

        # Calculate innovation and innovation covariance
        y = z - y_pred
        S = H @ self.P @ H.T + self.R

        # Calculate Kalman gain
        # Use pseudo-inverse for better numerical stability
        K = self.P @ H.T @ np.linalg.pinv(S)

        # Update state estimate
        self.x_hat = self.x_hat + K @ y

        # Update covariance using the Joseph form for better numerical stability
        self.P = (self.I - K @ H) @ self.P @ (self.I - K @ H).T + K @ self.R @ K.T

        # Ensure the covariance matrix stays symmetric
        self.P = (self.P + self.P.T) / 2

    def estimate(self, u: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Perform a complete estimation step (predict + update).

        Args:
            u: Control input vector
            z: Measurement vector
        """
        self.predict(u)
        self.update(z)
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """Return the current state estimate."""
        return self.x_hat

    def get_covariance(self) -> np.ndarray:
        """Return the current state estimate covariance."""
        return self.P
