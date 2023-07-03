import math
import numpy as np

from gym import logger
from gym.envs.classic_control import CartPoleEnv


class BaseCartPole(CartPoleEnv):
    """Base CartPole class

    Define function for sampling states
    Termination condition is overridable
    Define function for custom reset
    Modified reward to be correct
    """

    def sample_states(self, n_states: int) -> np.ndarray:
        states = np.random.uniform(
            low=(-4.8, -4.0, -0.4, -3.0),
            high=(4.8, 4.0, 0.4, 3.0),
            size=((n_states, 4)),
        )
        return states

    def is_done(self, states: np.ndarray) -> np.ndarray:
        is_safe = (
            (states[..., 0] > -self.x_threshold)
            * (states[..., 0] < self.x_threshold)
            * (states[..., 2] > -self.theta_threshold_radians)
            * (states[..., 2] < self.theta_threshold_radians)
        )
        return ~is_safe

    def reset_to(self, state: np.ndarray):
        self.reset()
        self.state = state
        return state

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        # Changed this line
        done = self.is_done(np.array(self.state)).item()

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            # Changed this line too, from 1 to 0
            reward = 0.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, done, {}


class CartPoleAEnv(BaseCartPole):
    def is_done(self, states: np.ndarray) -> np.ndarray:
        is_safe = (states[..., 0] > -self.x_threshold) * (
            states[..., 0] < self.x_threshold
        )
        return ~is_safe


class CartPoleBEnv(BaseCartPole):
    def is_done(self, states: np.ndarray) -> np.ndarray:
        is_safe = (states[..., 2] > -self.theta_threshold_radians) * (
            states[..., 2] < self.theta_threshold_radians
        )
        return ~is_safe


class CartPoleCEnv(BaseCartPole):
    def is_done(self, states: np.ndarray) -> np.ndarray:
        is_safe = (
            (states[..., 0] > -self.x_threshold)
            * (states[..., 2] > -self.theta_threshold_radians)
            *
            # Just here to prevent overflow...
            (states[..., 0] < 1e6)
            * (states[..., 2] < 1e6)
        )
        return ~is_safe


class CartPoleDEnv(BaseCartPole):
    def is_done(self, states: np.ndarray) -> np.ndarray:
        is_safe = (
            (states[..., 0] < self.x_threshold)
            * (states[..., 2] < self.theta_threshold_radians)
            *
            # Just here to prevent underflow...
            (states[..., 0] > -1e6)
            * (states[..., 2] > -1e6)
        )
        return ~is_safe
