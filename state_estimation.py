import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la


class EKF:
    def __init__(self):
        self.state_posterior = np.array([[0., 0.1, np.deg2rad(30), np.deg2rad(2)]]).T
        self.gt = np.zeros_like(self.state_posterior)
        self.gt_prev = np.array([[0., 0.1, np.deg2rad(30), np.deg2rad(2)]]).T
        print(self.state_posterior.shape)
        self.state_prior = np.zeros_like(self.state_posterior)
        self.u = np.zeros((2, 1))
        self.meas = np.zeros_like(self.state_posterior)
        self.cov_prior = np.zeros(4)
        self.cov_posterior = np.zeros(4)
        # self.Q = np.diag([0.1, 0.3])  # covariance of measurement noise (Q)
        self.Q = np.diag([0.1, 0.3])  # covariance of measurement noise (Q)
        self.R = 100*np.diag([0.01,0.01,0.01, 0.01])

        # Params
        self.delta_t = 0.1
        self.duration = 10.0
        self.L = 1.0

        self.gt_history = None
        self.z_history = None
        self.estimated_history = None

    @staticmethod
    def input(t):
        u = np.zeros((2, 1))
        u[0] = 10 * np.sin(t)
        u[1] = 0.01
        return u

    @staticmethod
    def motion_model(x_prev, u, delta_t, L, noise=None):
        x_t = np.zeros_like(x_prev)
        x_t[0, 0] = x_prev[0] + delta_t * u[0] * np.cos(x_prev[2])
        x_t[1, 0] = x_prev[1] + delta_t * u[0] * np.sin(x_prev[2])
        x_t[2, 0] = x_prev[2] + delta_t * u[0] * np.tan(x_prev[3]) / L
        x_t[3, 0] = x_prev[3] + delta_t * u[1]
        if noise is None:
            return x_t
        else:
            return x_t + noise

    @staticmethod
    def measure_model(state, noise):
        x = state[0]
        y = state[1]
        z = np.zeros((2, 1))
        z[0] = np.sqrt(x * x + y * y)
        z[1] = np.arctan2(y, x)
        return z + noise

    @staticmethod
    def G(state, u, delta_t, L):
        # Jacobian of G was determined analytically
        G = np.eye(4)
        G[0, 2] = -delta_t * u[0] * np.sin(state[2])
        G[1, 2] = delta_t * u[0] * np.cos(state[2])
        G[2, 3] = delta_t * u[0] / (L * np.cos(state[2]) * np.cos(state[2]))
        return G

    @staticmethod
    def H(prior):
        # Jacobian of H was determined analytically (identity matrix)
        H = np.zeros((2, 4))
        x = prior[0, 0]
        y = prior[1, 0]
        H[0, 0] = x / np.sqrt(x * x + y * y)
        H[0, 1] = y / np.sqrt(x * x + y * y)

        H[1, 0] = -y / (x * x + y * y)
        H[1, 1] = x / (x * x + y * y)
        return H

    @staticmethod
    def ground_truth(r, phi):
        gt = np.zeros((2,1))
        gt[0] = r*np.cos(phi)
        gt[1] = r * np.sin(phi)
        return gt

    # Run pose estimation
    def run(self):
        for step in np.arange(0, self.duration, self.delta_t):
            self.u = self.input(step)

            self.gt = self.motion_model(self.gt_prev, self.u, self.delta_t, self.L)
            if self.gt_history is None:
                self.gt_history = self.gt
            else:
                self.gt_history = np.append(self.gt_history, self.gt, axis=1)
            self.gt_prev = self.gt
            """
            Step 1
            """
            process_noise = np.zeros_like(self.state_prior)
            process_noise[0,0] = np.random.normal(loc=0, scale=np.sqrt(self.R[0, 0]))
            process_noise[1,0] = np.random.normal(loc=0, scale=np.sqrt(self.R[1, 1]))
            process_noise[2,0] = np.random.normal(loc=0, scale=np.sqrt(self.R[2, 2]))
            process_noise[3,0] = np.random.normal(loc=0, scale=np.sqrt(self.R[3, 3]))
            self.state_prior = self.motion_model(self.state_posterior, self.u, self.delta_t, self.L)
            """
            Step 2
            """
            G = self.G(self.state_posterior, self.u, self.delta_t, self.L)  # dim: 4x4a
            self.cov_prior = np.dot(G, np.dot(self.cov_prior, G.T)) + self.R

            """
            Step 3
            """
            H = self.H(self.state_prior)  # 2x4
            _inv = np.dot(np.dot(H, self.cov_prior), H.T) + self.Q
            inv = la.inv(_inv)
            K = np.dot(np.dot(self.cov_prior, H.T), inv)  # K.shape = (4,2)
            assert (K.shape == (4, 2)), "Dimension of K incorrect"

            """
            Step 4
            """
            # USE GT?
            meas_noise = np.zeros((2,1))
            h = self.measure_model(self.state_prior, meas_noise)  # no noise injected when calculation with our prior

            meas_noise[0,0] = np.random.normal(loc=0, scale=np.sqrt(self.Q[0, 0]))
            meas_noise[1,0] = np.random.normal(loc=0, scale=np.sqrt(self.Q[1, 1]))
            z = self.measure_model(self.gt, meas_noise)  # Noise is only injected when simulating the measurement
            if self.z_history is None:
                self.z_history = z
            else:
                self.z_history = np.append(self.z_history, z, axis=1)

            assert (z.shape == h.shape), "z and h does not share the same dimensions"
            innovation = z - h
            self.state_posterior = self.state_prior + np.dot(K, innovation)  # output dim: 4x1

            """
            Step 5
            """
            self.cov_posterior = np.dot((np.eye(4) - np.dot(K, H)), self.cov_prior)  # output dim: 4x4

            # Display pose estimate
            print("state: {}".format(self.state_posterior))
            if self.estimated_history is None:
                self.estimated_history = self.state_posterior
            else:
                self.estimated_history = np.append(self.estimated_history, self.state_posterior, axis=1)

            assert (self.state_prior.shape == self.state_posterior.shape), "State dimensions do not match"


if __name__ == '__main__':
    try:
        ekf = EKF()
        ekf.run()
        # Plotting
        t = np.arange(0, 10, 0.1)
        fig = plt.figure(figsize=(24, 12))
        fig.add_subplot(3, 2, 1)
        plt.plot(t, ekf.gt_history[0], 'bo')
        plt.plot(t, ekf.z_history[0]*np.cos(ekf.z_history[1]), 'r+')
        plt.plot(t, ekf.estimated_history[0], 'g-')
        plt.legend(["Ground Truth", "Measurement", "Estimated"])
        plt.ylabel("x[m]")
        plt.xlabel("t[s]")
        plt.title("X")
        fig.add_subplot(3, 2, 2)
        plt.plot(t, ekf.gt_history[1], 'bo')
        plt.plot(t, ekf.z_history[0]*np.sin(ekf.z_history[1]), 'r+')
        plt.plot(t, ekf.estimated_history[1], 'g-')
        plt.legend(["Ground Truth", "Measurement", "Estimated"])
        plt.ylabel("y[m]")
        plt.xlabel("t[s]")
        plt.title("Y")
        fig.add_subplot(3, 2, 3)
        plt.plot(t, ekf.gt_history[2], 'bo')
        plt.plot(t, ekf.estimated_history[2], 'g-')
        plt.legend(["Ground Truth", "Estimated"])  # No measurement available
        plt.ylabel("theta[rad]")
        plt.xlabel("t[s]")
        plt.title("Theta")
        fig.add_subplot(3, 2, 4)
        plt.plot(t, ekf.gt_history[3], 'bo')
        plt.plot(t, ekf.estimated_history[3], 'g-')
        plt.legend(["Ground Truth", "Estimated"])  # No measurement available
        plt.ylabel("delta[rad]")
        plt.xlabel("t[s]")
        plt.title("Delta")
        fig.add_subplot(3, 2, 5)
        plt.plot(ekf.gt_history[0], ekf.gt_history[1], "bo")
        plt.plot(ekf.estimated_history[0], ekf.estimated_history[1], "go")
        plt.legend(["Ground Truth", "Estimated"])  # No measurement available
        plt.ylabel("y[m]")
        plt.xlabel("x[m]")
        plt.title("XY")
        plt.suptitle(f"State Estimation Result with R = diag({ekf.R[0,0], ekf.R[1,1], ekf.R[2,2]})")
        plt.show()
        fig.savefig("plots/problem2_with_process_noise_large.eps")
    except KeyboardInterrupt:
        exit(0)
