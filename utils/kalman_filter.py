import numpy as np
import scipy.linalg


class KalmanFilter3D(object):
   
    def __init__(self):
        ndim, dt = 7, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement."""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            4 * self._std_weight_position * measurement[3],  # x
            2 * self._std_weight_position * measurement[4],  # y
            2 * self._std_weight_position * measurement[5],  # z
            1e-2,                                           # l
            1e-2,                                           # w
            1e-2,                                           # h
            1e-2,                                           # theta
            10 * self._std_weight_velocity * measurement[3], # vx
            1 * self._std_weight_velocity * measurement[4], # vy
            1 * self._std_weight_velocity * measurement[5], # vz
            1e-5,                                           # vl
            1e-5,                                           # vw
            1e-5,                                           # vh
            1e-5                                            # vtheta
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space."""
        ndim = 7  # 3D 상태 벡터의 차원

        # 측정 노이즈 행렬
        std = [
            self._std_weight_position * mean[3],  # x
            self._std_weight_position * mean[4],  # y
            self._std_weight_position * mean[5],  # z
            1e-2,                                 # l
            1e-2,                                 # w
            1e-2,                                 # h
            1e-1                                  # theta
        ]
        innovation_cov = np.diag(np.square(std))

        # 프로젝트
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))

        return mean, covariance + innovation_cov
    
    def multi_predict(self, mean, covariance):
        """Run Kalman filter prediction step (Vectorized version)."""
        ndim = 7  # 3D 상태 벡터의 차원

        std_pos = [self._std_weight_position * np.ones_like(mean[:, i]) for i in range(7)]
        std_vel = [self._std_weight_velocity * np.ones_like(mean[:, i]) for i in range(7)]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = np.array([np.diag(sqr[i]) for i in range(len(mean))])

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step."""
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))

        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False, metric='maha'):
        """Compute gating distance between state distribution and measurements."""
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:3], covariance[:3, :3]
            measurements = measurements[:, :3]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('Invalid distance metric')
        

'''
if __name__ == '__main__':
    kf = KalmanFilter3D()

    initial_measurement = np.array([1, 2, 3, 4, 5, 6, np.pi/4])

    # 필터 초기화
    mean, covariance = kf.initiate(initial_measurement)

    # 무작위 측정값 생성
    new_measurement = initial_measurement + np.random.normal(0, 1, initial_measurement.shape)

    # 칼만 필터 예측 및 업데이트
    predicted_mean, predicted_covariance = kf.predict(mean, covariance)
    updated_mean, updated_covariance = kf.update(predicted_mean, predicted_covariance, new_measurement)

    print("Predicted Mean:\n", predicted_mean)
    print("Predicted Covariance:\n", predicted_covariance)
    print("Updated Mean:\n", updated_mean)
    print("Updated Covariance:\n", updated_covariance)
'''