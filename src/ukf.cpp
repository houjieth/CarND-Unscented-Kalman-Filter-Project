#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

constexpr double EPS = 0.001;

void PrintMatrix(const MatrixXd& m, const string& name) {
  cout << "<" << name << ">" << endl;
  for (auto i = 0; i < m.rows(); ++i) {
    for (auto j = 0; j < m.cols(); ++j) {
      cout << m(i, j) << " ";
    }
    cout << endl;
  }
}

void PrintVector(const VectorXd& v, const string& name) {
  cout << "<" << name << ">" << endl;
  for (auto i = 0; i < v.rows(); ++i) {
    cout << v(i) << " ";
  }
  cout << endl;
}

// Make sure phi stays in [-PI, PI]
void NormalizeAngle(double *phi) {
  double normalized_phi = atan2(sin(*phi), cos(*phi));
  *phi = normalized_phi;
}

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  weights_ = VectorXd(2 * n_aug_ + 1);

  // Pre-compute weights for signma points
  const double weight_0 = lambda_ / (lambda_ + n_aug_);
  weights_(0) = weight_0;
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {
    double weight = 0.5 / (n_aug_ + lambda_);
    weights_(i) = weight;
  }

  R_radar_ = MatrixXd(3, 3);
  R_radar_ << std_radr_ *std_radr_, 0, 0,
              0, std_radphi_ * std_radphi_, 0,
              0, 0, std_radrd_ * std_radrd_;

  R_lidar_ = MatrixXd(2, 2);
  R_lidar_ << std_laspx_ * std_laspx_, 0,
              0, std_laspy_ * std_laspy_;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  if (!is_initialized_) {
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      const double px = meas_package.raw_measurements_[0];
      const double py = meas_package.raw_measurements_[1];
      x_ << px, py, 0, 0, 0;
    } else { // Radar
      const double rho = meas_package.raw_measurements_[0];
      const double phi = meas_package.raw_measurements_[1];
      const double rhod = meas_package.raw_measurements_[2];
      const double px = rho * cos(phi);
      const double py = rho * sin(phi);

      x_ << px, py, rhod, 0, 0;
    }
    time_us_ = meas_package.timestamp_;
    P_ = MatrixXd::Identity(5, 5);

    is_initialized_ = true;
    return;
  }

  const double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;
  Prediction(delta_t);
  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  } else {  // Radar
    UpdateRadar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  // Step 1: Get sigma points

  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // Create x_aug
  VectorXd x_aug = VectorXd(n_aug_);
  for (int i = 0; i < n_x_; ++i) {
    x_aug(i) = x_(i);
  }
  x_aug(n_x_) = 0;  // mean of a
  x_aug(n_x_ + 1) = 0;  // mean of yawdd

  // Create P_aug
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;  // variance of a
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;  // variance of yawdd

  MatrixXd L = P_aug.llt().matrixL();  // i.e., sqrt of P_aug

  // Create Xsig_aug
  Xsig_aug.col(0) = x_aug;  // First sigma point
  // And the rest 2 * n_aug_ sigma points
  const double sqt = sqrt(lambda_ + n_aug_);
  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug.col(i + 1) = x_aug + sqt * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqt * L.col(i);
  }

  // Step 2: Make predictions for these sigma points

  for (int i = 0; i< 2 * n_aug_ + 1; i++)
  {
    const double p_x = Xsig_aug(0, i);
    const double p_y = Xsig_aug(1, i);
    const double v = Xsig_aug(2, i);
    const double yaw = Xsig_aug(3, i);
    const double yawd = Xsig_aug(4, i);
    const double a = Xsig_aug(5, i);  // longitudinal acceleration (process noise)
    const double yawdd = Xsig_aug(6, i);  // yaw dot dot (process noise)

    double px_p, py_p;

    // Avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    } else {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // Add noise
    px_p = px_p + 0.5 * a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * a * delta_t * delta_t * sin(yaw);
    v_p = v_p + a * delta_t;

    yaw_p = yaw_p + 0.5 * yawdd * delta_t * delta_t;
    yawd_p = yawd_p + yawdd * delta_t;

    // Write predicted sigma point into right column
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }

  // Step 3: Reconstruct Gaussian (including mean and covariance) from predicted sigma points

  // Calculate new mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  // Calculate new covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    NormalizeAngle(&x_diff(3));

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  // Step 1: Transform predicted sigma points from state space into measurement space

  const int n_z = 2;

  // Create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // Transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    const double px = Xsig_pred_(0, i);
    const double py = Xsig_pred_(1, i);
    // v, yaw, yawd are not useful here, so we don't read them

    Zsig(0, i) = px;
    Zsig(1, i) = py;
  }

  // Step 2: Reconstruct Gaussian with these predicted sigma points in measurement space

  // Calculate mean for the Gaussian
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // Calculate covariance for the Gaussian
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // Generate covariance
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise covariance matrix (notice, the noise is purely additive here)
  S = S + R_lidar_;

  // Step 3: Calculate cross correlation matrix

  // Calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {

    // Measurement difference
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Step 4: Calculate Kalman gain
  MatrixXd K = Tc * S.inverse();

  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  // Step 5: Update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  // Step 1: Transform predicted sigma points from state space into measurement space
  const int n_z = 3;

  // Create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // Transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    const double px = Xsig_pred_(0, i);
    const double py = Xsig_pred_(1, i);
    const double v = Xsig_pred_(2, i);
    const double yaw = Xsig_pred_(3, i);
    // yawd is not useful here, so we don't read it

    const double vx = v * cos(yaw);
    const double vy = v * sin(yaw);

    Zsig(0, i) = sqrt(px * px + py * py);                        // rho
    if (px != 0 && py != 0) {
      Zsig(1, i) = atan2(py, px);                                  // phi
      Zsig(2, i) = (px * vx + py * vy) / sqrt(px * px + py * py);  // rho_dot
    } else {
      Zsig(1, i) = 0;                                                        // phi
      Zsig(2, i) = (px * vx + py * vy) / max(sqrt(px * px + py * py), EPS);  // rho_dot
    }
  }

  // Step 2: Reconstruct Gaussian with these predicted sigma points in measurement space

  // Calculate mean for the Gaussian
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // Calculate covariance for the Gaussian
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;

    NormalizeAngle(&z_diff(1));

    // Generate covariance
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise covariance matrix (notice, the noise is purely additive here)
  S = S + R_radar_;

  // Step 3: Calculate cross corelation matrix
  //calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {

    // Measurement difference
    VectorXd z_diff = Zsig.col(i) - z_pred;

    NormalizeAngle(&z_diff(1));

    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    NormalizeAngle(&x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Step 4: Calculate Kalman gain
  MatrixXd K = Tc * S.inverse();

  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  NormalizeAngle(&z_diff(1));

  // Step 5: Update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
}
