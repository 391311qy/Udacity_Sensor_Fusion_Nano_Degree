#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
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
  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  // std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  // std_yawdd_ = 30;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

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
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  is_initialized_ = false;
  n_x_ = 5;
  n_aug_ = 7;
  std_a_ = 1;// Process noise standard deviation longitudinal acceleration in m/s^2
  std_yawdd_ = 1;  // Process noise standard deviation yaw acceleration in rad/s^2
  lambda_ = 3 - n_x_;// define spreading parameter
  weights_ = VectorXd(2*n_aug_+1);// create vector for weights
  weights_(0) = lambda_ / (lambda_ + n_aug_);// set weights
  for (int i = 1; i < 2 * n_aug_ + 1; ++i) {
      weights_(i) = 1 / (2 * (lambda_ + n_aug_));
  }
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);// create matrix with predicted sigma points as columns
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */



  if(!is_initialized_) {
    // initialize using RADAR
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      x_ << meas_package.raw_measurements_[0] * cos(meas_package.raw_measurements_[1]), 
            meas_package.raw_measurements_[0] * sin(meas_package.raw_measurements_[1]), 
            0, 
            0, 
            0;
    // initialize using LIDAR
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_ << meas_package.raw_measurements_[0], 
            meas_package.raw_measurements_[1], 
            0, 
            0, 
            0;
    }
    is_initialized_ = true;
    // update current time
    time_us_ = meas_package.timestamp_;
  } else {
    long double delta_t = ((meas_package.timestamp_ - time_us_) * 1e-6);
    time_us_ = meas_package.timestamp_; 
    // perform prediction
    Prediction(delta_t);
    // perform update
    // std::cout<<"Begin Update ..."<<std::endl;
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      UpdateRadar(meas_package);
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      UpdateLidar(meas_package);
    }
  }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */



  // Generate Sigma Points
  VectorXd x_aug = VectorXd::Zero(n_aug_);// create augmented mean vector 
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);// create augmented state covariance
  MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_, 2 * n_aug_ + 1);
  // calculate sigma points ...
  x_aug(5) = 0;
  x_aug(6) = 0;
  x_aug.block<5,1>(0,0) = x_;
  P_aug.block<5,5>(0,0) = P_;// assign augmented covariance matrix
  P_aug.block<2,2>(5,5) << std_a_ * std_a_,          0,
                          0,  std_yawdd_ * std_yawdd_;
  MatrixXd P_sqrt = P_aug.llt().matrixL(); // create square root matrix
  Xsig_aug.col(0) = x_aug;// create augmented sigma points
  double coeff = sqrt(lambda_ + n_aug_);
  for (int i = 0; i < n_aug_; i++) {
      Xsig_aug.col(i + 1)          = x_aug + coeff * P_sqrt.col(i);
      Xsig_aug.col(i + 1 + n_aug_) = x_aug - coeff * P_sqrt.col(i);
  }


  // Sigma Point Prediction
  for (int i = 0; i < 2* n_aug_ + 1; i++) {
        VectorXd state;
        VectorXd F(5);
        VectorXd noise(5);
        VectorXd x_t = Xsig_aug.block<5,1>(0,i);
        VectorXd x_nt = Xsig_aug.block<2,1>(5,i);
        if (fabs(x_t(4)) > 0.00001) {
            F << x_t(2)/x_t(4) * (sin(x_t(3) + delta_t * x_t(4)) - sin(x_t(3))),
                x_t(2)/x_t(4) * (-cos(x_t(3) + delta_t * x_t(4)) + cos(x_t(3))),
                0, 
                delta_t * x_t(4),
                0;    
        } else {
            F<< x_t(2) * cos(x_t(3)) * delta_t,
                x_t(2) * sin(x_t(3)) * delta_t,
                0,
                delta_t * x_t(4),
                0;
        }
        noise << 0.5 * delta_t * delta_t * cos(x_t(3)) * x_nt(0), 
                0.5 * delta_t * delta_t * sin(x_t(3)) * x_nt(0), 
                delta_t * x_nt(0),
                0.5 * delta_t * delta_t * x_nt(1),
                delta_t * x_nt(1);
        state = x_t + F + noise;
        Xsig_pred_.col(i) = state;
  }


  // predict state mean and covariance
  x_ = Xsig_pred_ * weights_;// state mean
  P_ = (Xsig_pred_.colwise() - x_) * weights_.asDiagonal() * (Xsig_pred_.colwise() - x_).transpose();// state covariance 
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  int n_z = 2; // Lidar measurements
  VectorXd z = VectorXd(n_z); // coming radar measurment
  VectorXd z_pred = VectorXd(n_z); // create predicted measurement
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1); // matrix with sigma points in measurement space   
  MatrixXd S = MatrixXd(n_z,n_z); // create predicted covariance
  z = meas_package.raw_measurements_;

  // Z sigma points
  for (int i = 0; i < Zsig.cols(); i++) {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double phi = Xsig_pred_(3, i);
    double dphi = Xsig_pred_(4, i);
    VectorXd vec(2);
    vec << px,
           py;
    Zsig.col(i) = vec;
  }

  // predicted measurement mean
  z_pred = Zsig * weights_;

  // innovation sequences
  MatrixXd innov_x = Xsig_pred_.colwise() - x_;
  MatrixXd innov_z = Zsig.colwise() - z_pred;

  // measurement covariance
  MatrixXd R(2,2);
  R << std_laspx_ * std_laspx_, 0,
        0, std_laspy_ * std_laspy_;
  S = (innov_z) * weights_.asDiagonal() * (innov_z).transpose() + R;

  // cross correlation 
  MatrixXd Tc = MatrixXd(n_x_, n_z);   
  Tc = (innov_x) * weights_.asDiagonal() * (innov_z).transpose();
  MatrixXd K;// calculate Kalman gain K;
  K = Tc * S.inverse();
  // update state mean and covariance matrix
  VectorXd zdiff = z - z_pred;
  x_ = x_ + K * (zdiff);
  P_ = P_ - K * S * K.transpose();

}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  int n_z = 3; // radar measurements
  VectorXd z = VectorXd(n_z); // coming radar measurment
  VectorXd z_pred = VectorXd(n_z); // create predicted measurement
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1); // matrix with sigma points in measurement space   
  MatrixXd S = MatrixXd(n_z,n_z); // create predicted covariance

  z = meas_package.raw_measurements_;
  // Z sigma points
  for (int i = 0; i < Zsig.cols(); i++) {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double phi = Xsig_pred_(3, i);
    double dphi = Xsig_pred_(4, i);
    Zsig(0, i) = sqrt(px * px + py * py);
    Zsig(1, i) = atan2(py, px);
    Zsig(2, i) = (px*v*cos(phi) + py*v*sin(phi)) / sqrt(px*px + py*py);
  }

  // predicted measurement mean
  z_pred = Zsig * weights_;

  // innovation sequences
  MatrixXd innov_x = Xsig_pred_.colwise() - x_;
  MatrixXd innov_z = Zsig.colwise() - z_pred;

  // measurement covariance
  MatrixXd R(3,3);
  R << std_radr_*std_radr_, 0, 0,
       0, std_radphi_*std_radphi_, 0,
       0, 0, std_radrd_*std_radrd_;
  S = (innov_z) * weights_.asDiagonal() * (innov_z).transpose() + R;

  // cross correlation  
  MatrixXd Tc = MatrixXd(n_x_, n_z);  
  Tc = (innov_x) * weights_.asDiagonal() * (innov_z).transpose();
  MatrixXd K;// calculate Kalman gain K;
  K = Tc * S.inverse();
  // update state mean and covariance matrix
  VectorXd zdiff = z - z_pred;
  zdiff(1) = std::fmod(zdiff(1), M_PI);
  x_ = x_ + K * (zdiff);
  P_ = P_ - K * S * K.transpose();


}