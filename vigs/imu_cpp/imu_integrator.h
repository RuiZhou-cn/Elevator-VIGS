// IMU preintegration between two keyframes (Forster et al.), the C++ integrator of the base
// system: bias-corrected increments dP/dV/dR, their covariance and the residual Jacobians
// of the paper's Eq. (2) and Eq. (3). Elevator-VIGS adds transport_columns(), the two
// Jacobian columns per keyframe of Eq. (5) and Eq. (6) for the transport state (u_k, h_k).
#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <sophus/so3.hpp>

class IMUIntegrator {
public:
    IMUIntegrator(double prev, double curr,
                  const Eigen::Vector3d& init_bg,
                  const Eigen::Vector3d& init_ba,
                  const Eigen::Vector3d& init_g,
                  double freq,
                  double gyro_noise_density,
                  double acc_noise_density,
                  double gyro_random_walk,
                  double acc_random_walk);

    void set_new_bias(const Eigen::VectorXd& bias);

    Eigen::Vector3d get_updated_dP() const;
    Sophus::SO3d    get_updated_dR() const;
    Eigen::Vector3d get_updated_dV() const;

    Eigen::VectorXd compute_error(const Eigen::Matrix4d& G0,
                                  const Eigen::Matrix4d& G1,
                                  const Eigen::Vector3d& V0,
                                  const Eigen::Vector3d& V1,
                                  const Eigen::Matrix3d& Rwg,
                                  double scale) const;

    auto jacobian(const Eigen::Matrix4d& G0,
                  const Eigen::Matrix4d& G1,
                  const Eigen::Vector3d& V0,
                  const Eigen::Vector3d& V1,
                  const Eigen::Matrix3d& Rwg,
                  double scale)
        -> std::tuple<
            Eigen::Matrix<double,9,6>,
            Eigen::Matrix<double,9,6>,
            Eigen::Matrix<double,9,3>,
            Eigen::Matrix<double,9,3>,
            Eigen::Matrix<double,9,3>,
            Eigen::Matrix<double,9,3>,
            Eigen::Matrix<double,9,1>>;

    // Jacobian columns for the transport state (u_k, h_k) -- [Jh_i | Ju_i | Jh_j | Ju_j] (9x4)
    // of the paper's Eq. (5) and Eq. (6); sign convention as in jacobian() (already negated).
    Eigen::Matrix<double,9,4> transport_columns(const Eigen::Matrix4d& G0,
                                                const Eigen::Matrix3d& Rwg,
                                                double scale) const;

    Eigen::Matrix<double,9,3> jacobian_GDir(const Eigen::Matrix4d& G0,
                                            const Eigen::Matrix4d& G1,
                                            const Eigen::Vector3d& V0,
                                            const Eigen::Vector3d& V1,
                                            const Eigen::Matrix3d& Rwg) const;

    void integrate(const std::vector<Eigen::Matrix<double,7,1>>& meas);

    double get_dT() const { return dT_; }
    const Eigen::Matrix<double,9,9>& info() const { return info_; }
    const Eigen::Matrix<double,6,6>& info2() const { return info2_; }
    const Eigen::Matrix<double,15,15>& cov() const { return cov_; }
    const Eigen::Vector3d& get_g() const { return g_; }

private:
    // integrator state
    double prev_, curr_;
    double dT_;
    Eigen::Vector3d g_;

    Eigen::Vector3d dP_;
    Sophus::SO3d    dR_;
    Eigen::Vector3d dV_;

    Eigen::Vector3d bg_, ba_;
    Eigen::Vector3d bg_new_, ba_new_;

    Eigen::Matrix3d JPg_, JPa_, JRg_, JVg_, JVa_;

    Eigen::Matrix<double,6,6> Nga_;
    Eigen::Matrix<double,6,6> NgaWalk_;
    Eigen::Matrix<double,15,15> cov_;

    Eigen::Matrix<double,9,9> info_;
    Eigen::Matrix<double,6,6> info2_;

    // Pre-allocated buffers for integrate_once (avoid per-call allocation)
    Eigen::Matrix<double,9,9> temp_A_;
    Eigen::Matrix<double,9,6> temp_B_;
    Eigen::Matrix<double,9,9> temp_cov_result_;

private:
    // SO(3) helpers
    static Eigen::Matrix3d hat(const Eigen::Vector3d& v);
    static Eigen::Matrix3d rightJ(const Eigen::Vector3d& v);
    static Eigen::Matrix3d invRightJ(const Eigen::Vector3d& v);

    std::tuple<double,Eigen::Vector3d,Eigen::Vector3d>
    average(const Eigen::Matrix<double,7,1>& m0,
            const Eigen::Matrix<double,7,1>& m1,
            bool first, bool last) const;

    void integrate_once(const Eigen::Matrix<double,7,1>& m0,
                        const Eigen::Matrix<double,7,1>& m1,
                        bool first, bool last);

    void compute_error_raw(const Eigen::Matrix4d& G0,
                            const Eigen::Matrix4d& G1,
                            const Eigen::Vector3d& V0,
                            const Eigen::Vector3d& V1,
                            const Eigen::Matrix3d& Rwg,
                            double scale,
                            Sophus::SO3d& eR_out,
                            Eigen::Vector3d& eV_out,
                            Eigen::Vector3d& eP_out) const;
};
