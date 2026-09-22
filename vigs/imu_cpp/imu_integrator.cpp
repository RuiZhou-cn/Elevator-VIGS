// IMU preintegration of the base system (see imu_integrator.h); the transport columns of
// the paper's Eq. (5) and Eq. (6) are the one addition of Elevator-VIGS.
#include "imu_integrator.h"

#include <cmath>
#include <tuple>
#include <vector>
#include <Eigen/Eigenvalues>

// =======================
// Helper static functions
// =======================

Eigen::Matrix3d IMUIntegrator::hat(const Eigen::Vector3d& v) {
    Eigen::Matrix3d W;
    W <<        0.0, -v.z(),  v.y(),
             v.z(),    0.0, -v.x(),
            -v.y(),  v.x(),   0.0;
    return W;
}

// right Jacobian of SO(3)
Eigen::Matrix3d IMUIntegrator::rightJ(const Eigen::Vector3d& v) {
    double d = v.norm();
    double d2 = d * d;
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

    if (d < 1e-5) {
        return I;
    }

    Eigen::Matrix3d W = hat(v);
    double c = std::cos(d);
    double s = std::sin(d);

    Eigen::Matrix3d term1 = W * ((1.0 - c) / d2);
    Eigen::Matrix3d term2 = W * W * ((d - s) / (d2 * d));

    return I - term1 + term2;
}

// inverse of the right Jacobian of SO(3)
Eigen::Matrix3d IMUIntegrator::invRightJ(const Eigen::Vector3d& v) {
    double d = v.norm();
    double d2 = d * d;
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

    if (d < 1e-5) {
        return I;
    }

    Eigen::Matrix3d W = hat(v);
    double c = std::cos(d);
    double s = std::sin(d);

    double a = 1.0 / d2 - (1.0 + c) / (2.0 * d * s);

    return I + 0.5 * W + a * (W * W);
}

// =======================
// Constructor
// =======================

IMUIntegrator::IMUIntegrator(double prev, double curr,
                             const Eigen::Vector3d& init_bg,
                             const Eigen::Vector3d& init_ba,
                             const Eigen::Vector3d& init_g,
                             double freq,
                             double gyro_noise_density,
                             double acc_noise_density,
                             double gyro_random_walk,
                             double acc_random_walk)
    : prev_(prev)
    , curr_(curr)
    , dT_(0.0)
    , g_(init_g)
    , dP_(Eigen::Vector3d::Zero())
    , dR_(Sophus::SO3d())                 // identity
    , dV_(Eigen::Vector3d::Zero())
    , bg_(init_bg)
    , ba_(init_ba)
    , bg_new_(init_bg)
    , ba_new_(init_ba)
    , JPg_(Eigen::Matrix3d::Zero())
    , JPa_(Eigen::Matrix3d::Zero())
    , JRg_(Eigen::Matrix3d::Zero())
    , JVg_(Eigen::Matrix3d::Zero())
    , JVa_(Eigen::Matrix3d::Zero())
    , Nga_(Eigen::Matrix<double,6,6>::Zero())
    , NgaWalk_(Eigen::Matrix<double,6,6>::Zero())
    , cov_(Eigen::Matrix<double,15,15>::Zero())
    , info_(Eigen::Matrix<double,9,9>::Zero())
    , info2_(Eigen::Matrix<double,6,6>::Zero())
    , temp_A_(Eigen::Matrix<double,9,9>::Zero())
    , temp_B_(Eigen::Matrix<double,9,6>::Zero())
    , temp_cov_result_(Eigen::Matrix<double,9,9>::Zero())
{
    double FREQ = freq;

    Nga_.setZero();
    NgaWalk_.setZero();

    // gyro/acc noise
    double gyro_var = std::pow(gyro_noise_density * std::sqrt(FREQ), 2);
    double acc_var  = std::pow(acc_noise_density  * std::sqrt(FREQ), 2);
    Nga_.block<3,3>(0,0) = gyro_var * Eigen::Matrix3d::Identity();
    Nga_.block<3,3>(3,3) = acc_var  * Eigen::Matrix3d::Identity();

    // gyro/acc random walk
    double gyro_rw_var = std::pow(gyro_random_walk / std::sqrt(FREQ), 2);
    double acc_rw_var  = std::pow(acc_random_walk  / std::sqrt(FREQ), 2);
    NgaWalk_.block<3,3>(0,0) = gyro_rw_var * Eigen::Matrix3d::Identity();
    NgaWalk_.block<3,3>(3,3) = acc_rw_var  * Eigen::Matrix3d::Identity();

    cov_.setZero();
}

// =======================
// Bias update & updated dP/dR/dV
// =======================

void IMUIntegrator::set_new_bias(const Eigen::VectorXd& bias) {
    // bias: 6x1 [bg; ba]
    bg_new_ = bias.head<3>();
    ba_new_ = bias.tail<3>();
}

Eigen::Vector3d IMUIntegrator::get_updated_dP() const {
    Eigen::Vector3d dbg = bg_new_ - bg_;
    Eigen::Vector3d dba = ba_new_ - ba_;
    return dP_ + JPg_ * dbg + JPa_ * dba;
}

Sophus::SO3d IMUIntegrator::get_updated_dR() const {
    Eigen::Vector3d dbg = bg_new_ - bg_;
    return dR_ * Sophus::SO3d::exp(JRg_ * dbg);
}

Eigen::Vector3d IMUIntegrator::get_updated_dV() const {
    Eigen::Vector3d dbg = bg_new_ - bg_;
    Eigen::Vector3d dba = ba_new_ - ba_;
    return dV_ + JVg_ * dbg + JVa_ * dba;
}

// =======================
// compute_error
// =======================

// shared error computation used by compute_error() and jacobian(); returns eR/eV/eP via out-params
void IMUIntegrator::compute_error_raw(
    const Eigen::Matrix4d& G0,
    const Eigen::Matrix4d& G1,
    const Eigen::Vector3d& V0,
    const Eigen::Vector3d& V1,
    const Eigen::Matrix3d& Rwg,
    double scale,
    Sophus::SO3d& eR_out,
    Eigen::Vector3d& eV_out,
    Eigen::Vector3d& eP_out) const
{
    Eigen::Matrix3d R0 = G0.block<3,3>(0,0);
    Eigen::Vector3d t0 = G0.block<3,1>(0,3);
    Eigen::Matrix3d R1 = G1.block<3,3>(0,0);
    Eigen::Vector3d t1 = G1.block<3,1>(0,3);

    Eigen::Matrix3d R0T = R0.transpose();
    Eigen::Vector3d g = Rwg * g_;
    double dt = dT_;

    Eigen::Vector3d dP = get_updated_dP();
    Sophus::SO3d dR = get_updated_dR();
    Eigen::Vector3d dV = get_updated_dV();
    Eigen::Vector3d eP =
        R0T * (scale * (t1 - t0 - V0 * dt) - 0.5 * g * dt * dt) - dP;

    Eigen::Vector3d eV =
        R0T * (scale * (V1 - V0) - g * dt) - dV;
    Sophus::SO3d R_rel(R0T * R1); // SOPHUS_DISABLE_ENSURES (CMakeLists.txt): a numerically drifted product fails the orthogonality check
    Sophus::SO3d eR = dR.inverse() * R_rel;
    eR_out = eR;
    eV_out = eV;
    eP_out = eP;
}

Eigen::VectorXd IMUIntegrator::compute_error(const Eigen::Matrix4d& G0,
                                             const Eigen::Matrix4d& G1,
                                             const Eigen::Vector3d& V0,
                                             const Eigen::Vector3d& V1,
                                             const Eigen::Matrix3d& Rwg,
                                             double scale) const
{
    Sophus::SO3d eR;
    Eigen::Vector3d eV, eP;
    compute_error_raw(G0, G1, V0, V1, Rwg, scale, eR, eV, eP);

    // packed [eR.log(), eV, eP]
    Eigen::VectorXd err(9);
    err.segment<3>(0) = eR.log();
    err.segment<3>(3) = eV;
    err.segment<3>(6) = eP;
    return err;
}

// =======================
// jacobian
// =======================

auto IMUIntegrator::jacobian(const Eigen::Matrix4d& G0,
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
        Eigen::Matrix<double,9,1>>
{
    Eigen::Matrix3d R0 = G0.block<3,3>(0,0);
    Eigen::Vector3d t0 = G0.block<3,1>(0,3);
    Eigen::Matrix3d R1 = G1.block<3,3>(0,0);
    Eigen::Vector3d t1 = G1.block<3,1>(0,3);

    Eigen::Matrix3d R0T = R0.transpose();
    Eigen::Vector3d g = Rwg * g_;
    double dt = dT_;

    Sophus::SO3d eR;
    Eigen::Vector3d eV, eP;
    compute_error_raw(G0, G1, V0, V1, Rwg, scale, eR, eV, eP);

    Eigen::Vector3d eRlog = eR.log();
    Eigen::Matrix3d invJr = invRightJ(eRlog);

    Eigen::Matrix<double,9,6> Ji = Eigen::Matrix<double,9,6>::Zero();
    Eigen::Matrix<double,9,6> Jj = Eigen::Matrix<double,9,6>::Zero();
    Eigen::Matrix<double,9,3> Jvi = Eigen::Matrix<double,9,3>::Zero();
    Eigen::Matrix<double,9,3> Jvj = Eigen::Matrix<double,9,3>::Zero();
    Eigen::Matrix<double,9,3> Jbg = Eigen::Matrix<double,9,3>::Zero();
    Eigen::Matrix<double,9,3> Jba = Eigen::Matrix<double,9,3>::Zero();
    Eigen::Matrix<double,9,1> Jscale = Eigen::Matrix<double,9,1>::Zero();

    // === Ji ===
    // Ji[6:9,:3] = -np.eye(3) * scale
    Ji.block<3,3>(6,0) = -scale * Eigen::Matrix3d::Identity();

    // Ji[:3,3:6] = -invJr @ G1[:3,:3].transpose() @ G0[:3,:3]
    Ji.block<3,3>(0,3) = -invJr * (R1.transpose() * R0);

    // Ji[3:6,3:6] = sp.SO3.hat(G0[:3, :3].transpose() @ (scale*(V1 - V0) - g*self.dT))
    {
        Eigen::Vector3d vec = R0T * (scale * (V1 - V0) - g * dt);
        Ji.block<3,3>(3,3) = hat(vec);
    }

    // Ji[6:9,3:6] = sp.SO3.hat(G0[:3, :3].transpose()
    //                @ (scale*(G1[:3, 3] - G0[:3, 3] - V0*self.dT)
    //                - 0.5*g*self.dT*self.dT))
    {
        Eigen::Vector3d vec =
            R0T * (scale * (t1 - t0 - V0 * dt) - 0.5 * g * dt * dt);
        Ji.block<3,3>(6,3) = hat(vec);
    }

    // === Jj ===
    // Jj[6:9,:3] = scale*G0[:3,:3].transpose() @ G1[:3,:3]
    Jj.block<3,3>(6,0) = scale * (R0T * R1);

    // Jj[:3,3:6] = invJr
    Jj.block<3,3>(0,3) = invJr;

    // === Jvi ===
    // Jvi[3:6,:] = -scale*G0[:3,:3].transpose()
    // Jvi[6:9,:] = -scale*G0[:3,:3].transpose() * self.dT
    Jvi.block<3,3>(3,0) = -scale * R0T;
    Jvi.block<3,3>(6,0) = -scale * R0T * dt;

    // === Jvj ===
    // Jvj[3:6,:] = scale*G0[:3,:3].transpose()
    Jvj.block<3,3>(3,0) = scale * R0T;

    // === Jbg ===
    // dbg = self.bg_new - self.bg
    Eigen::Vector3d dbg = bg_new_ - bg_;
    // Jbg[:3,:] = -invJr @ eR.matrix().transpose()
    //                @ rightJ(self.JRg @ dbg) @ self.JRg
    {
        Eigen::Matrix3d eRmatT = eR.matrix().transpose();
        Eigen::Vector3d JRg_dbg = JRg_ * dbg;
        Eigen::Matrix3d Rj = rightJ(JRg_dbg);
        Jbg.block<3,3>(0,0) = -invJr * eRmatT * Rj * JRg_;
    }

    // Jbg[3:6,:] = -self.JVg
    Jbg.block<3,3>(3,0) = -JVg_;
    // Jbg[6:9,:] = -self.JPg
    Jbg.block<3,3>(6,0) = -JPg_;

    // === Jba ===
    // Jba[3:6,:] = -self.JVa
    // Jba[6:9,:] = -self.JPa
    Jba.block<3,3>(3,0) = -JVa_;
    Jba.block<3,3>(6,0) = -JPa_;

    // === Jscale ===
    // Jscale[3:6,0] = G0[:3, :3].transpose() @ (V1 - V0)
    // Jscale[6:9,0] = G0[:3, :3].transpose()
    //                 @ (G1[:3, 3] - G0[:3, 3] - V0*self.dT)
    Jscale.block<3,1>(3,0) = R0T * (V1 - V0);
    Jscale.block<3,1>(6,0) = R0T * (t1 - t0 - V0 * dt);

    // sign convention: the velocity, bias and scale columns are returned negated
    return std::make_tuple(
        Ji,
        Jj,
        -Jvi,
        -Jvj,
        -Jbg,
        -Jba,
        -Jscale
    );
}

// =======================
// transport_columns
// =======================
// The two transport columns of the paper's Eq. (5) and Eq. (6), for the per-keyframe transport
// state (u_k, h_k): the elevator's vertical velocity u and its rise h along the up axis
// e_z = -g/|g|. The e_R rows are zero; columns are negated as jacobian() does.
Eigen::Matrix<double,9,4>
IMUIntegrator::transport_columns(const Eigen::Matrix4d& G0,
                                 const Eigen::Matrix3d& Rwg,
                                 double scale) const
{
    Eigen::Vector3d gw = Rwg * g_;
    Eigen::Vector3d up = -gw / (gw.norm() + 1e-12);
    Eigen::Vector3d c = scale * (G0.block<3,3>(0,0).transpose() * up);   // scale * R_i^T e_z
    const double dt = dT_;

    Eigen::Matrix<double,9,4> J = Eigen::Matrix<double,9,4>::Zero();
    J.block<3,1>(6,0) =  c;         // -Jh_i : Jh_i[6:9] = -c     (Eq. (5), d r_pos / d h_i)
    J.block<3,1>(3,1) =  c;         // -Ju_i : Ju_i[3:6] = -c     (Eq. (6), d r_vel / d u_i)
    J.block<3,1>(6,1) =  c * dt;    //         Ju_i[6:9] = -c*dT  (Eq. (5), d r_pos / d u_i)
    J.block<3,1>(6,2) = -c;         // -Jh_j : Jh_j[6:9] =  c     (Eq. (5), d r_pos / d h_j)
    J.block<3,1>(3,3) = -c;         // -Ju_j : Ju_j[3:6] =  c     (Eq. (6), d r_vel / d u_j)
    return J;
}

// =======================
// jacobian_GDir
// =======================

Eigen::Matrix<double,9,3>
IMUIntegrator::jacobian_GDir(const Eigen::Matrix4d& G0,
                             const Eigen::Matrix4d& /*G1*/,
                             const Eigen::Vector3d& /*V0*/,
                             const Eigen::Vector3d& /*V1*/,
                             const Eigen::Matrix3d& Rwg) const
{
    Eigen::Matrix<double,3,2> Gm;
    Gm.setZero();
    Gm(0,1) = -9.81;
    Gm(1,0) =  9.81;

    Eigen::Matrix<double,3,2> dGdTheta = Rwg * Gm; // gravity dir has only 2 DOF; Jgdir's 3rd column stays zero

    Eigen::Matrix3d R0 = G0.block<3,3>(0,0);
    Eigen::Matrix3d R0T = R0.transpose();
    double dt = dT_;

    Eigen::Matrix<double,9,3> Jgdir = Eigen::Matrix<double,9,3>::Zero();

    Jgdir.block<3,2>(3,0) = -R0T * dGdTheta * dt;
    Jgdir.block<3,2>(6,0) = -0.5 * R0T * dGdTheta * dt * dt;

    return -Jgdir;
}

// =======================
// Integration helpers
// =======================

std::tuple<double,Eigen::Vector3d,Eigen::Vector3d>
IMUIntegrator::average(const Eigen::Matrix<double,7,1>& m0,
                       const Eigen::Matrix<double,7,1>& m1,
                       bool first, bool last) const
{
    double t0 = m0(0);
    double t1 = m1(0);

    Eigen::Vector3d gyr0 = m0.segment<3>(1);
    Eigen::Vector3d gyr1 = m1.segment<3>(1);
    Eigen::Vector3d acc0 = m0.segment<3>(4);
    Eigen::Vector3d acc1 = m1.segment<3>(4);

    double dt;
    Eigen::Vector3d acc, gyr;

    if (first) {
        double tini = t0 - prev_;
        double tab  = t1 - t0;
        dt = tini + tab;

        Eigen::Vector3d acc_diff = acc1 - acc0;
        Eigen::Vector3d gyr_diff = gyr1 - gyr0;

        acc = (acc0 + acc1 - acc_diff * (tini / tab)) * 0.5;
        gyr = (gyr0 + gyr1 - gyr_diff * (tini / tab)) * 0.5;
    } else if (last) {
        double tend = curr_ - t1;
        double tab  = t1 - t0;
        dt = tab + tend;

        Eigen::Vector3d acc_diff = acc1 - acc0;
        Eigen::Vector3d gyr_diff = gyr1 - gyr0;

        acc = (acc0 + acc1 + acc_diff * (tend / tab)) * 0.5;
        gyr = (gyr0 + gyr1 + gyr_diff * (tend / tab)) * 0.5;
    } else {
        dt = t1 - t0;
        acc = 0.5 * (acc0 + acc1);
        gyr = 0.5 * (gyr0 + gyr1);
    }

    return std::make_tuple(dt, acc, gyr);
}

void IMUIntegrator::integrate_once(const Eigen::Matrix<double,7,1>& m0,
                                   const Eigen::Matrix<double,7,1>& m1,
                                   bool first, bool last)
{
    auto [dt, acc_raw, gyr_raw] = average(m0, m1, first, last);

    dT_ += dt;

    // Remove biases
    const Eigen::Vector3d acc = acc_raw - ba_;
    const Eigen::Vector3d gyr = gyr_raw - bg_;

    // Cache frequently reused values
    const double dt_sq = dt * dt;
    const double half_dt_sq = 0.5 * dt_sq;
    const Eigen::Vector3d dt_gyr = dt * gyr;

    // Get rotation matrix once
    const Eigen::Matrix3d prev_dR = dR_.matrix();
    Eigen::Vector3d acc_w;
    acc_w.noalias() = prev_dR * acc;

    // Pre-compute rightJ (used twice: for JRg and B matrix)
    const Eigen::Matrix3d rightJ_dtgyr = rightJ(dt_gyr);
    const Sophus::SO3d deltaR = Sophus::SO3d::exp(dt_gyr);
    const Eigen::Matrix3d deltaR_T = deltaR.matrix().transpose();

    // Cache common products
    Eigen::Matrix3d prev_dR_dt;
    prev_dR_dt.noalias() = prev_dR * dt;
    Eigen::Matrix3d prev_dR_half_dt_sq;
    prev_dR_half_dt_sq.noalias() = prev_dR * half_dt_sq;
    const Eigen::Matrix3d Wacc = hat(acc);
    Eigen::Matrix3d Wacc_JRg;
    Wacc_JRg.noalias() = Wacc * JRg_;

    // Update preintegrated state (use noalias to avoid temporaries)
    dP_.noalias() += dV_ * dt;
    dP_.noalias() += acc_w * half_dt_sq;
    dV_.noalias() += acc_w * dt;
    dR_ *= deltaR;

    // Update Jacobians w.r.t. biases
    JPa_.noalias() += JVa_ * dt;
    JPa_ -= prev_dR_half_dt_sq;
    JPg_.noalias() += JVg_ * dt;
    JPg_.noalias() -= prev_dR_half_dt_sq * Wacc_JRg;
    JVa_ -= prev_dR_dt;
    JVg_.noalias() -= prev_dR_dt * Wacc_JRg;
    JRg_ = deltaR_T * JRg_ - rightJ_dtgyr * dt;

    // Build A matrix in pre-allocated buffer
    temp_A_.setIdentity();
    temp_A_.block<3,3>(0,0) = deltaR_T;
    temp_A_.block<3,3>(3,0).noalias() = -prev_dR_dt * Wacc;
    temp_A_.block<3,3>(6,0).noalias() = -prev_dR_half_dt_sq * Wacc;
    temp_A_.block<3,3>(6,3).diagonal().setConstant(dt);

    // Build B matrix in pre-allocated buffer
    temp_B_.setZero();
    temp_B_.block<3,3>(0,0).noalias() = rightJ_dtgyr * dt;
    temp_B_.block<3,3>(3,3) = prev_dR_dt;
    temp_B_.block<3,3>(6,3) = prev_dR_half_dt_sq;

    // Covariance propagation: cov = A * cov * A^T + B * Nga * B^T
    // Use pre-allocated buffer to avoid allocation
    temp_cov_result_.noalias() = temp_A_ * cov_.block<9,9>(0,0) * temp_A_.transpose();
    temp_cov_result_.noalias() += temp_B_ * Nga_ * temp_B_.transpose();
    cov_.block<9,9>(0,0) = temp_cov_result_;

    // Bias random walk covariance
    cov_.block<6,6>(9,9) += NgaWalk_;
}

// =======================
// integrate (full preintegration)
// =======================

void IMUIntegrator::integrate(const std::vector<Eigen::Matrix<double,7,1>>& meas)
{
    if (meas.size() < 2) {
        // Nothing to integrate, keep cov/info as zero
        return;
    }

    const int N = static_cast<int>(meas.size());

    for (int i = 0; i < N - 1; ++i) {
        bool first = (i == 0);
        bool last  = (i == N - 2);
        integrate_once(meas[i], meas[i+1], first, last);
    }

    Eigen::Matrix<double,9,9> cov9 = cov_.block<9,9>(0,0);
    Eigen::Matrix<double,9,9> info_mat = cov9.inverse();
    info_mat = 0.5 * (info_mat + info_mat.transpose());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,9,9>> es(info_mat);
    Eigen::VectorXd w = es.eigenvalues();
    Eigen::Matrix<double,9,9> v = es.eigenvectors();

    info_ = v * w.asDiagonal() * v.transpose();

    Eigen::Matrix<double,6,6> cov6 = cov_.block<6,6>(9,9);
    info2_ = cov6.inverse();
}
