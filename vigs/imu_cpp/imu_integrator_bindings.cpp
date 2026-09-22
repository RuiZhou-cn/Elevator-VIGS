// Python bindings of the IMU integrator (module imu_integrator_cpp): the IMUIntegrator class
// and batched_preint_factors_cpu, which assembles the per-edge inertial residual, information
// and Jacobians for a whole edge list in parallel, at 15 columns or, with transport_columns,
// at the 17-wide layout [pose6|vel3|bias6|u@15|h@16] of the paper's Eq. (5) and Eq. (6).
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "imu_integrator.h"

#include <pybind11/numpy.h>
#include <Eigen/Core>
#include <vector>
#include <atomic>
#include <cmath>     // for std::isnan, std::isinf
#include <omp.h>

namespace py = pybind11;


static py::dict batched_preint_factors_cpu(
    py::list integrators,
    py::array_t<double, py::array::c_style | py::array::forcecast> G0s,   // (K,4,4)
    py::array_t<double, py::array::c_style | py::array::forcecast> G1s,   // (K,4,4)
    py::array_t<double, py::array::c_style | py::array::forcecast> V0s,   // (K,3)
    py::array_t<double, py::array::c_style | py::array::forcecast> V1s,   // (K,3)
    py::array_t<double, py::array::c_style | py::array::forcecast> B0s,   // (K,6)
    py::array_t<double, py::array::c_style | py::array::forcecast> Rwg,   // (3,3) or (K,3,3)
    bool wo_pose,
    bool GDir,
    double scale,
    bool transport_columns_on
) {
    if (transport_columns_on && wo_pose) throw std::runtime_error("transport_columns=True requires wo_pose=False");
    const ssize_t K = integrators.size();
    if (K <= 0) throw std::runtime_error("integrators list is empty");

    // Extract all integrator pointers WITH GIL held (Python list access requires GIL)
    std::vector<IMUIntegrator*> integrator_ptrs(K);
    for (ssize_t k = 0; k < K; ++k) {
        try {
            integrator_ptrs[k] = integrators[k].cast<IMUIntegrator*>();
        } catch (const py::cast_error& e) {
            throw std::runtime_error("Failed to cast integrator at index " + std::to_string(k) + ": " + e.what());
        }
        if (integrator_ptrs[k] == nullptr) {
            throw std::runtime_error("Null integrator pointer at index " + std::to_string(k));
        }
        // Validate integrator has valid dT (sanity check)
        double dT = integrator_ptrs[k]->get_dT();
        if (dT <= 0 || std::isnan(dT) || std::isinf(dT)) {
            throw std::runtime_error("Invalid dT=" + std::to_string(dT) + " at integrator index " + std::to_string(k));
        }
    }

    // All numpy array access must happen WITH GIL held
    auto bG0 = G0s.request();
    auto bG1 = G1s.request();
    auto bV0 = V0s.request();
    auto bV1 = V1s.request();
    auto bB0 = B0s.request();
    auto bRg = Rwg.request();

    auto require = [](bool cond, const char* msg) {
        if (!cond) throw std::runtime_error(msg);
    };

    require(bG0.ndim == 3 && bG0.shape[0] == K && bG0.shape[1] == 4 && bG0.shape[2] == 4, "G0s must be (K,4,4)");
    require(bG1.ndim == 3 && bG1.shape[0] == K && bG1.shape[1] == 4 && bG1.shape[2] == 4, "G1s must be (K,4,4)");
    require(bV0.ndim == 2 && bV0.shape[0] == K && bV0.shape[1] == 3, "V0s must be (K,3)");
    require(bV1.ndim == 2 && bV1.shape[0] == K && bV1.shape[1] == 3, "V1s must be (K,3)");
    require(bB0.ndim == 2 && bB0.shape[0] == K && bB0.shape[1] == 6, "B0s must be (K,6)");

    const bool Rwg_batched = (bRg.ndim == 3);
    if (!Rwg_batched) {
        require(bRg.ndim == 2 && bRg.shape[0] == 3 && bRg.shape[1] == 3, "Rwg must be (3,3) or (K,3,3)");
    } else {
        require(bRg.shape[0] == K && bRg.shape[1] == 3 && bRg.shape[2] == 3, "Rwg must be (K,3,3)");
    }

    const double* pG0 = static_cast<const double*>(bG0.ptr);
    const double* pG1 = static_cast<const double*>(bG1.ptr);
    const double* pV0 = static_cast<const double*>(bV0.ptr);
    const double* pV1 = static_cast<const double*>(bV1.ptr);
    const double* pB0 = static_cast<const double*>(bB0.ptr);
    const double* pRg = static_cast<const double*>(bRg.ptr);

    // Dimensions:
    // error D = 9, fixed by IMUIntegrator::compute_error
    constexpr int D = 9;
    const int pose_cols = wo_pose ? 0 : 6;
    const int elev_cols = transport_columns_on ? 2 : 0; // [u@15 | h@16]
    const int Mi = pose_cols + 3 + 3 + 3 + elev_cols; // (Ji?) + Jvi + Jbg + Jba (+ Ju_i + Jh_i)
    const int Mj = pose_cols + 3 + 3 + 3 + elev_cols; // (Jj?) + Jvj + zeros + zeros (+ Ju_j + Jh_j)
    const int Mg = GDir ? (3 + 1) : 0;    // Jgdir (9x3) + Js (9x1)

    // Allocate outputs (numpy float64):
    py::array_t<double> out_info({K, (ssize_t)D, (ssize_t)D});         // (K,9,9)
    py::array_t<double> out_Jinti({K, (ssize_t)D, (ssize_t)Mi});       // (K,9,Mi)
    py::array_t<double> out_Jintj({K, (ssize_t)D, (ssize_t)Mj});       // (K,9,Mj)
    py::array_t<double> out_eint({K, (ssize_t)D});                     // (K,9)
    py::array_t<double> out_Jgs;                                       // (K,9,4)
    if (GDir) out_Jgs = py::array_t<double>({K, (ssize_t)D, (ssize_t)Mg});

    auto oinfo  = out_info.mutable_unchecked<3>();
    auto oJinti = out_Jinti.mutable_unchecked<3>();
    auto oJintj = out_Jintj.mutable_unchecked<3>();
    auto oeint  = out_eint.mutable_unchecked<2>();

    // For GDir case, use raw pointer to output array data
    double* pJgs = GDir ? static_cast<double*>(out_Jgs.mutable_data()) : nullptr;

    // Thread-safe error handling for OMP loop
    std::atomic<bool> has_error{false};
    std::string error_message;
    ssize_t error_index = -1;
    
    // NOW release GIL for the parallel computation (after all Python/numpy setup is done)
    {
        py::gil_scoped_release release;
        
        // Main loop in C++ with OpenMP parallelization
        #pragma omp parallel for schedule(dynamic)
        for (ssize_t k = 0; k < K; ++k) {
            // Skip remaining iterations if error occurred
            if (has_error.load()) continue;

            IMUIntegrator* inter = integrator_ptrs[k];  // Use pre-extracted pointer

            // Map inputs (RowMajor matches numpy contiguous default)
            Eigen::Map<const Eigen::Matrix<double,4,4,Eigen::RowMajor>> G0(pG0 + k*16);
            Eigen::Map<const Eigen::Matrix<double,4,4,Eigen::RowMajor>> G1(pG1 + k*16);
            Eigen::Map<const Eigen::Vector3d> V0(pV0 + k*3);
            Eigen::Map<const Eigen::Vector3d> V1(pV1 + k*3);
            Eigen::Map<const Eigen::Matrix<double,6,1>> B0(pB0 + k*6);

            Eigen::Matrix3d Rg;
            if (!Rwg_batched) {
                Rg = Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>>(pRg);
            } else {
                Rg = Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>>(pRg + k*9);
            }

            inter->set_new_bias(B0);

            // error (size 9)
            Eigen::VectorXd e = inter->compute_error(G0, G1, V0, V1, Rg, scale);

            // Thread-safe error check (no exception in OMP region)
            if (e.size() != D) {
                bool expected = false;
                if (has_error.compare_exchange_strong(expected, true)) {
                    #pragma omp critical
                    {
                        error_message = "compute_error returned size " + std::to_string(e.size()) + " (expected 9) at index " + std::to_string(k);
                        error_index = k;
                    }
                }
                continue;
            }

            for (int r = 0; r < D; ++r) oeint(k, r) = e(r);

            // info (9x9)
            const auto& info = inter->info();
            for (int r = 0; r < D; ++r)
                for (int c = 0; c < D; ++c)
                    oinfo(k, r, c) = info(r, c);

            // jacobian tuple. Python parity: the 15-wide path calls jacobian() with the
            // default scale=1, the transport_columns_on path passes scale through (geom/ba.py).
            auto tup = inter->jacobian(G0, G1, V0, V1, Rg, transport_columns_on ? scale : 1.0);
            Eigen::Matrix<double,9,4> Jt;   // [Jh_i | Ju_i | Jh_j | Ju_j]
            if (transport_columns_on) Jt = inter->transport_columns(G0, Rg, scale);

            const auto& Ji  = std::get<0>(tup); // (9,6)
            const auto& Jj  = std::get<1>(tup); // (9,6)
            const auto& Jvi = std::get<2>(tup); // (9,3)
            const auto& Jvj = std::get<3>(tup); // (9,3)
            const auto& Jbg = std::get<4>(tup); // (9,3)
            const auto& Jba = std::get<5>(tup); // (9,3)
            const auto& Js  = std::get<6>(tup); // (9,1)

            // Fill Jinti
            int col = 0;
            if (!wo_pose) {
                for (int c = 0; c < 6; ++c, ++col)
                    for (int r = 0; r < D; ++r)
                        oJinti(k, r, col) = Ji(r, c);
            }
            for (int c = 0; c < 3; ++c, ++col)
                for (int r = 0; r < D; ++r)
                    oJinti(k, r, col) = Jvi(r, c);
            for (int c = 0; c < 3; ++c, ++col)
                for (int r = 0; r < D; ++r)
                    oJinti(k, r, col) = Jbg(r, c);
            for (int c = 0; c < 3; ++c, ++col)
                for (int r = 0; r < D; ++r)
                    oJinti(k, r, col) = Jba(r, c);
            if (transport_columns_on) {                       // u at col 15, h at col 16
                for (int r = 0; r < D; ++r) oJinti(k, r, col)     = Jt(r, 1);   // Ju_i
                for (int r = 0; r < D; ++r) oJinti(k, r, col + 1) = Jt(r, 0);   // Jh_i
                col += 2;
            }

            // Fill Jintj
            col = 0;
            if (!wo_pose) {
                for (int c = 0; c < 6; ++c, ++col)
                    for (int r = 0; r < D; ++r)
                        oJintj(k, r, col) = Jj(r, c);
            }
            for (int c = 0; c < 3; ++c, ++col)
                for (int r = 0; r < D; ++r)
                    oJintj(k, r, col) = Jvj(r, c);

            // zeros for bg/ba blocks on j side
            for (int c = 0; c < 6; ++c, ++col)
                for (int r = 0; r < D; ++r)
                    oJintj(k, r, col) = 0.0;
            if (transport_columns_on) {
                for (int r = 0; r < D; ++r) oJintj(k, r, col)     = Jt(r, 3);   // Ju_j
                for (int r = 0; r < D; ++r) oJintj(k, r, col + 1) = Jt(r, 2);   // Jh_j
                col += 2;
            }

            // Optional gravity dir
            if (GDir && pJgs) {
                Eigen::Matrix<double,9,3> Jgdir = inter->jacobian_GDir(G0, G1, V0, V1, Rg);
                // out_Jgs shape is (K, D, Mg) where Mg = 4, stored in row-major (C-style)
                // Index: pJgs[k * D * Mg + r * Mg + gcol]
                int gcol = 0;
                for (int c = 0; c < 3; ++c, ++gcol)
                    for (int r = 0; r < D; ++r)
                        pJgs[k * D * Mg + r * Mg + gcol] = Jgdir(r, c);

                for (int c = 0; c < 1; ++c, ++gcol)
                    for (int r = 0; r < D; ++r)
                        pJgs[k * D * Mg + r * Mg + gcol] = Js(r, c);
            }
        }
    }
    // Check for errors after OMP region (safe to throw here)
    if (has_error.load()) {
        throw std::runtime_error(error_message);
    }

    py::dict out;
    out["info"]  = out_info;
    out["Jinti"] = out_Jinti;
    out["Jintj"] = out_Jintj;
    out["eint"]  = out_eint;
    if (GDir) out["Jgs"] = out_Jgs;
    return out;
}


PYBIND11_MODULE(imu_integrator_cpp, m) {
    py::class_<IMUIntegrator>(m, "IMUIntegrator")
        .def(py::init<double,double,
                      const Eigen::Vector3d&,
                      const Eigen::Vector3d&,
                      const Eigen::Vector3d&,
                      double,double,double,double,double>(),
             py::arg("prev"),
             py::arg("curr"),
             py::arg("init_bg"),
             py::arg("init_ba"),
             py::arg("init_g"),
             py::arg("freq"),
             py::arg("gyro_noise_density"),
             py::arg("acc_noise_density"),
             py::arg("gyro_random_walk"),
             py::arg("acc_random_walk"))

        .def("set_new_bias", &IMUIntegrator::set_new_bias)

        .def("compute_error",
             [](IMUIntegrator& self,
                const Eigen::Matrix4d& G0,
                const Eigen::Matrix4d& G1,
                const Eigen::Vector3d& V0,
                const Eigen::Vector3d& V1,
                const Eigen::Matrix3d& Rwg,
                double scale) {
                 py::gil_scoped_release release;
                 return self.compute_error(G0, G1, V0, V1, Rwg, scale);
             })

        .def("jacobian",
             [](IMUIntegrator& self,
                const Eigen::Matrix4d& G0,
                const Eigen::Matrix4d& G1,
                const Eigen::Vector3d& V0,
                const Eigen::Vector3d& V1,
                const Eigen::Matrix3d& Rwg,
                double scale,
                bool transport_columns_on) -> py::object {
                 // 7-tuple, plus the four transport columns when transport_columns=True.
                 std::tuple<Eigen::Matrix<double,9,6>, Eigen::Matrix<double,9,6>,
                            Eigen::Matrix<double,9,3>, Eigen::Matrix<double,9,3>,
                            Eigen::Matrix<double,9,3>, Eigen::Matrix<double,9,3>,
                            Eigen::Matrix<double,9,1>> tup;
                 Eigen::Matrix<double,9,4> Jt;
                 {
                     py::gil_scoped_release release;
                     tup = self.jacobian(G0, G1, V0, V1, Rwg, scale);
                     if (transport_columns_on) Jt = self.transport_columns(G0, Rwg, scale);
                 }
                 if (!transport_columns_on) return py::cast(tup);
                 return py::make_tuple(std::get<0>(tup), std::get<1>(tup), std::get<2>(tup),
                                       std::get<3>(tup), std::get<4>(tup), std::get<5>(tup),
                                       std::get<6>(tup),
                                       Eigen::Matrix<double,9,1>(Jt.col(0)),   // Jh_i
                                       Eigen::Matrix<double,9,1>(Jt.col(1)),   // Ju_i
                                       Eigen::Matrix<double,9,1>(Jt.col(2)),   // Jh_j
                                       Eigen::Matrix<double,9,1>(Jt.col(3)));  // Ju_j
             },
             py::arg("G0"), py::arg("G1"), py::arg("V0"), py::arg("V1"), py::arg("Rwg"),
             py::arg("scale") = 1.0, py::arg("transport_columns") = false)

        .def("jacobian_GDir",
             [](IMUIntegrator& self,
                const Eigen::Matrix4d& G0,
                const Eigen::Matrix4d& G1,
                const Eigen::Vector3d& V0,
                const Eigen::Vector3d& V1,
                const Eigen::Matrix3d& Rwg) {
                 py::gil_scoped_release release;
                 return self.jacobian_GDir(G0, G1, V0, V1, Rwg);
             })

        .def("integrate",
             [](IMUIntegrator& self,
                const std::vector<Eigen::Matrix<double,7,1>>& meas) {
                 py::gil_scoped_release release;
                 self.integrate(meas);
             })
          .def("get_updated_dP",
               [](const IMUIntegrator& self) {
                    py::gil_scoped_release release;
                    return self.get_updated_dP();
               },
               "Get updated preintegrated position delta")
     
          .def("get_updated_dV",
               [](const IMUIntegrator& self) {
                    py::gil_scoped_release release;
                    return self.get_updated_dV();
               },
               "Get updated preintegrated velocity delta")
          
          .def("get_updated_dR_log",
               [](const IMUIntegrator& self) {
                    py::gil_scoped_release release;
                    return self.get_updated_dR().log();
               },"Get updated preintegrated rotation (log)")


        .def_property_readonly("dT", &IMUIntegrator::get_dT)
        .def_property_readonly("info", &IMUIntegrator::info)
        .def_property_readonly("info2", &IMUIntegrator::info2)
        .def_property_readonly("cov", &IMUIntegrator::cov)
        .def_property_readonly("g", &IMUIntegrator::get_g);
    m.def("batched_preint_factors_cpu",
        &batched_preint_factors_cpu,
        py::arg("integrators"),
        py::arg("G0s"),
        py::arg("G1s"),
        py::arg("V0s"),
        py::arg("V1s"),
        py::arg("B0s"),
        py::arg("Rwg"),
        py::arg("wo_pose") = false,
        py::arg("GDir") = false,
        py::arg("scale") = 1.0,
        py::arg("transport_columns") = false);
}
