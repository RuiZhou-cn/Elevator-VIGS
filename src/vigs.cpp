#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
torch::Tensor depth_filter_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ix,
    torch::Tensor thresh);


torch::Tensor frame_distance_cuda(
  torch::Tensor poses,
  torch::Tensor disps,
  torch::Tensor intrinsics,
  torch::Tensor ii,
  torch::Tensor jj,
  const float beta);

torch::Tensor covis_distance_cuda(
  torch::Tensor poses,
  torch::Tensor disps,
  torch::Tensor intrinsics,
  torch::Tensor ii);

torch::Tensor iproj_cuda(
  torch::Tensor poses,
  torch::Tensor disps,
  torch::Tensor intrinsics);

std::vector<torch::Tensor> bi_inter_cuda(
  torch::Tensor scales,
  torch::Tensor grids);

std::vector<torch::Tensor> proj_trans_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor ii,
    torch::Tensor jj);

std::vector<torch::Tensor> ba_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor eta,
    torch::Tensor ii,
    torch::Tensor jj,
    const int t0,
    const int t1,
    const int iterations,
    const float lm,
    const float ep,
    const bool motion_only, const bool fix_pose);

torch::Tensor inertial_ba_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor Tij,
    torch::Tensor Tibj,
    torch::Tensor Tcb,
    torch::Tensor Hint,
    torch::Tensor vint,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor eta,
    torch::Tensor ii,
    torch::Tensor jj,
    const int t0,
    const int t1,
    const int iterations,
    const float lm,
    const float ep,
    const bool fix_pose,
    torch::Tensor gblind);



std::vector<torch::Tensor> pgba_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor eta,
    torch::Tensor Hs,
    torch::Tensor vs,
    torch::Tensor Eii,
    torch::Tensor Eij,
    torch::Tensor Cii,
    torch::Tensor wi,
    torch::Tensor Hsp,
    torch::Tensor vsp,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor iip,
    torch::Tensor jjp,
    const int t0,
    const int t1,
    const float lm,
    const float ep,
    const bool use_vision_constraints);

std::vector<torch::Tensor> bias_prior_factors_cuda(
    torch::Tensor biass,
    torch::Tensor ii,
    const float preint_scale,
    const int D);

std::vector<torch::Tensor> bias_factors_cuda(
    torch::Tensor biass,
    torch::Tensor info2s,
    torch::Tensor ii,
    torch::Tensor jj,
    const float preint_scale,
    const int D);

std::vector<torch::Tensor> corr_index_cuda_forward(
  torch::Tensor volume,
  torch::Tensor coords,
  int radius);

std::vector<torch::Tensor> corr_index_cuda_backward(
  torch::Tensor volume,
  torch::Tensor coords,
  torch::Tensor corr_grad,
  int radius);

std::vector<torch::Tensor> altcorr_cuda_forward(
  torch::Tensor fmap1,
  torch::Tensor fmap2,
  torch::Tensor coords,
  int radius);

std::vector<torch::Tensor> altcorr_cuda_backward(
  torch::Tensor fmap1,
  torch::Tensor fmap2,
  torch::Tensor coords,
  torch::Tensor corr_grad,
  int radius);


#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> proj_trans(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor ii,
    torch::Tensor jj) {

  CHECK_INPUT(targets);
  CHECK_INPUT(weights);
  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);
  CHECK_INPUT(ii);
  CHECK_INPUT(jj);

  return proj_trans_cuda(poses, disps, intrinsics, targets, weights, ii, jj);
}

std::vector<torch::Tensor> ba(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor eta,
    torch::Tensor ii,
    torch::Tensor jj,
    const int t0,
    const int t1,
    const int iterations,
    const float lm,
    const float ep,
    const bool motion_only,
    const bool fix_pose) {

  CHECK_INPUT(targets);
  CHECK_INPUT(weights);
  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);
  CHECK_INPUT(ii);
  CHECK_INPUT(jj);

  return ba_cuda(poses, disps, intrinsics, targets, weights,
                 eta, ii, jj, t0, t1, iterations, lm, ep, motion_only, fix_pose);

}

torch::Tensor inertial_ba(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor Tij,
    torch::Tensor Tibj,
    torch::Tensor Tcb,
    torch::Tensor Hint,
    torch::Tensor vint,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor eta,
    torch::Tensor ii,
    torch::Tensor jj,
    const int t0,
    const int t1,
    const int iterations,
    const float lm,
    const float ep,
    const bool fix_pose,
    torch::Tensor gblind) {

  CHECK_INPUT(targets);
  CHECK_INPUT(weights);
  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);
  CHECK_INPUT(ii);
  CHECK_INPUT(jj);

  return inertial_ba_cuda(poses, disps, intrinsics, Tij, Tibj, Tcb, Hint, vint, targets, weights,
                          eta, ii, jj, t0, t1, iterations, lm, ep, fix_pose, gblind);

}

std::vector<torch::Tensor> pgba(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor eta,
    torch::Tensor Hs,
    torch::Tensor vs,
    torch::Tensor Eii,
    torch::Tensor Eij,
    torch::Tensor Cii,
    torch::Tensor wi,
    torch::Tensor Hsp,
    torch::Tensor vsp,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor iip,
    torch::Tensor jjp,
    const int t0,
    const int t1,
    const float lm,
    const float ep,
    const bool use_vision_constraints) {

  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(ii);
  CHECK_INPUT(jj);

  return pgba_cuda(poses, disps, eta,
                Hs, vs, Eii, Eij, Cii, wi,
                Hsp, vsp, ii, jj, iip, jjp, t0, t1, lm, ep,
                use_vision_constraints);

}


torch::Tensor frame_distance(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ii,
    torch::Tensor jj,
    const float beta) {

  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);
  CHECK_INPUT(ii);
  CHECK_INPUT(jj);

  return frame_distance_cuda(poses, disps, intrinsics, ii, jj, beta);

}

torch::Tensor covis_distance(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ii) {

  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);
  CHECK_INPUT(ii);

  return covis_distance_cuda(poses, disps, intrinsics, ii);

}

torch::Tensor iproj(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics) {
  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);

  return iproj_cuda(poses, disps, intrinsics);
}

std::vector<torch::Tensor> bi_inter(
    torch::Tensor scales,
    torch::Tensor grids) {
  CHECK_INPUT(scales);
  CHECK_INPUT(grids);

  return bi_inter_cuda(scales, grids);
}

// c++ python binding
std::vector<torch::Tensor> corr_index_forward(
    torch::Tensor volume,
    torch::Tensor coords,
    int radius) {
  CHECK_INPUT(volume);
  CHECK_INPUT(coords);

  return corr_index_cuda_forward(volume, coords, radius);
}

std::vector<torch::Tensor> corr_index_backward(
    torch::Tensor volume,
    torch::Tensor coords,
    torch::Tensor corr_grad,
    int radius) {
  CHECK_INPUT(volume);
  CHECK_INPUT(coords);
  CHECK_INPUT(corr_grad);

  auto volume_grad = corr_index_cuda_backward(volume, coords, corr_grad, radius);
  return {volume_grad};
}

std::vector<torch::Tensor> altcorr_forward(
    torch::Tensor fmap1,
    torch::Tensor fmap2,
    torch::Tensor coords,
    int radius) {
  CHECK_INPUT(fmap1);
  CHECK_INPUT(fmap2);
  CHECK_INPUT(coords);

  return altcorr_cuda_forward(fmap1, fmap2, coords, radius);
}

std::vector<torch::Tensor> altcorr_backward(
    torch::Tensor fmap1,
    torch::Tensor fmap2,
    torch::Tensor coords,
    torch::Tensor corr_grad,
    int radius) {
  CHECK_INPUT(fmap1);
  CHECK_INPUT(fmap2);
  CHECK_INPUT(coords);
  CHECK_INPUT(corr_grad);

  return altcorr_cuda_backward(fmap1, fmap2, coords, corr_grad, radius);
}


torch::Tensor depth_filter(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ix,
    torch::Tensor thresh) {

    CHECK_INPUT(poses);
    CHECK_INPUT(disps);
    CHECK_INPUT(intrinsics);
    CHECK_INPUT(ix);
    CHECK_INPUT(thresh);

    return depth_filter_cuda(poses, disps, intrinsics, ix, thresh);
}


std::vector<torch::Tensor> bias_prior_factors(
    torch::Tensor biass,
    torch::Tensor ii,
    const float preint_scale,
    const int D) {

  CHECK_INPUT(biass);
  CHECK_INPUT(ii);

  return bias_prior_factors_cuda(biass, ii, preint_scale, D);
}


std::vector<torch::Tensor> bias_factors(
    torch::Tensor biass,
    torch::Tensor info2s,
    torch::Tensor ii,
    torch::Tensor jj,
    const float preint_scale,
    const int D) {

  CHECK_INPUT(biass);
  CHECK_INPUT(info2s);
  CHECK_INPUT(ii);
  CHECK_INPUT(jj);

  return bias_factors_cuda(biass, info2s, ii, jj, preint_scale, D);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // bundle adjustment kernels
  m.def("ba", &ba, "bundle adjustment");
  m.def("inertial_ba", &inertial_ba, "Inertial Bundle Adjustment");
  m.def("pgba", &pgba, "pose graph bundle adjustment");
  m.def("proj_trans", &proj_trans, "projective transform");
  m.def("frame_distance", &frame_distance, "frame_distance");
  m.def("covis_distance", &covis_distance, "covis_distance");
  m.def("depth_filter", &depth_filter, "depth_filter");
  m.def("iproj", &iproj, "back projection");
  m.def("bi_inter", &bi_inter, "bilinear interpolation");

  // IMU bias factor kernels
  m.def("bias_prior_factors", &bias_prior_factors, "compute bias prior factors for inertial BA",
        py::arg("biass"), py::arg("ii"), py::arg("preint_scale"), py::arg("D"));
  m.def("bias_factors", &bias_factors, "compute bias consistency factors for inertial BA",
        py::arg("biass"), py::arg("info2s"), py::arg("ii"), py::arg("jj"),
        py::arg("preint_scale"), py::arg("D"));

  // correlation volume kernels
  m.def("altcorr_forward", &altcorr_forward, "ALTCORR forward");
  m.def("altcorr_backward", &altcorr_backward, "ALTCORR backward");
  m.def("corr_index_forward", &corr_index_forward, "INDEX forward");
  m.def("corr_index_backward", &corr_index_backward, "INDEX backward");
}