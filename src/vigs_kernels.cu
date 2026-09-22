// The CUDA bundle-adjustment kernels of the base system (DROID-SLAM / VIGS-SLAM): dense
// projective residuals, the depth Schur complement, the sparse Cholesky solve and the
// retractions, plus the loop-closure (Sim3) and the IMU bias factors. Elevator-VIGS changes
// only the state width: inertial_ba_cuda takes the pre-assembled IMU blocks at 15 or 17
// columns, the 17-wide layout [pose6|vel3|bias6|u@15|h@16] carrying the transport state of
// the paper's Eq. (4), and the Schur terms are evaluated over the six pose rows only.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

typedef Eigen::Triplet<double> T;



#define MIN_DEPTH 0.25

#define THREADS 256
#define NUM_BLOCKS(batch_size) ((batch_size + THREADS - 1) / THREADS)


#define GPU_1D_KERNEL_LOOP(k, n) \
  for (size_t k = threadIdx.x; k<n; k += blockDim.x)


__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid +  8];
  sdata[tid] += sdata[tid +  4];
  sdata[tid] += sdata[tid +  2];
  sdata[tid] += sdata[tid +  1];
}

__device__ void blockReduce(volatile float *sdata) {
  unsigned int tid = threadIdx.x;
  __syncthreads();

  if (threadIdx.x < 128) {sdata[tid] += sdata[tid + 128]; } __syncthreads();
  if (threadIdx.x <  64) {sdata[tid] += sdata[tid +  64]; } __syncthreads();

  if (tid < 32) warpReduce(sdata, tid);
  __syncthreads();
}

__device__ float2 proj(const float *Xj, const float *intrinsics)
{
  float2 xnyn;
  xnyn.x = intrinsics[0] * (Xj[0] / Xj[2]) + intrinsics[2];
  xnyn.y = intrinsics[1] * (Xj[1] / Xj[2]) + intrinsics[3];
  return xnyn;
}

__device__ void iproj(float u, float v, const float *intrinsics, float * X, float di)
{
  X[0] = (u - intrinsics[2]) / intrinsics[0];
  X[1] = (v - intrinsics[3]) / intrinsics[1];
  X[2] = 1;
  X[3] = di;
}



__device__ void
actSO3(const float *q, const float *X, float *Y) {
  float uv[3];
  uv[0] = 2.0 * (q[1]*X[2] - q[2]*X[1]);
  uv[1] = 2.0 * (q[2]*X[0] - q[0]*X[2]);
  uv[2] = 2.0 * (q[0]*X[1] - q[1]*X[0]);

  Y[0] = X[0] + q[3]*uv[0] + (q[1]*uv[2] - q[2]*uv[1]);
  Y[1] = X[1] + q[3]*uv[1] + (q[2]*uv[0] - q[0]*uv[2]);
  Y[2] = X[2] + q[3]*uv[2] + (q[0]*uv[1] - q[1]*uv[0]);
}

__device__  void
actSE3(const float *t, const float *q, const float *X, float *Y) {
  actSO3(q, X, Y);
  Y[3] = X[3];
  Y[0] += X[3] * t[0];
  Y[1] += X[3] * t[1];
  Y[2] += X[3] * t[2];
}

__device__ void
adjSE3(const float *t, const float *q, const float *X, float *Y) {
  float qinv[4] = {-q[0], -q[1], -q[2], q[3]};
  actSO3(qinv, &X[0], &Y[0]);
  actSO3(qinv, &X[3], &Y[3]);

  float u[3], v[3];
  u[0] = t[2]*X[1] - t[1]*X[2];
  u[1] = t[0]*X[2] - t[2]*X[0];
  u[2] = t[1]*X[0] - t[0]*X[1];

  actSO3(qinv, u, v);
  Y[3] += v[0];
  Y[4] += v[1];
  Y[5] += v[2];
}

__device__ void 
relSE3(const float *ti, const float *qi, const float *tj, const float *qj, float *tij, float *qij) {
  qij[0] = -qj[3] * qi[0] + qj[0] * qi[3] - qj[1] * qi[2] + qj[2] * qi[1],
  qij[1] = -qj[3] * qi[1] + qj[1] * qi[3] - qj[2] * qi[0] + qj[0] * qi[2],
  qij[2] = -qj[3] * qi[2] + qj[2] * qi[3] - qj[0] * qi[1] + qj[1] * qi[0],
  qij[3] =  qj[3] * qi[3] + qj[0] * qi[0] + qj[1] * qi[1] + qj[2] * qi[2],

  actSO3(qij, ti, tij);
  tij[0] = tj[0] - tij[0];
  tij[1] = tj[1] - tij[1];
  tij[2] = tj[2] - tij[2];
}

  
__device__ void
expSO3(const float *phi, float* q) {
  // SO3 exponential map
  float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  float theta_p4 = theta_sq * theta_sq;

  float theta = sqrtf(theta_sq);
  float imag, real;

  if (theta_sq < 1e-8) {
    imag = 0.5 - (1.0/48.0)*theta_sq + (1.0/3840.0)*theta_p4;
    real = 1.0 - (1.0/ 8.0)*theta_sq + (1.0/ 384.0)*theta_p4;
  } else {
    imag = sinf(0.5 * theta) / theta;
    real = cosf(0.5 * theta);
  }

  q[0] = imag * phi[0];
  q[1] = imag * phi[1];
  q[2] = imag * phi[2];
  q[3] = real;

}

__device__ void
crossInplace(const float* a, float *b) {
  float x[3] = {
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0], 
  };

  b[0] = x[0];
  b[1] = x[1];
  b[2] = x[2];
}

__device__ void
expSE3(const float *xi, float* t, float* q) {
  // SE3 exponential map

  expSO3(xi + 3, q);
  float tau[3] = {xi[0], xi[1], xi[2]};
  float phi[3] = {xi[3], xi[4], xi[5]};

  float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  float theta = sqrtf(theta_sq);

  t[0] = tau[0]; 
  t[1] = tau[1]; 
  t[2] = tau[2];

  if (theta > 1e-4) {
    float a = (1 - cosf(theta)) / theta_sq;
    crossInplace(phi, tau);
    t[0] += a * tau[0];
    t[1] += a * tau[1];
    t[2] += a * tau[2];

    float b = (theta - sinf(theta)) / (theta * theta_sq);
    crossInplace(phi, tau);
    t[0] += b * tau[0];
    t[1] += b * tau[1];
    t[2] += b * tau[2];
  }
}


__global__ void projective_transform2_kernel(
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> target,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Cii,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> bz)
{
  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;

  const int ht = disps.size(1);
  const int wd = disps.size(2);

  int ix = static_cast<int>(ii[block_id]);
  int jx = static_cast<int>(jj[block_id]);

  __shared__ float intrinsics_[4];
  __shared__ float fx;
  __shared__ float fy;

  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];

  // load intrinsics from global memory
  if (thread_id == 0) {
    intrinsics_[0] = intrinsics[0][0];
    intrinsics_[1] = intrinsics[0][1];
    intrinsics_[2] = intrinsics[0][2];
    intrinsics_[3] = intrinsics[0][3];
    fx = intrinsics[0][0];
    fy = intrinsics[0][1];
  }

  __syncthreads();

  // load poses from global memory
  if (thread_id < 3) {
    ti[thread_id] = poses[ix][thread_id];
    tj[thread_id] = poses[jx][thread_id];
  }

  if (thread_id < 4) {
    qi[thread_id] = poses[ix][thread_id+3];
    qj[thread_id] = poses[jx][thread_id+3];
  }

  __syncthreads();

  if (thread_id == 0) {
    relSE3(ti, qi, tj, qj, tij, qij);
  }

  __syncthreads();

  //points 
  float Xi[4];
  float Xj[4];
  float2 xnyn;

  // jacobians
  float Jp[6];
  float Jz;

  __syncthreads();

  GPU_1D_KERNEL_LOOP(k, ht*wd) {

    const int i = k / wd;
    const int j = k % wd;

    const float u = static_cast<float>(j);
    const float v = static_cast<float>(i);
    
    // homogenous coordinates
    iproj(u, v, intrinsics_, Xi, disps[ix][i][j]);

    // transform homogenous point
    actSE3(tij, qij, Xi, Xj);

    xnyn = proj(Xj, intrinsics_);

    const float x = Xj[0];
    const float y = Xj[1];
    const float z = (Xj[2] < MIN_DEPTH) ? 0.0 : Xj[2];
    const float h = Xj[3];

    const float d = (Xj[2] < MIN_DEPTH) ? 0.0 : 1.0 / Xj[2];
    const float d2 = d * d;

    float wu = (Xj[2] < MIN_DEPTH) ? 0.0 : .001 * weight[block_id][0][i][j];
    float wv = (Xj[2] < MIN_DEPTH) ? 0.0 : .001 * weight[block_id][1][i][j];
    const float ru = target[block_id][0][i][j] - xnyn.x;
    const float rv = target[block_id][1][i][j] - xnyn.y;

    // assume pinhole
    Jp[0] = fx * d;
    Jp[1] = 0;
    Jp[2] = fx * (-x * d2);
    Jp[3] = 0;
    Jp[4] = fy * d;
    Jp[5] = fy * (-y * d2);

    // x - coordinate
    Jz = Jp[0] * tij[0] + Jp[1] * tij[1] + Jp[2] * tij[2];
    Cii[block_id][k] = wu * Jz * Jz;
    bz[block_id][k] = wu * ru * Jz;

    // y - coordinate
    Jz = Jp[3] * tij[0] + Jp[4] * tij[1] + Jp[5] * tij[2];
    Cii[block_id][k] += wv * Jz * Jz;
    bz[block_id][k] += wv * rv * Jz;
  }
}

__global__ void projective_transform_kernel(
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> target,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Hs,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> vs,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Eii,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Eij,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Cii,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> bz)
{
  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;

  const int ht = disps.size(1);
  const int wd = disps.size(2);

  int ix = static_cast<int>(ii[block_id]);
  int jx = static_cast<int>(jj[block_id]);

  __shared__ float intrinsics_[4];
  __shared__ float fx;
  __shared__ float fy;

  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];

  // load intrinsics from global memory
  if (thread_id == 0) {
    intrinsics_[0] = intrinsics[0];
    intrinsics_[1] = intrinsics[1];
    intrinsics_[2] = intrinsics[2];
    intrinsics_[3] = intrinsics[3];
    fx = intrinsics[0];
    fy = intrinsics[1];
  }

  __syncthreads();

  // load poses from global memory
  if (thread_id < 3) {
    ti[thread_id] = poses[ix][thread_id];
    tj[thread_id] = poses[jx][thread_id];
  }

  if (thread_id < 4) {
    qi[thread_id] = poses[ix][thread_id+3];
    qj[thread_id] = poses[jx][thread_id+3];
  }

  __syncthreads();

  if (thread_id == 0) {
    relSE3(ti, qi, tj, qj, tij, qij);
  }

  __syncthreads();

  //points 
  float Xi[4];
  float Xj[4];
  float2 xnyn;

  // jacobians
  float Jp[6];
  float Jx[12];
  float Jz;

  float* Ji = &Jx[0];
  float* Jj = &Jx[6];

  // hessians
  float hij[12*(12+1)/2];

  float vi[6], vj[6];

  int l;
  for (l=0; l<12*(12+1)/2; l++) {
    hij[l] = 0;
  }

  for (int n=0; n<6; n++) {
    vi[n] = 0;
    vj[n] = 0;
  }

  __syncthreads();

  GPU_1D_KERNEL_LOOP(k, ht*wd) {

    const int i = k / wd;
    const int j = k % wd;

    const float u = static_cast<float>(j);
    const float v = static_cast<float>(i);
    
    // homogenous coordinates
    iproj(u, v, intrinsics_, Xi, disps[ix][i][j]);

    // transform homogenous point
    actSE3(tij, qij, Xi, Xj);

    xnyn = proj(Xj, intrinsics_);

    const float x = Xj[0];
    const float y = Xj[1];
    const float z = (Xj[2] < MIN_DEPTH) ? 0.0 : Xj[2];
    const float h = Xj[3];

    const float d = (Xj[2] < MIN_DEPTH) ? 0.0 : 1.0 / Xj[2];
    const float d2 = d * d;

    float wu = (Xj[2] < MIN_DEPTH) ? 0.0 : .001 * weight[block_id][0][i][j];
    float wv = (Xj[2] < MIN_DEPTH) ? 0.0 : .001 * weight[block_id][1][i][j];
    const float ru = target[block_id][0][i][j] - xnyn.x;
    const float rv = target[block_id][1][i][j] - xnyn.y;

    // assume pinhole
    Jp[0] = fx * d;
    Jp[1] = 0;
    Jp[2] = fx * (-x * d2);
    Jp[3] = 0;
    Jp[4] = fy * d;
    Jp[5] = fy * (-y * d2);

    // x - coordinate
    Jj[0] = Jp[0] * h;
    Jj[1] = Jp[1] * h;
    Jj[2] = Jp[2] * h;
    Jj[3] = -Jp[1] * z + Jp[2] * y;
    Jj[4] =  Jp[0] * z - Jp[2] * x;
    Jj[5] = -Jp[0] * y + Jp[1] * x;

    Jz = Jp[0] * tij[0] + Jp[1] * tij[1] + Jp[2] * tij[2];
    Cii[block_id][k] = wu * Jz * Jz;
    bz[block_id][k] = wu * ru * Jz;

    if (ix == jx) wu = 0;

    adjSE3(tij, qij, Jj, Ji);
    for (int n=0; n<6; n++) Ji[n] *= -1;

    l=0;
    for (int n=0; n<12; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += wu * Jx[n] * Jx[m];
        l++;
      }
    }

    for (int n=0; n<6; n++) {
      vi[n] += wu * ru * Ji[n];
      vj[n] += wu * ru * Jj[n];

      Eii[block_id][n][k] = wu * Jz * Ji[n];
      Eij[block_id][n][k] = wu * Jz * Jj[n];
    }

    // y - coordinate
    Jj[0] = Jp[3] * h;
    Jj[1] = Jp[4] * h;
    Jj[2] = Jp[5] * h;
    Jj[3] = -Jp[4] * z + Jp[5] * y;
    Jj[4] =  Jp[3] * z - Jp[5] * x;
    Jj[5] = -Jp[3] * y + Jp[4] * x;

    Jz = Jp[3] * tij[0] + Jp[4] * tij[1] + Jp[5] * tij[2];
    Cii[block_id][k] += wv * Jz * Jz;
    bz[block_id][k] += wv * rv * Jz;

    if (ix == jx) wv = 0;

    adjSE3(tij, qij, Jj, Ji);
    for (int n=0; n<6; n++) Ji[n] *= -1;

    l=0;
    for (int n=0; n<12; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += wv * Jx[n] * Jx[m];
        l++;
      }
    }

    for (int n=0; n<6; n++) {
      vi[n] += wv * rv * Ji[n];
      vj[n] += wv * rv * Jj[n];

      Eii[block_id][n][k] += wv * Jz * Ji[n];
      Eij[block_id][n][k] += wv * Jz * Jj[n];
    }


  }

  __syncthreads();

  __shared__ float sdata[THREADS];
  for (int n=0; n<6; n++) {
    sdata[threadIdx.x] = vi[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      vs[0][block_id][n] = sdata[0];
    }

    __syncthreads();

    sdata[threadIdx.x] = vj[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      vs[1][block_id][n] = sdata[0];
    }

  }

  l=0;
  for (int n=0; n<12; n++) {
    for (int m=0; m<=n; m++) {
      sdata[threadIdx.x] = hij[l];
      blockReduce(sdata);

      if (threadIdx.x == 0) {
        if (n<6 && m<6) {
          Hs[0][block_id][n][m] = sdata[0];
          Hs[0][block_id][m][n] = sdata[0];
        }
        else if (n >=6 && m<6) {
          Hs[1][block_id][m][n-6] = sdata[0];
          Hs[2][block_id][n-6][m] = sdata[0];
        }
        else {
          Hs[3][block_id][n-6][m-6] = sdata[0];
          Hs[3][block_id][m-6][n-6] = sdata[0];
        }
      }

      l++;
    }
  }
}


__global__ void frame_distance_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> dist,
    const float beta) {

  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;

  const int ht = disps.size(1);
  const int wd = disps.size(2);

  __shared__ int ix;
  __shared__ int jx;

  __shared__ float intrinsics_[4];

  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];

  // load intrinsics from global memory
  if (thread_id == 0) {
    ix = static_cast<int>(ii[block_id]);
    jx = static_cast<int>(jj[block_id]);
    intrinsics_[0] = intrinsics[0];
    intrinsics_[1] = intrinsics[1];
    intrinsics_[2] = intrinsics[2];
    intrinsics_[3] = intrinsics[3];
  }

  __syncthreads();


  //points 
  float Xi[4];
  float Xj[4];

  __shared__ float accum[THREADS]; accum[thread_id] = 0;
  __shared__ float valid[THREADS]; valid[thread_id] = 0;
  __shared__ float total[THREADS]; total[thread_id] = 0;

  __syncthreads();

  for (int n=0; n<1; n++) {

    if (thread_id < 3) {
      ti[thread_id] = poses[ix][thread_id];
      tj[thread_id] = poses[jx][thread_id];
    }

    if (thread_id < 4) {
      qi[thread_id] = poses[ix][thread_id+3];
      qj[thread_id] = poses[jx][thread_id+3];
    }

    __syncthreads();


    relSE3(ti, qi, tj, qj, tij, qij);

    float d, du, dv;
    float2 xnyn;

    GPU_1D_KERNEL_LOOP(k, ht*wd) {
      const int i = k / wd;
      const int j = k % wd;

      const float u = static_cast<float>(j);
      const float v = static_cast<float>(i);


      // homogenous coordinates
      iproj(u, v, intrinsics_, Xi, disps[ix][i][j]);

      // transform homogenous point
      actSE3(tij, qij, Xi, Xj);

      xnyn = proj(Xj, intrinsics_);
      du = xnyn.x - u;
      dv = xnyn.y - v;
      d = sqrtf(du*du + dv*dv);

      total[threadIdx.x] += beta;
      
      if (Xj[2] > MIN_DEPTH) {
        accum[threadIdx.x] += beta * d;
        valid[threadIdx.x] += beta;
      }

      // homogenous coordinates
      iproj(u, v, intrinsics_, Xi, disps[ix][i][j]);

      Xj[0] = Xi[0] + Xi[3] * tij[0];
      Xj[1] = Xi[1] + Xi[3] * tij[1];
      Xj[2] = Xi[2] + Xi[3] * tij[2];

      xnyn = proj(Xj, intrinsics_);
      du = xnyn.x - u;
      dv = xnyn.y - v;
      d = sqrtf(du*du + dv*dv);

      total[threadIdx.x] += (1 - beta);
      
      if (Xj[2] > MIN_DEPTH) {
        accum[threadIdx.x] += (1 - beta) * d;
        valid[threadIdx.x] += (1 - beta);
      }
    }

    if (threadIdx.x == 0) {
      int tmp = ix;
      ix = jx;
      jx = tmp;
    }

    __syncthreads();

  }
  __syncthreads(); blockReduce(accum);
  __syncthreads(); blockReduce(total);
  __syncthreads(); blockReduce(valid);

  __syncthreads();

  if (thread_id == 0) {
    dist[block_id] = (valid[0] / (total[0] + 1e-8) < 0.75) ? 1000.0 : accum[0] / valid[0];
  }
}

__global__ void covis_distance_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> dist) {

  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;

  const int ht = disps.size(1);
  const int wd = disps.size(2);

  __shared__ int ix;
  __shared__ int jx1, jx2;

  __shared__ float intrinsics_[4];

  __shared__ float ti[3], tj1[3], tij1[3], tj2[3], tij2[3];
  __shared__ float qi[4], qj1[4], qij1[4], qj2[4], qij2[4];

  // load intrinsics from global memory
  if (thread_id == 0) {
    ix = static_cast<int>(ii[block_id]);
    jx1 = ix - 1;
    jx2 = ix + 1;
    intrinsics_[0] = intrinsics[0];
    intrinsics_[1] = intrinsics[1];
    intrinsics_[2] = intrinsics[2];
    intrinsics_[3] = intrinsics[3];
  }

  __syncthreads();

  //points 
  float Xi[4];
  float Xj1[4], Xj2[4];

  __shared__ float accum[THREADS]; accum[thread_id] = 0;

  __syncthreads();

  for (int n=0; n<1; n++) {

    if (thread_id < 3) {
      ti[thread_id] = poses[ix][thread_id];
      tj1[thread_id] = poses[jx1][thread_id];
      tj2[thread_id] = poses[jx2][thread_id];
    }

    if (thread_id < 4) {
      qi[thread_id] = poses[ix][thread_id+3];
      qj1[thread_id] = poses[jx1][thread_id+3];
      qj2[thread_id] = poses[jx2][thread_id+3];
    }

    __syncthreads();

    relSE3(ti, qi, tj1, qj1, tij1, qij1);
    relSE3(ti, qi, tj2, qj2, tij2, qij2);

    float2 xnyn1, xnyn2;

    GPU_1D_KERNEL_LOOP(k, ht*wd) {
      const int i = k / wd;
      const int j = k % wd;

      const float u = static_cast<float>(j);
      const float v = static_cast<float>(i);

      // homogenous coordinates
      iproj(u, v, intrinsics_, Xi, disps[ix][i][j]);

      // transform homogenous point
      actSE3(tij1, qij1, Xi, Xj1);
      actSE3(tij2, qij2, Xi, Xj2);

      xnyn1 = proj(Xj1, intrinsics_);
      xnyn2 = proj(Xj2, intrinsics_);
      bool out1 = xnyn1.x < 0 || xnyn1.x >= wd || xnyn1.y < 0 || xnyn1.y >= ht;
      bool out2 = xnyn2.x < 0 || xnyn2.x >= wd || xnyn2.y < 0 || xnyn2.y >= ht;
      if (out1 && out2) {
        accum[threadIdx.x] += 1;
      }
    }

    __syncthreads();
  }
  __syncthreads(); blockReduce(accum);
  __syncthreads();

  if (thread_id == 0) {
    dist[block_id] = accum[0] / (ht*wd);
  }
}



__global__ void depth_filter_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> inds,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> thresh,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> counter)
{

  const int block_id = blockIdx.x;
  const int neigh_id = blockIdx.y;
  const int index = blockIdx.z * blockDim.x + threadIdx.x;

  const int num = disps.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  __shared__ int ix;
  __shared__ int jx;

  __shared__ float intrinsics_[4];

  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];

  if (threadIdx.x == 0) {
    ix = static_cast<int>(inds[block_id]);
    jx = (neigh_id < 3) ? ix - neigh_id - 1 : ix + neigh_id;
    intrinsics_[0] = intrinsics[0];
    intrinsics_[1] = intrinsics[1];
    intrinsics_[2] = intrinsics[2];
    intrinsics_[3] = intrinsics[3];
  }

  __syncthreads();

  if (jx < 0 || jx >= num) {
    return;
  }

  const float t = thresh[block_id];

  // load poses from global memory
  if (threadIdx.x < 3) {
    ti[threadIdx.x] = poses[ix][threadIdx.x];
    tj[threadIdx.x] = poses[jx][threadIdx.x];
  }

  if (threadIdx.x < 4) {
    qi[threadIdx.x] = poses[ix][threadIdx.x+3];
    qj[threadIdx.x] = poses[jx][threadIdx.x+3];
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    relSE3(ti, qi, tj, qj, tij, qij);
  }

  //points 
  float Xi[4];
  float Xj[4];
  float2 xnyn;

  __syncthreads();

  if (index < ht*wd) {
    const int i = index / wd;
    const int j = index % wd;

    const float ui = static_cast<float>(j);
    const float vi = static_cast<float>(i);
    const float di = disps[ix][i][j];
    
    // homogenous coordinates
    iproj(ui, vi, intrinsics_, Xi, di);

    // transform homogenous point
    actSE3(tij, qij, Xi, Xj);

    xnyn = proj(Xj, intrinsics_);
    const float uj = xnyn.x;
    const float vj = xnyn.y;
    const float dj = Xj[3] / Xj[2];

    const int u0 = static_cast<int>(floor(uj));
    const int v0 = static_cast<int>(floor(vj));

    if (u0 >= 0 && v0 >= 0 && u0 < wd-1 && v0 < ht-1) {
      const float wx = ceil(uj) - uj;
      const float wy = ceil(vj) - vj;

      const float d00 = disps[jx][v0+0][u0+0];
      const float d01 = disps[jx][v0+0][u0+1];
      const float d10 = disps[jx][v0+1][u0+0];
      const float d11 = disps[jx][v0+1][u0+1];

      const float dj_hat = wy*wx*d00 + wy*(1-wx)*d01 + (1-wy)*wx*d10 + (1-wy)*(1-wx)*d11;

      const float err = abs(1.0/dj - 1.0/dj_hat);
      if       (abs(1.0/dj - 1.0/d00) < t) atomicAdd(&counter[block_id][i][j], 1.0f);
      else if  (abs(1.0/dj - 1.0/d01) < t) atomicAdd(&counter[block_id][i][j], 1.0f);
      else if  (abs(1.0/dj - 1.0/d10) < t) atomicAdd(&counter[block_id][i][j], 1.0f);
      else if  (abs(1.0/dj - 1.0/d11) < t) atomicAdd(&counter[block_id][i][j], 1.0f);
    }
  }
}



__global__ void iproj_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> points)

{

  const int block_id = blockIdx.x;
  const int index = blockIdx.y * blockDim.x + threadIdx.x;


  const int num = disps.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  __shared__ float intrinsics_[4];

  __shared__ float t[3];
  __shared__ float q[4];

  if (threadIdx.x == 0) {
    intrinsics_[0] = intrinsics[0];
    intrinsics_[1] = intrinsics[1];
    intrinsics_[2] = intrinsics[2];
    intrinsics_[3] = intrinsics[3];
  }

  __syncthreads();


  // load poses from global memory
  if (threadIdx.x < 3) {
    t[threadIdx.x] = poses[block_id][threadIdx.x];
  }

  if (threadIdx.x < 4) {
    q[threadIdx.x] = poses[block_id][threadIdx.x+3];
  }

  __syncthreads();

  //points 
  float Xi[4];
  float Xj[4];

  if (index < ht*wd) {
    const int i = index / wd;
    const int j = index % wd;

    const float ui = static_cast<float>(j);
    const float vi = static_cast<float>(i);
    const float di = disps[block_id][i][j];
    
    // homogenous coordinates
    iproj(ui, vi, intrinsics_, Xi, di);

    // transform homogenous point
    actSE3(t, q, Xi, Xj);

    points[block_id][i][j][0] = Xj[0] / Xj[3];
    points[block_id][i][j][1] = Xj[1] / Xj[3];
    points[block_id][i][j][2] = Xj[2] / Xj[3];

  }
}

__global__ void bi_inter_kernel(
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> scales,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> grids,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> outputs,
    torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> Jacobis)

{

  const int block_id = blockIdx.x;
  const int index = blockIdx.y * blockDim.x + threadIdx.x;

  const int num = grids.size(0);
  const int ht = grids.size(1);
  const int wd = grids.size(2);

  __syncthreads();

  if (index < ht*wd) {
    const int i = index / wd;
    const int j = index % wd;

    const float xind = grids[block_id][i][j][0];
    const float yind = grids[block_id][i][j][1];

    int x0 = floor(xind);
    int y0 = floor(yind);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float wa = (x1 - xind) * (y1 - yind);
    float wb = (x1 - xind) * (yind - y0);
    float wc = (xind - x0) * (y1 - yind);
    float wd = (xind - x0) * (yind - y0);

    outputs[block_id][i][j] = wa * scales[block_id][y0][x0] + wb * scales[block_id][y1][x0] + wc * scales[block_id][y0][x1] + wd * scales[block_id][y1][x1];
    Jacobis[block_id][i][j][y0][x0] = wa;
    Jacobis[block_id][i][j][y1][x0] = wb;
    Jacobis[block_id][i][j][y0][x1] = wc;
    Jacobis[block_id][i][j][y1][x1] = wd;
  }
}



__global__ void accum_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> inps,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ptrs,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> idxs,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> outs)
{
  
  const int block_id = blockIdx.x;
  const int D = inps.size(1);   // inps is rank 2: its row width is size(1)

  const int start = ptrs[block_id];
  const int end = ptrs[block_id+1];

  for (int k=threadIdx.x; k<D; k+=blockDim.x) {
    float x = 0;
    for (int i=start; i<end; i++) {
      x += inps[idxs[i]][k];
    }
    outs[block_id][k] = x;
  }  
}


__device__ void
retrSE3(const float *xi, const float* t, const float* q, float* t1, float* q1) {
  // retraction on SE3 manifold

  float dt[3] = {0, 0, 0};
  float dq[4] = {0, 0, 0, 1};
  
  expSE3(xi, dt, dq);

  q1[0] = dq[3] * q[0] + dq[0] * q[3] + dq[1] * q[2] - dq[2] * q[1];
  q1[1] = dq[3] * q[1] + dq[1] * q[3] + dq[2] * q[0] - dq[0] * q[2];
  q1[2] = dq[3] * q[2] + dq[2] * q[3] + dq[0] * q[1] - dq[1] * q[0];
  q1[3] = dq[3] * q[3] - dq[0] * q[0] - dq[1] * q[1] - dq[2] * q[2];

  actSO3(dq, t, t1);
  t1[0] += dt[0];
  t1[1] += dt[1];
  t1[2] += dt[2];
}


__global__ void pose_retr_kernel(
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> dx,
    const int t0, const int t1, const bool fix_pose)  
{

  for (int k=t0+threadIdx.x; k<t1; k+=blockDim.x) {
    if (fix_pose) {
      // Skip updating poses if fix_pose is true
      continue;
    }
    float xi[6], q[4], q1[4], t[3], t1[3];

    t[0] = poses[k][0];
    t[1] = poses[k][1];
    t[2] = poses[k][2];

    q[0] = poses[k][3];
    q[1] = poses[k][4];
    q[2] = poses[k][5];
    q[3] = poses[k][6];
    
    for (int n=0; n<6; n++) {
      xi[n] = dx[k-t0][n];
    }

    retrSE3(xi, t, q, t1, q1);

    poses[k][0] = t1[0];
    poses[k][1] = t1[1];
    poses[k][2] = t1[2];

    poses[k][3] = q1[0];
    poses[k][4] = q1[1];
    poses[k][5] = q1[2];
    poses[k][6] = q1[3];
  }
}

__global__ void disp_retr_kernel(
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> dz,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> inds) 
{
  const int i = inds[blockIdx.x];
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  for (int k=threadIdx.x; k<ht*wd; k+=blockDim.x) {
    float d = disps[i][k/wd][k%wd] + dz[blockIdx.x][k];
    disps[i][k/wd][k%wd] = d;
  }
}

torch::Tensor accum_cuda(torch::Tensor data, torch::Tensor ix, torch::Tensor jx) {
  torch::Tensor ix_cpu = ix.to(torch::kCPU);
  torch::Tensor jx_cpu = jx.to(torch::kCPU);
  torch::Tensor inds = torch::argsort(ix_cpu);

  long* ix_data = ix_cpu.data_ptr<long>();
  long* jx_data = jx_cpu.data_ptr<long>();
  long* kx_data = inds.data_ptr<long>();

  int count = jx.size(0);
  std::vector<int> cols;

  torch::Tensor ptrs_cpu = torch::zeros({count+1}, 
    torch::TensorOptions().dtype(torch::kInt64));
  
  long* ptrs_data = ptrs_cpu.data_ptr<long>();
  ptrs_data[0] = 0;

  int i = 0;
  for (int j=0; j<count; j++) {
    while (i < ix.size(0) && ix_data[kx_data[i]] <= jx_data[j]) {
      if (ix_data[kx_data[i]] == jx_data[j])
        cols.push_back(kx_data[i]);
      i++;
    }
    ptrs_data[j+1] = cols.size();
  }

  torch::Tensor idxs_cpu = torch::zeros({long(cols.size())}, 
    torch::TensorOptions().dtype(torch::kInt64));

  long* idxs_data = idxs_cpu.data_ptr<long>();

  for (int i=0; i<cols.size(); i++) {
    idxs_data[i] = cols[i];
  }

  torch::Tensor ptrs = ptrs_cpu.to(torch::kCUDA);
  torch::Tensor idxs = idxs_cpu.to(torch::kCUDA);

  torch::Tensor out = torch::zeros({jx.size(0), data.size(1)},
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  accum_kernel<<<count, THREADS>>>(
    data.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    ptrs.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    idxs.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    out.packed_accessor32<float,2,torch::RestrictPtrTraits>());

  return out;
}


// Depth-marginalisation (Schur) terms of the pose system. `d` = rows of E: 6 for the SE3
// pose (the projective kernels write only the pose rows of Eii/Eij, the state's vel/bias/
// transport columns have no depth coupling), 7 for the Sim3 PGBA. Fixed at compile time so
// the per-thread accumulator stays in registers and the block reduction runs d*d times;
// at the full state width (15/17) it would run 225/289 reductions for 36 nonzero entries.
template <int d>
__global__ void EEt6x6_kernel(
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> E,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Q,
    const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> idx,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> S)
{
  const int ix = idx[blockIdx.x][0];
  const int jx = idx[blockIdx.x][1];
  const int kx = idx[blockIdx.x][2];
  const int D = E.size(2);
  float dS[d][d];
  float ei[d];
  float ej[d];
  for (int i=0; i<d; i++) for (int j=0; j<d; j++) dS[i][j] = 0;
  for (int k=threadIdx.x; k<D; k+=blockDim.x) {
    const float q = Q[kx][k];
    for (int n=0; n<d; n++) { ei[n] = E[ix][n][k] * q; ej[n] = E[jx][n][k]; }
    for (int n=0; n<d; n++) for (int m=0; m<d; m++) dS[n][m] += ei[n] * ej[m];
  }
  __syncthreads();
  __shared__ float sdata[THREADS];
  for (int n=0; n<d; n++) for (int m=0; m<d; m++) {
    sdata[threadIdx.x] = dS[n][m];
    blockReduce(sdata);
    if (threadIdx.x == 0) S[blockIdx.x][n][m] = sdata[0];
  }
}

template <int d>
__global__ void Ev6x1_kernel(
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> E,
    const torch::PackedTensorAccessor32<float, 2,torch::RestrictPtrTraits> Q,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> w,
    const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> idx,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> v)
{
  const int D = E.size(2);
  const int kx = idx[blockIdx.x][0];
  float b[d];
  for (int n=0; n<d; n++) b[n] = 0.0;
  for (int k=threadIdx.x; k<D; k+=blockDim.x) {
    const float q_w = Q[kx][k] * w[kx][k];
    for (int n=0; n<d; n++) b[n] += q_w * E[blockIdx.x][n][k];
  }
  __syncthreads();
  __shared__ float sdata[THREADS];
  for (int n=0; n<d; n++) {
    sdata[threadIdx.x] = b[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) v[blockIdx.x][n] += sdata[0];
  }
}

__global__ void EvT6x1_kernel(
  const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> E,
  const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> x,
  const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> idx,
  torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> w)
{
  const int d = E.size(1);
  const int D = E.size(2);
  const int ix = idx[blockIdx.x];

  if (idx[blockIdx.x] <= 0 || idx[blockIdx.x] >= x.size(0))
    return;

  for (int k=threadIdx.x; k<D; k+=blockDim.x) {
    float dw = 0;
    for (int n=0; n<d; n++) {
      dw += E[blockIdx.x][n][k] * x[ix][n];
    }
    w[blockIdx.x][k] = dw;
  }
}

class SparseBlock {
  public:

    Eigen::SparseMatrix<double> A;
    Eigen::VectorX<double> b;

    SparseBlock(int N, int M) : N(N), M(M) {
      A = Eigen::SparseMatrix<double>(N*M, N*M);
      b = Eigen::VectorXd::Zero(N*M);
    }

    SparseBlock(Eigen::SparseMatrix<double> const& A, Eigen::VectorX<double> const& b, 
        int N, int M) : A(A), b(b), N(N), M(M) {}

    void update_lhs(torch::Tensor As, torch::Tensor ii, torch::Tensor jj) {

      auto As_cpu = As.to(torch::kCPU).to(torch::kFloat64);
      auto ii_cpu = ii.to(torch::kCPU).to(torch::kInt64);
      auto jj_cpu = jj.to(torch::kCPU).to(torch::kInt64);

      auto As_acc = As_cpu.accessor<double,3>();
      auto ii_acc = ii_cpu.accessor<long,1>();
      auto jj_acc = jj_cpu.accessor<long,1>();

      std::vector<T> tripletList;
      for (int n=0; n<ii.size(0); n++) {
        const int i = ii_acc[n];
        const int j = jj_acc[n];

        if (i >= 0 && j >= 0) {
          for (int k=0; k<M; k++) {
            for (int l=0; l<M; l++) {
              double val = As_acc[n][k][l];
              // Structural zeros stay out of the pattern: the vision blocks fill only their
              // 6x6 pose corner of D x D, so pushing every entry would make the sparse LLT's
              // pattern (and its fill-in) nearly dense. Every matrix diagonal is kept, so the
              // damping in solve() lands on it.
              if (val != 0.0 || (i == j && k == l))
                tripletList.push_back(T(M*i + k, M*j + l, val));
            }
          }
        }
      }
      A.setFromTriplets(tripletList.begin(), tripletList.end());
    }

    void update_rhs(torch::Tensor bs, torch::Tensor ii) {
      auto bs_cpu = bs.to(torch::kCPU).to(torch::kFloat64);
      auto ii_cpu = ii.to(torch::kCPU).to(torch::kInt64);

      auto bs_acc = bs_cpu.accessor<double,2>();
      auto ii_acc = ii_cpu.accessor<long,1>();

      for (int n=0; n<ii.size(0); n++) {
        const int i = ii_acc[n];
        if (i >= 0) {
          for (int j=0; j<M; j++) {
            b(i*M + j) += bs_acc[n][j];
          }
        }
      }
    }

    SparseBlock operator-(const SparseBlock& S) {
      return SparseBlock(A - S.A, b - S.b, N, M);
    }

    torch::Tensor solve(const float lm=0.0001, const float ep=0.1) {

      torch::Tensor dx;

      Eigen::SparseMatrix<double> L(A);
      L.diagonal().array() += ep + lm * L.diagonal().array();
      // Eigen's sparse diagonal() only touches entries that already exist. Zero-padded
      // scale rows/cols (from the IMU Schur pad7) may be absent from the sparsity
      // structure, so `+= ep` is silently a no-op there and leaves a zero diagonal, which
      // makes LLT fail. Inserting the missing entries with coeffRef() before damping fixes
      // it but is far too slow here, so diagonal() is kept and the Python side seeds those
      // entries with a very small value instead.

      Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
      solver.compute(L);

      if (solver.info() == Eigen::Success) {
        Eigen::VectorXd x = solver.solve(b);
        dx = torch::from_blob(x.data(), {N, M}, torch::TensorOptions()
          .dtype(torch::kFloat64)).to(torch::kCUDA).to(torch::kFloat32);
      }
      else {
        dx = torch::zeros({N, M}, torch::TensorOptions()
          .device(torch::kCUDA).dtype(torch::kFloat32));
      }
      
      return dx;
    }

  private:
    const int N;
    const int M;

};


SparseBlock schur_block(torch::Tensor E,
                        torch::Tensor Q,
                        torch::Tensor w,
                        torch::Tensor ii,
                        torch::Tensor jj,
                        torch::Tensor kk,
                        const int t0,
                        const int t1,
                        const int Dout)
{

  torch::Tensor ii_cpu = ii.to(torch::kCPU);
  torch::Tensor jj_cpu = jj.to(torch::kCPU);
  torch::Tensor kk_cpu = kk.to(torch::kCPU);

  const int d = E.size(1);      // rows of E: 6 (SE3 pose) or 7 (Sim3)
  const int D = Dout;           // state width of the block system (6 / 7 / 15 / 17)
  const int P = t1 - t0;
  const long* ii_data = ii_cpu.data_ptr<long>();
  const long* jj_data = jj_cpu.data_ptr<long>();
  const long* kk_data = kk_cpu.data_ptr<long>();

  std::vector<std::vector<long>> graph(P);
  std::vector<std::vector<long>> index(P);

  for (int n=0; n<ii_cpu.size(0); n++) {
    const int j = jj_data[n];
    const int k = kk_data[n];

    if (j >= t0 && j <= t1) {
      const int t = j - t0;
      graph[t].push_back(k);
      index[t].push_back(n);
    }
  }

  std::vector<long> ii_list, jj_list, idx;

  for (int i=0; i<P; i++) {
    for (int j=0; j<P; j++) {
      for (int k=0; k < graph[i].size(); k++) {
        for (int l=0; l < graph[j].size(); l++) {
          if (graph[i][k] == graph[j][l]) {
            ii_list.push_back(i);
            jj_list.push_back(j);

            idx.push_back(index[i][k]);
            idx.push_back(index[j][l]);
            idx.push_back(graph[i][k]);
          }
        }
      }
    }
  }

  torch::Tensor ix_cuda = torch::from_blob(idx.data(), {long(idx.size())}, 
    torch::TensorOptions().dtype(torch::kInt64)).to(torch::kCUDA).view({-1, 3});

  torch::Tensor jx_cuda = torch::stack({kk_cpu}, -1)
    .to(torch::kCUDA).to(torch::kInt64);

  torch::Tensor ii2_cpu = torch::from_blob(ii_list.data(), {long(ii_list.size())}, 
    torch::TensorOptions().dtype(torch::kInt64)).view({-1});

  torch::Tensor jj2_cpu = torch::from_blob(jj_list.data(), {long(jj_list.size())}, 
    torch::TensorOptions().dtype(torch::kInt64)).view({-1});

  torch::Tensor S = torch::zeros({ix_cuda.size(0), D, D}, 
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  torch::Tensor v = torch::zeros({jx_cuda.size(0), D},
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  TORCH_CHECK(d == 6 || d == 7, "schur_block: E must have 6 (SE3) or 7 (Sim3) rows, got ", d);
  if (d == 6) {
    EEt6x6_kernel<6><<<ix_cuda.size(0), THREADS>>>(
      E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Q.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      ix_cuda.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
      S.packed_accessor32<float,3,torch::RestrictPtrTraits>());
    Ev6x1_kernel<6><<<jx_cuda.size(0), THREADS>>>(
      E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Q.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      w.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      jx_cuda.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
      v.packed_accessor32<float,2,torch::RestrictPtrTraits>());
  } else {
    EEt6x6_kernel<7><<<ix_cuda.size(0), THREADS>>>(
      E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Q.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      ix_cuda.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
      S.packed_accessor32<float,3,torch::RestrictPtrTraits>());
    Ev6x1_kernel<7><<<jx_cuda.size(0), THREADS>>>(
      E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Q.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      w.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      jx_cuda.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
      v.packed_accessor32<float,2,torch::RestrictPtrTraits>());
  }

  // schur block
  SparseBlock A(P, D);
  A.update_lhs(S, ii2_cpu, jj2_cpu);
  A.update_rhs(v, jj_cpu - t0);

  return A;
}

std::vector<torch::Tensor> proj_trans_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor ii,
    torch::Tensor jj)
{
  auto opts = poses.options();
  const int num = ii.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  std::tuple<torch::Tensor, torch::Tensor> kuniq = torch::_unique(ii, true, true);
  torch::Tensor kx = std::get<0>(kuniq);

  // initialize buffers
  torch::Tensor Cii = torch::zeros({num, ht*wd}, opts);
  torch::Tensor wi = torch::zeros({num, ht*wd}, opts);

  projective_transform2_kernel<<<num, THREADS>>>(
    targets.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    weights.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    Cii.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    wi.packed_accessor32<float,2,torch::RestrictPtrTraits>());

  torch::Tensor C = accum_cuda(Cii, ii, kx);
  torch::Tensor w = accum_cuda(wi, ii, kx);

  return {C, w};
}


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
    const bool motion_only,
    const bool fix_pose)
{
  auto opts = poses.options();
  const int num = ii.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  torch::Tensor ts = torch::arange(t0, t1).to(torch::kCUDA);
  torch::Tensor ii_exp = torch::cat({ts, ii}, 0);
  torch::Tensor jj_exp = torch::cat({ts, jj}, 0);

  std::tuple<torch::Tensor, torch::Tensor> kuniq = 
    torch::_unique(ii_exp, true, true);

  torch::Tensor kx = std::get<0>(kuniq);
  torch::Tensor kk_exp = std::get<1>(kuniq);

  torch::Tensor dx = torch::zeros({t1 - t0, 6}, opts);
  torch::Tensor dz;
  torch::Tensor Linv, dzcov;

  // initialize buffers
  torch::Tensor Hs = torch::zeros({4, num, 6, 6}, opts);
  torch::Tensor vs = torch::zeros({2, num, 6}, opts);
  torch::Tensor Eii = torch::zeros({num, 6, ht*wd}, opts);
  torch::Tensor Eij = torch::zeros({num, 6, ht*wd}, opts);
  torch::Tensor Cii = torch::zeros({num, ht*wd}, opts);
  torch::Tensor wi = torch::zeros({num, ht*wd}, opts);

  for (int itr=0; itr<iterations; itr++) {

    projective_transform_kernel<<<num, THREADS>>>(
      targets.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      weights.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
      ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      Hs.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      vs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Eii.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Eij.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Cii.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      wi.packed_accessor32<float,2,torch::RestrictPtrTraits>());


    // pose x pose block
    SparseBlock A(t1 - t0, 6);

    A.update_lhs(Hs.reshape({-1, 6, 6}), 
        torch::cat({ii, ii, jj, jj}) - t0, 
        torch::cat({ii, jj, ii, jj}) - t0);

    A.update_rhs(vs.reshape({-1, 6}), 
        torch::cat({ii, jj}) - t0);

    if (motion_only) {
      dx = A.solve(lm, ep);

      // update poses
      pose_retr_kernel<<<1, THREADS>>>(
        poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(), t0, t1, fix_pose);
    }
    else if (fix_pose) {
      torch::Tensor C = accum_cuda(Cii, ii, kx);
      torch::Tensor w = accum_cuda(wi, ii, kx);
      torch::Tensor Q = 1.0 / (C + eta.view({-1, ht*wd}));
      dx = torch::zeros_like(dx, dx.options().device(torch::kCUDA));  // No pose updates
      dz = Q * w; 
      // update disparity maps
      disp_retr_kernel<<<kx.size(0), THREADS>>>(
        disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        dz.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        kx.packed_accessor32<long,1,torch::RestrictPtrTraits>());
        
    }
    else {
      torch::Tensor C = accum_cuda(Cii, ii, kx);
      torch::Tensor w = accum_cuda(wi, ii, kx);
      torch::Tensor Q = 1.0 / (C + eta.view({-1, ht*wd}));

      torch::Tensor Ei = accum_cuda(Eii.view({num, 6*ht*wd}), ii, ts).view({t1-t0, 6, ht*wd});
      torch::Tensor E = torch::cat({Ei, Eij}, 0);

      if (fix_pose) {
        dx = torch::zeros_like(dx, dx.options().device(torch::kCUDA));  // No pose updates
      } else {
        SparseBlock S = schur_block(E, Q, w, ii_exp, jj_exp, kk_exp, t0, t1, 6);
        dx = (A - S).solve(lm, ep);
      }

      torch::Tensor ix = jj_exp - t0;
      torch::Tensor dw = torch::zeros({ix.size(0), ht*wd}, opts);

      EvT6x1_kernel<<<ix.size(0), THREADS>>>(
        E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        ix.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
        dw.packed_accessor32<float,2,torch::RestrictPtrTraits>());

      if (fix_pose) {
        dz = Q * w;  // Simplified update when poses are fixed
      } else {
        dz = Q * (w - accum_cuda(dw, ii_exp, kx));
      }

      // update poses
      pose_retr_kernel<<<1, THREADS>>>(
        poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(), t0, t1, fix_pose);

      // update disparity maps
      disp_retr_kernel<<<kx.size(0), THREADS>>>(
        disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        dz.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        kx.packed_accessor32<long,1,torch::RestrictPtrTraits>());
    }

  }

  return {dx, dz, dzcov};
}


__global__ void body_centric_projective_transform_kernel(
  const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> target,
  const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> weight,
  const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
  const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
  const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Tij,
  const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Tibj,
  const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> Tcb,
  const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
  const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
  torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Hs,
  torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> vs,
  torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Eii,
  torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Eij,
  torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Cii,
  torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> bz)
{
const int block_id = blockIdx.x;
const int thread_id = threadIdx.x;

const int ht = disps.size(1);
const int wd = disps.size(2);

int ix = static_cast<int>(ii[block_id]);
int jx = static_cast<int>(jj[block_id]);

__shared__ float intrinsics_[4];

__shared__ float fx;
__shared__ float fy;

__shared__ float tij[3], tibj[3], tcb[3];
__shared__ float qij[4], qibj[4], qcb[4];
__shared__ float basesize;

// load intrinsics from global memory
if (thread_id == 0) {
  intrinsics_[0] = intrinsics[0];
  intrinsics_[1] = intrinsics[1];
  intrinsics_[2] = intrinsics[2];
  intrinsics_[3] = intrinsics[3];
  fx = intrinsics[0];
  fy = intrinsics[1];
}

__syncthreads();


if (thread_id < 3) {
  tij[thread_id] = Tij[block_id][thread_id];
  tibj[thread_id] = Tibj[block_id][thread_id];
  tcb[thread_id] = Tcb[thread_id];
}

if (thread_id < 4) {
  qij[thread_id] = Tij[block_id][thread_id+3];
  qibj[thread_id] = Tibj[block_id][thread_id+3];
  qcb[thread_id] = Tcb[thread_id+3];
}

__syncthreads();

if (thread_id == 0) {
  basesize = sqrtf(tij[0] * tij[0] + tij[1] * tij[1] + tij[2] * tij[2]) * 40;
  basesize = (basesize < 5.) ? 5. : basesize;
  basesize = (basesize > 100.) ? 100. : basesize;
  basesize = 1. / basesize;
}

__syncthreads();

//points 
float Xi[4];
float Xj[4];
float2 xnyn;

// jacobians
float Jp[6];
float Jx[12];
float Jjc[6];
float Jz;

float* Ji = &Jx[0];
float* Jj = &Jx[6];

// hessians
float hij[12*(12+1)/2];

float vi[6], vj[6];

int l;
for (l=0; l<12*(12+1)/2; l++) {
  hij[l] = 0;
}

for (int n=0; n<6; n++) {
  vi[n] = 0;
  vj[n] = 0;
}

__syncthreads();

GPU_1D_KERNEL_LOOP(k, ht*wd) {

  const int i = k / wd;
  const int j = k % wd;

  const float u = static_cast<float>(j);
  const float v = static_cast<float>(i);
  
  // homogenous coordinates
  float di = disps[ix][i][j];
  iproj(u, v, intrinsics_, Xi, di);

  // transform homogenous point
  actSE3(tij, qij, Xi, Xj);

  if (Xi[3] > basesize && Xj[3] > basesize) {

    xnyn = proj(Xj, intrinsics_);
    const float x = Xj[0];
    const float y = Xj[1];
    const float z = (Xj[2] < MIN_DEPTH) ? 0.0 : Xj[2];
    const float hb = Xj[3];

    float wu = (Xj[2] < MIN_DEPTH) ? 0.0 : .001 * weight[block_id][0][i][j];
    float wv = (Xj[2] < MIN_DEPTH) ? 0.0 : .001 * weight[block_id][1][i][j];
    const float ru = target[block_id][0][i][j] - xnyn.x;
    const float rv = target[block_id][1][i][j] - xnyn.y;

    const float d = (Xj[2] < MIN_DEPTH) ? 0.0 : 1.0 / Xj[2];
    const float d2 = d * d;
    Jp[0] = fx * d;
    Jp[1] = 0;
    Jp[2] = fx * (-x * d2);
    Jp[3] = 0;
    Jp[4] = fy * d;
    Jp[5] = fy * (-y * d2);

    Jz = Jp[0] * tij[0] + Jp[1] * tij[1] + Jp[2] * tij[2];
    Cii[block_id][k] = wu * Jz * Jz;
    bz[block_id][k] = wu * ru * Jz;

    if (ix != jx) {
      Jjc[0] = Jp[0] * hb;
      Jjc[1] = Jp[1] * hb;
      Jjc[2] = Jp[2] * hb;
      Jjc[3] = Jp[1] * -z + Jp[2] * y;
      Jjc[4] = Jp[0] * z + Jp[2] * -x;
      Jjc[5] = Jp[0] * -y + Jp[1] * x;

      adjSE3(tibj, qibj, Jjc, Ji);
      adjSE3(tcb, qcb, Jjc, Jj);
      for (int n=0; n<6; n++) Ji[n] *= -1;

      l=0;
      for (int n=0; n<12; n++) {
        for (int m=0; m<=n; m++) {
          hij[l] += wu * Jx[n] * Jx[m];
          l++;
        }
      }

      for (int n=0; n<6; n++) {
        vi[n] += wu * ru * Ji[n];
        vj[n] += wu * ru * Jj[n];

        Eii[block_id][n][k] = wu * Jz * Ji[n];
        Eij[block_id][n][k] = wu * Jz * Jj[n];
      }
    }

    // y - coordinate
    Jz = Jp[3] * tij[0] + Jp[4] * tij[1] + Jp[5] * tij[2];
    Cii[block_id][k] += wv * Jz * Jz;
    bz[block_id][k] += wv * rv * Jz;

    if (ix != jx) {
      Jjc[0] = Jp[3] * hb;
      Jjc[1] = Jp[4] * hb;
      Jjc[2] = Jp[5] * hb;
      Jjc[3] = Jp[4] * -z + Jp[5] * y;
      Jjc[4] = Jp[3] * z + Jp[5] * -x;
      Jjc[5] = Jp[3] * -y + Jp[4] * x;

      adjSE3(tibj, qibj, Jjc, Ji);
      adjSE3(tcb, qcb, Jjc, Jj);
      for (int n=0; n<6; n++) Ji[n] *= -1;

      l=0;
      for (int n=0; n<12; n++) {
        for (int m=0; m<=n; m++) {
          hij[l] += wv * Jx[n] * Jx[m];
          l++;
        }
      }

      for (int n=0; n<6; n++) {
        vi[n] += wv * rv * Ji[n];
        vj[n] += wv * rv * Jj[n];

        Eii[block_id][n][k] += wv * Jz * Ji[n];
        Eij[block_id][n][k] += wv * Jz * Jj[n];
      }
    }
  }
}

__syncthreads();

__shared__ float sdata[THREADS];
for (int n=0; n<6; n++) {
  sdata[threadIdx.x] = vi[n];
  blockReduce(sdata);
  if (threadIdx.x == 0) {
    vs[0][block_id][n] = sdata[0];
  }

  __syncthreads();

  sdata[threadIdx.x] = vj[n];
  blockReduce(sdata);
  if (threadIdx.x == 0) {
    vs[1][block_id][n] = sdata[0];
  }

}

l=0;
for (int n=0; n<12; n++) {
  for (int m=0; m<=n; m++) {
    sdata[threadIdx.x] = hij[l];
    blockReduce(sdata);

    if (threadIdx.x == 0) {
      if (n<6 && m<6) {
        Hs[0][block_id][n][m] = sdata[0];
        Hs[0][block_id][m][n] = sdata[0];
      }
      else if (n >=6 && m<6) {
        Hs[1][block_id][m][n-6] = sdata[0];
        Hs[2][block_id][n-6][m] = sdata[0];
      }
      else {
        Hs[3][block_id][n-6][m-6] = sdata[0];
        Hs[3][block_id][m-6][n-6] = sdata[0];
      }
    }

    l++;
  }
}
}


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
  torch::Tensor gblind
  )
{
auto opts = poses.options();
const int num = ii.size(0);
const int ht = disps.size(1);
const int wd = disps.size(2);
// State-block width, read off the pre-assembled IMU factor blocks rather than hard-coded,
// so Schur, marginalisation and the Cholesky solve below stay width-agnostic: 15 =
// [pose6|vel3|bias6] outside a ride, 17 = [pose6|vel3|bias6|u@15|h@16] from the departure
// until the fold, the (v^E, u, h) split of the paper's Eq. (4) that carries the elevator's
// motion jointly with vision instead of in a separate solve.
const int D = Hint.size(2);

torch::Tensor ts = torch::arange(t0, t1).to(torch::kCUDA);
torch::Tensor ii_exp = torch::cat({ts, ii}, 0);
torch::Tensor jj_exp = torch::cat({ts, jj}, 0);

std::tuple<torch::Tensor, torch::Tensor> kuniq = 
  torch::_unique(ii_exp, true, true);

torch::Tensor kx = std::get<0>(kuniq);
torch::Tensor kk_exp = std::get<1>(kuniq);
  
torch::Tensor dx;
torch::Tensor dz;

// initialize buffers
torch::Tensor Hs = torch::zeros({4, num, D, D}, opts);
torch::Tensor vs = torch::zeros({2, num, D}, opts);
torch::Tensor Eii = torch::zeros({num, 6, ht*wd}, opts);
torch::Tensor Eij = torch::zeros({num, 6, ht*wd}, opts);
torch::Tensor Cii = torch::zeros({num, ht*wd}, opts);
torch::Tensor wi = torch::zeros({num, ht*wd}, opts);

for (int itr=0; itr<iterations; itr++) {

  body_centric_projective_transform_kernel<<<num, THREADS>>>(
    targets.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    weights.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    Tij.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    Tibj.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    Tcb.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    Hs.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    vs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    Eii.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    Eij.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    Cii.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    wi.packed_accessor32<float,2,torch::RestrictPtrTraits>());

  // Gravity-blind visual factor: project the up axis out of vision's pose translation
  // sub-block, so vision neither constrains nor leaks into the vertical. gblind is (t1,3),
  // the per-KF unit up axis in the body pose tangent (R_bw @ e_z) on the rows to blind and
  // zero elsewhere. With P_k = blockdiag(I3 - d_k d_k^T, I3) this applies J_pose <- J_pose * P,
  // i.e. on each edge's 6x6 pose corner: Hs[0] <- Pi H Pi, Hs[1] <- Pi H Pj, Hs[2] <- Pj H Pi,
  // Hs[3] <- Pj H Pj, vs <- P vs, plus the depth-Schur cross rows Eii/Eij[0:3] <- P3 E, so the
  // marginalization Hs - E Q E^T cannot reintroduce the vertical. Zero rows give P = I, so a
  // cross-boundary edge blinds only its flagged endpoint. Elevator-VIGS keeps the vertical
  // in the transport state instead and passes an empty gblind (depth_video.inertial_ba),
  // which skips the whole block and leaves the solve bit-identical.
  if (gblind.size(0) > 0) {
    using torch::indexing::Slice;
    auto di = gblind.index_select(0, ii);                          // (num,3) i-endpoint up-dir
    auto dj = gblind.index_select(0, jj);                          // (num,3) j-endpoint up-dir
    auto I3 = torch::eye(3, opts);
    auto P3i = I3.unsqueeze(0) - di.unsqueeze(2) * di.unsqueeze(1); // (num,3,3) I - d d^T
    auto P3j = I3.unsqueeze(0) - dj.unsqueeze(2) * dj.unsqueeze(1);
    auto P6i = torch::eye(6, opts).unsqueeze(0).repeat({num, 1, 1});
    auto P6j = torch::eye(6, opts).unsqueeze(0).repeat({num, 1, 1});
    P6i.index_put_({Slice(), Slice(0,3), Slice(0,3)}, P3i);        // blockdiag(P3, I3)
    P6j.index_put_({Slice(), Slice(0,3), Slice(0,3)}, P3j);
    // pose 6x6 corner of each of the 4 vision blocks (clone to avoid in-place alias)
    auto h0 = Hs.index({0, Slice(), Slice(0,6), Slice(0,6)}).clone();
    auto h1 = Hs.index({1, Slice(), Slice(0,6), Slice(0,6)}).clone();
    auto h2 = Hs.index({2, Slice(), Slice(0,6), Slice(0,6)}).clone();
    auto h3 = Hs.index({3, Slice(), Slice(0,6), Slice(0,6)}).clone();
    Hs.index_put_({0, Slice(), Slice(0,6), Slice(0,6)}, torch::bmm(torch::bmm(P6i, h0), P6i));
    Hs.index_put_({1, Slice(), Slice(0,6), Slice(0,6)}, torch::bmm(torch::bmm(P6i, h1), P6j));
    Hs.index_put_({2, Slice(), Slice(0,6), Slice(0,6)}, torch::bmm(torch::bmm(P6j, h2), P6i));
    Hs.index_put_({3, Slice(), Slice(0,6), Slice(0,6)}, torch::bmm(torch::bmm(P6j, h3), P6j));
    // rhs (gradient) must be projected too, else the system is inconsistent
    auto v0 = vs.index({0, Slice(), Slice(0,6)}).clone();
    auto v1 = vs.index({1, Slice(), Slice(0,6)}).clone();
    vs.index_put_({0, Slice(), Slice(0,6)}, torch::bmm(P6i, v0.unsqueeze(2)).squeeze(2));
    vs.index_put_({1, Slice(), Slice(0,6)}, torch::bmm(P6j, v1.unsqueeze(2)).squeeze(2));
    // depth-Schur cross-term translation rows (rows 0:3); rotation rows 3:6 untouched
    auto e_i = Eii.index({Slice(), Slice(0,3), Slice()}).clone();  // (num,3,ht*wd)
    auto e_j = Eij.index({Slice(), Slice(0,3), Slice()}).clone();
    Eii.index_put_({Slice(), Slice(0,3), Slice()}, torch::bmm(P3i, e_i));
    Eij.index_put_({Slice(), Slice(0,3), Slice()}, torch::bmm(P3j, e_j));
  }

  // pose x pose block
  SparseBlock A(t1 - t0, D);

  // add constraints
  torch::Tensor Hs_all = torch::cat({Hs.reshape({-1,D,D}), Hint.reshape({-1,D,D})});
  torch::Tensor vs_all = torch::cat({vs.reshape({-1,D}), vint.reshape({-1,D})});

  torch::Tensor iii = torch::arange(t0-1, t1-1).to(torch::kCUDA);
  torch::Tensor jji = iii + 1;
  torch::Tensor ind1_all = torch::cat({ii, ii, jj, jj, iii, iii, jji, jji, iii, iii, jji, jji, iii, iii});
  torch::Tensor ind2_all = torch::cat({ii, jj, ii, jj, iii, jji, iii, jji, iii, jji, iii, jji, iii, iii});
  torch::Tensor ind3_all = torch::cat({ii, jj, iii, jji, iii, jji, iii});

  A.update_lhs(Hs_all, ind1_all - t0, ind2_all - t0);
  A.update_rhs(vs_all, ind3_all - t0);

  // solve system
  torch::Tensor C = accum_cuda(Cii, ii, kx);
  torch::Tensor w = accum_cuda(wi, ii, kx);
  torch::Tensor Q = 1.0 / (C + eta.view({-1, ht*wd}));

  torch::Tensor Ei = accum_cuda(Eii.view({num, 6*ht*wd}), ii, ts).view({t1-t0, 6, ht*wd});
  torch::Tensor E = torch::cat({Ei, Eij}, 0);

  SparseBlock S = schur_block(E, Q, w, ii_exp, jj_exp, kk_exp, t0, t1, D);
  dx = (A - S).solve(lm, ep);

  torch::Tensor ix = jj_exp - t0;
  torch::Tensor dw = torch::zeros({ix.size(0), ht*wd}, opts);

  EvT6x1_kernel<<<ix.size(0), THREADS>>>(
    E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    ix.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    dw.packed_accessor32<float,2,torch::RestrictPtrTraits>());

  dz = Q * (w - accum_cuda(dw, ii_exp, kx));

  // update poses
  pose_retr_kernel<<<1, THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(), t0, t1, fix_pose);

  // update disparity maps
  disp_retr_kernel<<<kx.size(0), THREADS>>>(
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    dz.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    kx.packed_accessor32<long,1,torch::RestrictPtrTraits>());
}

return dx;
}

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
    const bool use_vision_constraints)
{
  auto opts = poses.options();
  const int num = ii.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);
  const int D = 7;

  torch::Tensor ts = torch::arange(t0, t1).to(torch::kCUDA);
  torch::Tensor ii_exp = torch::cat({ts, ii}, 0);
  torch::Tensor jj_exp = torch::cat({ts, jj}, 0);

  std::tuple<torch::Tensor, torch::Tensor> kuniq =
    torch::_unique(ii_exp, true, true);

  torch::Tensor kx = std::get<0>(kuniq);
  torch::Tensor kk_exp = std::get<1>(kuniq);

  torch::Tensor dx, dz;
  torch::Tensor Linv, dzcov;

  // pose x pose block
  SparseBlock A(t1 - t0, D);

  // add constraints
  torch::Tensor iii = torch::arange(0, t1-1).to(torch::kCUDA);
  torch::Tensor jji = iii + 1;

  torch::Tensor Hs_all, vs_all, ind1_all, ind2_all, ind3_all;
  if (use_vision_constraints) {
    Hs_all = torch::cat({Hs.reshape({-1,D,D}), Hsp.reshape({-1,D,D})});
    vs_all = torch::cat({vs.reshape({-1,D}), vsp.reshape({-1,D})});
    ind1_all = torch::cat({ii, ii, jj, jj, iip, iip, jjp, jjp});
    ind2_all = torch::cat({ii, jj, ii, jj, iip, jjp, iip, jjp});
    ind3_all = torch::cat({ii, jj, iip, jjp});
  } else {
    Hs_all = Hsp.reshape({-1,D,D});
    vs_all = vsp.reshape({-1,D});
    ind1_all = torch::cat({iip, iip, jjp, jjp});
    ind2_all = torch::cat({iip, jjp, iip, jjp});
    ind3_all = torch::cat({iip, jjp});
  }

  A.update_lhs(Hs_all, ind1_all - t0, ind2_all - t0);
  A.update_rhs(vs_all, ind3_all - t0);

  // solve system
  torch::Tensor C = accum_cuda(Cii, ii, kx);
  torch::Tensor w = accum_cuda(wi, ii, kx);
  torch::Tensor Q = 1.0 / (C + eta.view({-1, ht*wd}));

  torch::Tensor Ei = accum_cuda(Eii.view({num, D*ht*wd}), ii, ts).view({t1-t0, D, ht*wd});
  torch::Tensor E = torch::cat({Ei, Eij}, 0);

  SparseBlock S = schur_block(E, Q, w, ii_exp, jj_exp, kk_exp, t0, t1, D);
  // When not using vision constraints, A contains only IMU/LC terms.
  // Subtracting the visual Schur complement S from a vision-free A makes
  // B = A - S indefinite (S can dominate), causing SimplicialLLT to fail.
  SparseBlock B = use_vision_constraints ? A - S : A;
  dx = B.solve(lm, ep);
  torch::Tensor ix = jj_exp - t0;
  torch::Tensor dw = torch::zeros({ix.size(0), ht*wd}, opts);

  EvT6x1_kernel<<<ix.size(0), THREADS>>>(
    E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    ix.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    dw.packed_accessor32<float,2,torch::RestrictPtrTraits>());

  dz = Q * (w - accum_cuda(dw, ii_exp, kx));

  // update disparity maps
  disp_retr_kernel<<<kx.size(0), THREADS>>>(
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    dz.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    kx.packed_accessor32<long,1,torch::RestrictPtrTraits>());

  return {dx, dz};
}


torch::Tensor frame_distance_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ii,
    torch::Tensor jj,
    const float beta)
{
  auto opts = poses.options();
  const int num = ii.size(0);

  torch::Tensor dist = torch::zeros({num}, opts);

  frame_distance_kernel<<<num, THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    dist.packed_accessor32<float,1,torch::RestrictPtrTraits>(), beta);

  return dist;
}

torch::Tensor covis_distance_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ii)
{
  auto opts = poses.options();
  const int num = ii.size(0);

  torch::Tensor dist = torch::zeros({num}, opts);

  covis_distance_kernel<<<num, THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    dist.packed_accessor32<float,1,torch::RestrictPtrTraits>());

  return dist;
}


torch::Tensor depth_filter_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ix,
    torch::Tensor thresh)
{
  const int num = ix.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  torch::Tensor counter = torch::zeros({num, ht, wd}, disps.options());

  dim3 blocks(num, 6, NUM_BLOCKS(ht * wd));

  depth_filter_kernel<<<blocks, THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    ix.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    thresh.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    counter.packed_accessor32<float,3,torch::RestrictPtrTraits>());

  return counter;
}


torch::Tensor iproj_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics)
{

  const int nm = disps.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  auto opts = disps.options();
  torch::Tensor points = torch::zeros({nm, ht, wd, 3}, opts);

  dim3 blocks(nm, NUM_BLOCKS(ht * wd));

  iproj_kernel<<<blocks, THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    points.packed_accessor32<float,4,torch::RestrictPtrTraits>());

  return points;

}

std::vector<torch::Tensor> bi_inter_cuda(
    torch::Tensor scales,
    torch::Tensor grids)
{

  const int nm = grids.size(0);
  const int ht = grids.size(1);
  const int wd = grids.size(2);
  const int hs = scales.size(1);
  const int ws = scales.size(2);

  auto opts = grids.options();
  torch::Tensor outputs = torch::zeros({nm, ht, wd}, opts);
  torch::Tensor Jacobis = torch::zeros({nm, ht, wd, hs, ws}, opts);

  dim3 blocks(nm, NUM_BLOCKS(ht * wd));

  bi_inter_kernel<<<blocks, THREADS>>>(
    scales.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    grids.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    outputs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    Jacobis.packed_accessor32<float,5,torch::RestrictPtrTraits>());

  return {outputs, Jacobis};

}


// ============================================================================
// Bias Prior Factors CUDA Kernel
// Computes H = J^T @ info @ J and v = J^T @ info @ err for bias prior factors
// where J is a sparse 6xD matrix with eye(6) in columns 9:15. D is always TRANSPORT_D=17
// ([pose6|vel3|bias6|u@15|h@16]), whose u/h columns the bias prior does not touch. Only the
// 9:15 rows/cols are ever written; the rest of H/v stays at the zero torch::zeros already
// put there.
// ============================================================================

__global__ void bias_prior_factors_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> biass,  // (N, 6) all biases
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,      // (num,) indices
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> H,            // (num, D, D) output Hessians, pre-zeroed
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> v,            // (num, D) output residuals, pre-zeroed
    const float preint_scale)
{
  const int k = blockIdx.x;  // edge index
  const int tid = threadIdx.x;

  if (k >= ii.size(0)) return;

  const long idx = ii[k];

  // Information matrix diagonal values (preint_scale already applied)
  // info = diag([1e3, 1e3, 1e3, 1e2, 1e2, 1e2]) * preint_scale
  const float info_vals[6] = {
    1e3f * preint_scale, 1e3f * preint_scale, 1e3f * preint_scale,
    1e2f * preint_scale, 1e2f * preint_scale, 1e2f * preint_scale
  };

  // Compute error: err = bias_ref - bias[idx] (bias_ref is biass[0])
  // J = [0_{6x9} | I_6], so:
  // H = J^T @ info @ J has only 6x6 block at [9:15, 9:15] = info
  // v = J^T @ info @ err has only 6 elements at [9:15] = info @ err

  // Only the 6 diagonal entries at [9:15, 9:15] and v[9:15] are non-zero; every other
  // element is already 0 from torch::zeros, so nothing else is written.
  if (tid < 6) {
    const float err = biass[0][tid] - biass[idx][tid];
    const float info_val = info_vals[tid];

    // H[9+tid, 9+tid] = info[tid, tid] (diagonal)
    H[k][9 + tid][9 + tid] = info_val;

    // v[9+tid] = info[tid,tid] * err[tid]
    v[k][9 + tid] = info_val * err;
  }
}

std::vector<torch::Tensor> bias_prior_factors_cuda(
    torch::Tensor biass,     // (N, 6) bias states for all keyframes
    torch::Tensor ii,        // (num,) indices for which to compute factors
    const float preint_scale,
    const int D)             // state width = TRANSPORT_D = 17 [pose6|vel3|bias6|u@15|h@16]
{
  auto opts = biass.options();
  const int num = ii.size(0);
  TORCH_CHECK(D == 17, "bias_prior_factors: state width D must be TRANSPORT_D=17, got ", D);

  // Output tensors: H is (num, D, D), v is (num, D)
  torch::Tensor H = torch::zeros({num, D, D}, opts);
  torch::Tensor v = torch::zeros({num, D}, opts);

  if (num > 0) {
    bias_prior_factors_kernel<<<num, THREADS>>>(
      biass.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      H.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      v.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      preint_scale);
  }

  // Add batch dimension to match Python API: (1, num, D, D), (1, num, D)
  return {H.unsqueeze(0), v.unsqueeze(0)};
}


// ============================================================================
// Bias Consistency Factors CUDA Kernel
// Computes H and v for bias consistency factors between consecutive frames
// J_i = [0_{6x9} | I_6], J_j = -J_i, padded to the state width D, always TRANSPORT_D=17
// ([pose6|vel3|bias6|u@15|h@16]), whose u/h columns bias consistency does not touch. Only
// the 9:15 block is ever written; the rest of H/v stays at the zero torch::zeros already
// put there.
// ============================================================================

__global__ void bias_factors_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> biass,   // (N, 6) all biases
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> info2s,  // (num, 6, 6) info matrices
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,       // (num,) source indices
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,       // (num,) target indices
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Hii,           // (num, D, D) output, pre-zeroed
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Hij,           // (num, D, D) output, pre-zeroed
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Hji,           // (num, D, D) output, pre-zeroed
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Hjj,           // (num, D, D) output, pre-zeroed
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> vi,            // (num, D) output, pre-zeroed
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> vj,            // (num, D) output, pre-zeroed
    const float preint_scale)
{
  const int k = blockIdx.x;  // edge index
  const int tid = threadIdx.x;

  if (k >= ii.size(0)) return;

  const long idx_i = ii[k];
  const long idx_j = jj[k];

  // Compute error: err = bias[j] - bias[i]
  // J_i = [0 | I_6], J_j = -J_i = [0 | -I_6]
  //
  // H_ii = J_i^T @ info @ J_i -> 9:15 x 9:15 block = info
  // H_ij = J_i^T @ info @ J_j -> 9:15 x 9:15 block = -info
  // H_ji = J_j^T @ info @ J_i -> 9:15 x 9:15 block = -info
  // H_jj = J_j^T @ info @ J_j -> 9:15 x 9:15 block = info
  // v_i = J_i^T @ info @ err -> 9:15 = info @ err
  // v_j = J_j^T @ info @ err -> 9:15 = -info @ err

  // Use shared memory for info matrix and error
  __shared__ float s_info[6][6];
  __shared__ float s_err[6];
  __shared__ float s_info_err[6];

  // Load info matrix and compute error (threads 0-35 load info, 0-5 compute error)
  if (tid < 36) {
    int row = tid / 6;
    int col = tid % 6;
    s_info[row][col] = preint_scale * info2s[k][row][col];
  }
  if (tid < 6) {
    s_err[tid] = biass[idx_j][tid] - biass[idx_i][tid];
  }
  __syncthreads();

  // Compute info @ err
  if (tid < 6) {
    float sum = 0.0f;
    for (int c = 0; c < 6; c++) {
      sum += s_info[tid][c] * s_err[c];
    }
    s_info_err[tid] = sum;
  }
  __syncthreads();

  // Fill the only non-zero part: the 6x6 block at [9:15, 9:15] and v[9:15].
  GPU_1D_KERNEL_LOOP(idx, 36) {
    const int r6 = idx / 6;
    const int c6 = idx % 6;
    const float info_val = s_info[r6][c6];

    Hii[k][9 + r6][9 + c6] = info_val;
    Hij[k][9 + r6][9 + c6] = -info_val;
    Hji[k][9 + r6][9 + c6] = -info_val;
    Hjj[k][9 + r6][9 + c6] = info_val;
  }
  if (tid < 6) {
    vi[k][9 + tid] = s_info_err[tid];
    vj[k][9 + tid] = -s_info_err[tid];
  }
}

std::vector<torch::Tensor> bias_factors_cuda(
    torch::Tensor biass,      // (N, 6) bias states for all keyframes
    torch::Tensor info2s,     // (num, 6, 6) info matrices from preintegrators
    torch::Tensor ii,         // (num,) source indices
    torch::Tensor jj,         // (num,) target indices
    const float preint_scale,
    const int D)              // state width = TRANSPORT_D = 17 [pose6|vel3|bias6|u@15|h@16]
{
  auto opts = biass.options();
  const int num = ii.size(0);
  TORCH_CHECK(D == 17, "bias_factors: state width D must be TRANSPORT_D=17, got ", D);

  // Output tensors
  torch::Tensor Hii = torch::zeros({num, D, D}, opts);
  torch::Tensor Hij = torch::zeros({num, D, D}, opts);
  torch::Tensor Hji = torch::zeros({num, D, D}, opts);
  torch::Tensor Hjj = torch::zeros({num, D, D}, opts);
  torch::Tensor vi = torch::zeros({num, D}, opts);
  torch::Tensor vj = torch::zeros({num, D}, opts);

  if (num > 0) {
    bias_factors_kernel<<<num, THREADS>>>(
      biass.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      info2s.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      Hii.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Hij.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Hji.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Hjj.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      vi.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      vj.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      preint_scale);
  }

  // Add batch dimension to match Python API
  return {Hii.unsqueeze(0), Hij.unsqueeze(0), Hji.unsqueeze(0), Hjj.unsqueeze(0),
          vi.unsqueeze(0), vj.unsqueeze(0)};
}
