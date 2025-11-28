// src/kernels.hpp
#pragma once
#include <sycl/sycl.hpp>

namespace kernels {

// v1.1: RMSNorm Forward
// x_norm = x / sqrt(mean(x^2) + epsilon)
void rms_norm_fwd(sycl::queue& q, const float* x, float* out, float* rstd_out,
                  int batch, int dim, float eps = 1e-5f) {
  q.parallel_for(sycl::range<1>(batch), [=](sycl::id<1> idx) {
     int b = idx[0];
     float sum_sq = 0.0f;
     for (int i = 0; i < dim; ++i) {
       sum_sq += x[b * dim + i] * x[b * dim + i];
     }
     float rstd = 1.0f / sycl::sqrt(sum_sq / dim + eps);
     rstd_out[b] = rstd;  // 儲存供 backward 使用
     for (int i = 0; i < dim; ++i) {
       out[b * dim + i] = x[b * dim + i] * rstd;
     }
   }).wait();
}

// v1.1: BitLinear Quantization (AbsMax)
// W_q = round(W / gamma) * gamma, where gamma = max(|W|)
void quantize_weights(sycl::queue& q, const float* w, float* w_q,
                      float* scale_out, int size) {
  // 簡化版：計算整體的 Max Abs (實際應用通常是 per-channel)
  // 注意：這裡為了演示使用單一 buffer reduction，實際高效能需使用 group
  // algorithm
  float max_val = 0.0f;
  // Host 端的簡單處理以簡化 demo 代碼複雜度，實際應全在 Device
  for (int i = 0; i < size; ++i) max_val = std::max(max_val, std::abs(w[i]));

  float scale = max_val > 1e-6f ? 1.0f / max_val : 1.0f;
  *scale_out = max_val;  // 存儲 scale (beta * gamma / Qb)

  q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
     int i = idx[0];
     float val = w[i] * scale;
     // Round to {-1, 0, 1}
     float q_val = sycl::round(val);
     if (q_val > 1.0f) q_val = 1.0f;
     if (q_val < -1.0f) q_val = -1.0f;
     w_q[i] = q_val;
   }).wait();
}

// v1.1: Spiking Unit Surrogate Gradient
// Forward: s = 1 if v > theta else 0
// Backward: grad_v = grad_s * (1 / (1 + (alpha * (v - theta))^2))
void spike_fwd_bwd(sycl::queue& q, const float* v, float* s,
                   const float* grad_s, float* grad_v, int size, float theta,
                   float alpha, bool is_backward) {
  q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
     int i = idx[0];
     if (!is_backward) {
       // Forward
       s[i] = (v[i] > theta) ? 1.0f : 0.0f;
     } else {
       // Backward (Surrogate)
       float h = alpha * (v[i] - theta);
       float surrogate_deriv = 1.0f / (1.0f + h * h);
       grad_v[i] = grad_s[i] * surrogate_deriv;
     }
   }).wait();
}

void sigmoid_fwd(sycl::queue& q, const float* x, float* out, int size) {
  q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
     float val = x[idx];
     // Fast sigmoid approximation or standard
     out[idx] = 1.0f / (1.0f + sycl::exp(-val));
   }).wait();
}

// v1.1: Fast Weights Update Kernel
// M_new = clamp(lambda * M + eta * gate * (k x v), -c, c)
// 為了平行化效率，我們以 M 的每個元素 (r, c) 為執行緒
// 並在 Kernel 內部對所有 MicroCircuit (num_circuits) 進行加總
void fast_weight_update(sycl::queue& q,
                        float* M,           // [batch, dim, dim]
                        const float* k,     // [batch, num_circuits, dim]
                        const float* v,     // [batch, num_circuits, dim]
                        const float* gate,  // [batch, num_circuits, 1]
                        int batch, int num_circuits, int dim, float lambda,
                        float eta, float clip_val) {
  q.parallel_for(sycl::range<3>(batch, dim, dim), [=](sycl::id<3> idx) {
     int b = idx[0];
     int r = idx[1];  // row index of M
     int c = idx[2];  // col index of M

     // 1. 讀取舊的 M 值並衰減
     int m_idx = b * dim * dim + r * dim + c;
     float m_val = M[m_idx] * lambda;

     // 2. 累加所有 MicroCircuit 的更新量 (Hebbian Update)
     // sum_{i} (g_i * k_i[r] * v_i[c])
     float update_sum = 0.0f;
     for (int i = 0; i < num_circuits; ++i) {
       // k, v shape: [batch, num_circuits, dim]
       int vec_offset = b * num_circuits * dim + i * dim;
       int gate_idx = b * num_circuits + i;

       float g = gate[gate_idx];
       float k_val = k[vec_offset + r];
       float v_val = v[vec_offset + c];

       update_sum += g * k_val * v_val;
     }

     // 3. 套用更新率
     m_val += eta * update_sum;

     // 4. 數值鉗制 (Clamp)
     m_val = sycl::clamp(m_val, -clip_val, clip_val);

     // 寫回
     M[m_idx] = m_val;
   }).wait();
}
}  // namespace kernels
