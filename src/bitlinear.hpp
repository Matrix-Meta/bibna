// src/bitlinear.hpp
#pragma once
#include "common.hpp"
#include "kernels.hpp"
#include <fstream>

class BitLinear
{
    int in_dim, out_dim;
    sycl::queue &q;

  public:
    Tensor weights;   // 浮點權重 (主)
    Tensor weights_q; // 量化權重
    float *scale;     // 量化比例

    // 反向傳播快取
    // 已移除內部的 rstd_cache，改由外部傳遞以支援 BPTT

    BitLinear(int in, int out, sycl::queue &queue)
        : in_dim(in), out_dim(out), q(queue), weights(in * out, queue), weights_q(in * out, queue)
    {
        weights.random_init();
        scale = sycl::malloc_shared<float>(1, q);
    }

    // 移動建構子
    BitLinear(BitLinear &&other) noexcept
        : in_dim(other.in_dim), out_dim(other.out_dim), q(other.q), weights(std::move(other.weights)),
          weights_q(std::move(other.weights_q)), scale(other.scale)
    {
        other.scale = nullptr;
    }

    // 刪除複製和移動賦值
    BitLinear(const BitLinear &) = delete;
    BitLinear &operator=(const BitLinear &) = delete;
    BitLinear &operator=(BitLinear &&) = delete;

    ~BitLinear()
    {
        if (scale)
            sycl::free(scale, q);
    }

    // 回傳: {輸出 Tensor, RMSNorm 統計 Tensor (rstd)}
    std::pair<Tensor, Tensor> forward(Tensor &input)
    {
        int batch = input.size / in_dim;

        // 1. RMSNorm
        Tensor x_norm(input.size, q);
        Tensor rstd(batch, q); // 為每一步分配新的 rstd
        kernels::rms_norm_fwd(q, input.data, x_norm.data, rstd.data, batch, in_dim);

        // 2. 量化權重
        kernels::quantize_weights(q, weights.data, weights_q.data, scale, weights.size);

        // 3. 矩陣乘法
        Tensor output(batch * out_dim, q);
        int d_in = in_dim;
        int d_out = out_dim;

        q.parallel_for(sycl::range<2>(batch, out_dim), [=, w_ptr = weights_q.data, x_ptr = x_norm.data,
                                                        out_ptr = output.data, scale_ptr = scale](sycl::id<2> idx) {
             int b = idx[0];
             int o = idx[1];
             float sum = 0.0f;
             for (int i = 0; i < d_in; ++i)
             {
                 sum += x_ptr[b * d_in + i] * w_ptr[o * d_in + i];
             }
             out_ptr[b * d_out + o] = sum * (*scale_ptr);
         });

        return {std::move(output), std::move(rstd)};
    }

    // 真實 STE 反向傳播 (Straight-Through Estimator)
    // 新增參數: input_tensor, rstd (從 forward 回傳)
    void backward(Tensor &grad_output, Tensor &input_tensor, Tensor &rstd)
    {
        int batch = grad_output.size / out_dim;
        int d_in = in_dim;
        int d_out = out_dim;

        // 0. 重建 RMSNorm 的輸出 (為了計算 dW)
        // 我們需要重新計算 x_norm = input * rstd
        // 為了節省記憶體，我們在這裡重新計算 x_norm，使用正確的 rstd。
        
        Tensor x_norm(input_tensor.size, q);
        
        q.parallel_for(sycl::range<1>(batch), [=, in_ptr = input_tensor.data, r_ptr = rstd.data, out_ptr = x_norm.data](sycl::id<1> idx) {
            int b = idx[0];
            float r = r_ptr[b];
            for(int i=0; i<d_in; ++i) {
                out_ptr[b*d_in + i] = in_ptr[b*d_in + i] * r;
            }
        });

        // 1. 計算輸入梯度 (dL/dx = dL/dy * W)
        if (input_tensor.grad)
        {
            // A. 先計算對 x_norm 的梯度
            Tensor grad_x_norm(input_tensor.size, q);
            q.fill(grad_x_norm.data, 0.0f, grad_x_norm.size).wait(); // 歸零
            
            q.parallel_for(sycl::range<2>(batch, d_in), [=, grad_y = grad_output.data, w = weights.data,
                                                         g_xn = grad_x_norm.data](sycl::id<2> idx) {
                 int b = idx[0];
                 int i = idx[1];
                 float sum = 0.0f;
                 for (int o = 0; o < d_out; ++o)
                 {
                     sum += grad_y[b * d_out + o] * w[o * d_in + i];
                 }
                 g_xn[b * d_in + i] += sum;
            });
            
            // B. 通過 RMSNorm 反向傳播
            kernels::rms_norm_bwd(q, grad_x_norm.data, input_tensor.data, input_tensor.grad, rstd.data, batch, d_in);
        }

        // 2. 計算權重梯度 (dL/dW = x_norm^T * dL/dy)
        q.parallel_for(sycl::range<2>(d_out, d_in), [=, grad_y = grad_output.data, x = x_norm.data,
                                                     w_grad = weights.grad, scale_ptr = scale](sycl::id<2> idx) {
             int o = idx[0];
             int i = idx[1];
             float sum = 0.0f;
             for (int b = 0; b < batch; ++b)
             {
                 sum += x[b * d_in + i] * grad_y[b * d_out + o];
             }
             w_grad[o * d_in + i] += sum * (*scale_ptr);
         });
    }

    void save(std::ofstream &out)
    {
        // 儲存原始FP32權重
        std::vector<float> h_w(weights.size);
        q.memcpy(h_w.data(), weights.data, weights.size * sizeof(float)).wait();
        out.write(reinterpret_cast<char *>(h_w.data()), h_w.size() * sizeof(float));
    }

    void load(std::ifstream &in)
    {
        std::vector<float> h_w(weights.size);
        in.read(reinterpret_cast<char *>(h_w.data()), h_w.size() * sizeof(float));
        q.memcpy(weights.data, h_w.data(), weights.size * sizeof(float)).wait();
        
        // 載入後重新量化以初始化 weights_q 和 scale
        kernels::quantize_weights(q, weights.data, weights_q.data, scale, weights.size);
    }
};
