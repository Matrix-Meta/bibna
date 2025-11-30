// src/kernels.hpp
#pragma once
#include <sycl/sycl.hpp>

namespace kernels
{

// v1.1: RMSNorm 前向傳播
// x_norm = x / sqrt(mean(x^2) + epsilon)
void rms_norm_fwd(sycl::queue &q, const float *x, float *out, float *rstd_out, int batch, int dim, float eps = 1e-5f)
{
    q.parallel_for(sycl::range<1>(batch), [=](sycl::id<1> idx) {
         int b = idx[0];
         float sum_sq = 0.0f;
         for (int i = 0; i < dim; ++i)
         {
             sum_sq += x[b * dim + i] * x[b * dim + i];
         }
         float rstd = 1.0f / sycl::sqrt(sum_sq / dim + eps);
         rstd_out[b] = rstd; // 儲存供 backward 使用
         for (int i = 0; i < dim; ++i)
         {
             out[b * dim + i] = x[b * dim + i] * rstd;
         }
     });
}

// v1.1: BitLinear 量化 (AbsMax)
// W_q = round(W / gamma) * gamma, where gamma = max(|W|)
void quantize_weights(sycl::queue &q, const float *w, float *w_q, float *scale_out, int size)
{
    // 使用 SYCL reduction 在 device 上計算 max(|W|)
    // 1. 初始化 scale_out 為 0
    q.fill(scale_out, 0.0f, 1);

    // 2. Reduction
    auto reduction = sycl::reduction(scale_out, sycl::maximum<float>());
    q.parallel_for(sycl::range<1>(size), reduction, [=](sycl::id<1> idx, auto &max_combiner) {
        max_combiner.combine(sycl::fabs(w[idx]));
    });

    // 3. 量化核心
    // 此核心會排在歸約操作之後執行 (假設佇列為順序執行或依賴關係已妥善處理)
    q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
         float max_val = *scale_out;
         float scale = max_val > 1e-6f ? 1.0f / max_val : 1.0f;
         
         int i = idx[0];
         float val = w[i] * scale;
         // 四捨五入到 {-1, 0, 1}
         float q_val = sycl::round(val);
         if (q_val > 1.0f)
             q_val = 1.0f;
         if (q_val < -1.0f)
             q_val = -1.0f;
         w_q[i] = q_val;
     });
}

// v1.1: 脈衝單元代理梯度
// 前向傳播: s = 1 if v > theta else 0
// 反向傳播: grad_v = grad_s * (1 / (1 + (alpha * (v - theta))^2))
void spike_fwd_bwd(sycl::queue &q, const float *v, float *s, const float *grad_s, float *grad_v, int size, float theta,
                   float alpha, bool is_backward)
{
    q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
         int i = idx[0];
         if (!is_backward)
         {
             // 前向傳播
             s[i] = (v[i] > theta) ? 1.0f : 0.0f;
         }
         else
         {
             // 反向傳播 (代理)
             float h = alpha * (v[i] - theta);
             float surrogate_deriv = 1.0f / (1.0f + h * h);
             grad_v[i] = grad_s[i] * surrogate_deriv;
         }
     });
}

void sigmoid_fwd(sycl::queue &q, const float *x, float *out, int size)
{
    q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
         float val = x[idx];
         // 快速 Sigmoid 近似或標準
         out[idx] = 1.0f / (1.0f + sycl::exp(-val));
     });
}

// v1.1: 快速權重更新核心
// M_new = clamp(lambda * M + eta * gate * (k x v), -c, c)
// 為了平行化效率，我們以 M 的每個元素 (r, c) 為執行緒
// 並在 Kernel 內部對所有 MicroCircuit (num_circuits) 進行加總
void fast_weight_update(sycl::queue &q,
                        float *M,          // [批次, 維度, 維度]
                        const float *k,    // [批次, 微電路數量, 維度]
                        const float *v,    // [批次, 微電路數量, 維度]
                        const float *gate, // [批次, 微電路數量, 1]
                        int batch, int num_circuits, int dim, float lambda, float eta, float clip_val)
{
    q.parallel_for(sycl::range<3>(batch, dim, dim), [=](sycl::id<3> idx) {
         int b = idx[0];
         int r = idx[1]; // M 的行索引
         int c = idx[2]; // M 的列索引

         // 1. 讀取舊的 M 值並衰減
         int m_idx = b * dim * dim + r * dim + c;
         float m_val = M[m_idx] * lambda;

         // 2. 累加所有 MicroCircuit 的更新量 (Hebbian Update)
         // sum_{i} (g_i * k_i[r] * v_i[c])
         float update_sum = 0.0f;
         for (int i = 0; i < num_circuits; ++i)
         {
             // k, v 形狀: [批次, 微電路數量, 維度]
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
     });
}

// v1.2: 交叉熵損失前向與反向傳播 (支援批次)
// logits: [Batch, Vocab]
// target_indices: [Batch]
// scratch_pad: [Batch, 2] -> (max_val, sum_exp) per batch item
void cross_entropy_fw_bw(sycl::queue &q, const float *logits, const int *target_indices, float *grad_logits, float *loss_out,
                         int batch, int vocab_size, float *scratch_pad)
{
    // 1. 尋找最大值 (Per row reduction)
    // 為了簡化實作並避免複雜的 group reduction，我們先用 naive 的方式
    // 每個 batch item 啟動一個 work item 負責找該 row 的 max? 不，這樣太慢 (serial within row).
    // 更好的方式：使用 nd_range parallel_for。
    // 但為了保持這份代碼的簡潔性 (且 vocab 只有 ~300)，我們可以使用簡單的層次化並行。
    
    // 優化：針對小 Vocab，每個 batch item 一個 thread 其實不慢。
    // 假設 Vocab < 1024，我們可以用一個 kernel 處理所有事情。
    
    q.parallel_for(sycl::range<1>(batch), [=](sycl::id<1> idx) {
        int b = idx[0];
        int base = b * vocab_size;
        
        // A. Find Max
        float max_val = -1e9f;
        for(int i=0; i<vocab_size; ++i) {
            max_val = sycl::fmax(max_val, logits[base + i]);
        }
        
        // B. Compute Sum Exp & Softmax & Loss & Grad
        float sum_exp = 0.0f;
        for(int i=0; i<vocab_size; ++i) {
            sum_exp += sycl::exp(logits[base + i] - max_val);
        }
        
        // C. Final Pass
        int target = target_indices[b];
        float batch_loss = 0.0f;
        
        for(int i=0; i<vocab_size; ++i) {
            float prob = sycl::exp(logits[base + i] - max_val) / sum_exp;
            
            // Gradient: p - y
            float y = (i == target) ? 1.0f : 0.0f;
            grad_logits[base + i] = (prob - y) / batch; // Normalize gradient by batch size immediately?
            // 通常 Loss = sum(losses) / batch. Gradient = dLoss/dx.
            // 所以 gradient 每個元素也要除以 batch。
            
            if (i == target) {
                batch_loss = -sycl::log(prob + 1e-9f);
            }
        }
        
        // 寫入 Loss (這裡我們寫入每個 batch 的 loss 到 loss_out 的不同位置，還是累加?)
        // 為了避免 atomic，我們讓 loss_out 成為 [Batch] 大小，由 CPU 或是另一個 kernel 加總。
        // 這裡為了簡單，假設 loss_out 是 [Batch] 大小的陣列。
        loss_out[b] = batch_loss;
    });
}

// v1.3: 融合微電路加法與脈衝
// v = u_in + u_rec
// s = (v > theta) ? 1 : 0
void fused_add_spike(sycl::queue &q, const float *u_in, const float *u_rec, float *v_out, float *s_out, int size, float theta) {
    q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
        int i = idx[0];
        float val = u_in[i] + u_rec[i];
        v_out[i] = val;
        s_out[i] = (val > theta) ? 1.0f : 0.0f;
    });
}

// v1.3: 融合殘差加法
// out = x + proj
void fused_residual_add(sycl::queue &q, const float *x, const float *proj, float *out, int size) {
    q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
        int i = idx[0];
        out[i] = x[i] + proj[i];
    });
}

// v1.4: RMSNorm 反向傳播
// dL/dx = (dL/dy * rstd) - (x * rstd^3 * sum(dL/dy * x) / dim)
void rms_norm_bwd(sycl::queue &q, const float *grad_y, const float *x, float *grad_x, const float *rstd, int batch, int dim) {
    q.parallel_for(sycl::range<1>(batch), [=](sycl::id<1> idx) {
        int b = idx[0];
        float current_rstd = rstd[b];
        
        // 1. 計算 sum(grad_y * x)
        float dot_sum = 0.0f;
        for(int i=0; i<dim; ++i) {
            dot_sum += grad_y[b*dim + i] * x[b*dim + i];
        }
        
        float term2 = dot_sum * current_rstd * current_rstd * current_rstd / dim;
        
        // 2. 計算 grad_x
        for(int i=0; i<dim; ++i) {
            grad_x[b*dim + i] = grad_y[b*dim + i] * current_rstd - x[b*dim + i] * term2;
        }
    });
}

// v1.5: 向量加法反向傳播 (複製梯度)
// z = x + y => dx = dz, dy = dz
void add_bwd(sycl::queue &q, const float *grad_z, float *grad_x, float *grad_y, int size) {
    q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
        int i = idx[0];
        float gz = grad_z[i];
        // 累積梯度 (因為 x 或 y 可能在其他地方也被使用，雖然在此架構中通常是唯一的)
        // 為了安全起見，我們假設這裡是覆蓋或者累積？
        // 通常 backprop 是累積到 grad buffer。
        // 這裡我們直接賦值，假設調用者處理好了累積，或者這是鏈中的第一步。
        // *修正*: 為了通用性，這裡應該是 atomic add，或者調用者確保 grad_x 已初始化。
        // 鑑於現有架構的簡單性，我們使用累積 +=
        // 需注意如果是首次計算，需先歸零。但我們假設 grad buffers 在 backward 開始前已歸零。
        
        // 但等等，atomic 對全域記憶體很慢。
        // 在 MicroCircuit 中，x 和 y 是不同的來源 (input, recurrent)。
        // 所以我們直接寫入可能覆蓋之前的梯度？
        // 不，grad_x 是 input.grad，grad_y 是 recurrent.grad。
        // 這些是此操作的唯一來源。
        grad_x[i] += gz;
        grad_y[i] += gz;
    });
}

// v1.6: CorticalHub 記憶體讀取前向傳播
// y = M * q
// M: [Batch, Dim, Dim]
// q: [Batch, N, Dim]
// out: [Batch, N, Dim]
void memory_read_fwd(sycl::queue &q, const float *M, const float *query, float *out, int batch, int num_circuits, int dim) {
    q.parallel_for(sycl::range<3>(batch, num_circuits, dim), [=](sycl::id<3> idx) {
        int b = idx[0];
        int n = idx[1];
        int d_out = idx[2]; // Output dim index
        
        // M_b is [Dim, Dim] at M[b * dim * dim]
        // q_bn is [Dim] at query[b * num_circuits * dim + n * dim]
        
        int m_base = b * dim * dim;
        int q_base = b * num_circuits * dim + n * dim;
        int out_idx = b * num_circuits * dim + n * dim + d_out;
        
        float sum = 0.0f;
        for(int d_in = 0; d_in < dim; ++d_in) {
            // y[d_out] = sum(M[d_out, d_in] * q[d_in])
            sum += M[m_base + d_out * dim + d_in] * query[q_base + d_in];
        }
        out[out_idx] = sum;
    });
}

// v1.6: CorticalHub 記憶體讀取反向傳播
// dL/dq = M^T * dL/dy
// dL/dM = dL/dy * q^T
// 我們只實作 dL/dq，因為 dL/dM 需要複雜的 BPTT 或者在 fast_weight_update 中處理。
// 這裡僅讓梯度流回 query。
void memory_read_bwd_query(sycl::queue &q, const float *M, const float *grad_y, float *grad_q, int batch, int num_circuits, int dim) {
    // Initialize grad_q to 0? Assumed handled by caller or we set it.
    // Since grad_q is from query projection, it might be accumulative.
    // But this is the ONLY place q is used in this op.
    // Let's assume accumulative += for safety.
    
    q.parallel_for(sycl::range<3>(batch, num_circuits, dim), [=](sycl::id<3> idx) {
        int b = idx[0];
        int n = idx[1];
        int d_in = idx[2]; // Input dim index (for q)
        
        int m_base = b * dim * dim;
        int gy_base = b * num_circuits * dim + n * dim;
        int gq_idx = b * num_circuits * dim + n * dim + d_in;
        
        float sum = 0.0f;
        for(int d_out = 0; d_out < dim; ++d_out) {
            // dq[d_in] = sum(M[d_out, d_in] * dy[d_out])
            // M[d_out, d_in] is the transpose access
            sum += M[m_base + d_out * dim + d_in] * grad_y[gy_base + d_out];
        }
        
        // Atomic add not needed as threads map 1-to-1 to q elements
        grad_q[gq_idx] += sum;
    });
}

// v1.7: 記憶體讀取反向傳播 - 計算 dL/dM
// dL/dM += sum_over_circuits(dL/dy_n * q_n^T)
// M: [Batch, Dim, Dim]
// grad_y: [Batch, N, Dim]
// q: [Batch, N, Dim]
// grad_M: [Batch, Dim, Dim] (Accumulate)
void memory_read_bwd_M(sycl::queue &q, const float *grad_y, const float *query, float *grad_M, int batch, int num_circuits, int dim) {
    q.parallel_for(sycl::range<3>(batch, dim, dim), [=](sycl::id<3> idx) {
        int b = idx[0];
        int r = idx[1]; // M row (output dim of read)
        int c = idx[2]; // M col (input dim of read / query dim)
        
        float sum = 0.0f;
        for(int n=0; n<num_circuits; ++n) {
            // dy[b, n, r] * q[b, n, c]
            int base_idx = b * num_circuits * dim + n * dim;
            sum += grad_y[base_idx + r] * query[base_idx + c];
        }
        
        // M[b, r, c] gradient accumulation
        grad_M[b * dim * dim + r * dim + c] += sum;
    });
}

// v1.8: 快速權重更新反向傳播
// Forward: M_t = clamp(lambda * M_{t-1} + eta * sum(g * k * v^T))
// Backward inputs: 
//   d_M_t (gradient on the updated memory M_t)
//   M_old (M_{t-1}, to check clamp boundaries if strictly needed, but simplified: ignore clamp deriv for speed/stability or assume linear region)
//   k, v, g (values used in update)
// Backward outputs:
//   d_M_old = d_M_t * lambda (if not clamped)
//   d_k += d_M_t * eta * g * v
//   d_v += d_M_t * eta * g * k
//   d_g += d_M_t * eta * k * v
void fast_weight_update_bwd(sycl::queue &q, 
                            const float *d_M_t, // [B, D, D]
                            const float *k, const float *v, const float *g_sig, // [B, N, D], [B, N, D], [B, N, 1]
                            float *d_M_old,     // [B, D, D]
                            float *d_k, float *d_v, float *d_g_sig, // Accumulators
                            int batch, int num_circuits, int dim, 
                            float lambda, float eta) 
{
    // 1. Compute d_M_old (Decay)
    q.parallel_for(sycl::range<1>(batch * dim * dim), [=](sycl::id<1> idx) {
        // d_M_{t-1} += d_M_t * lambda
        // Note: We assume d_M_old is initialized (usually d_M from previous backward step)
        // Wait, d_M_old IS d_M for the next backward step.
        // In `CorticalHub::backward`, we have `d_M_accum`. 
        // We update it in place: d_M_accum *= lambda (plus new contributions).
        // But here we strictly separate inputs/outputs for clarity.
        // Let's assume d_M_old is an output buffer initialized to 0 or accumulating.
        d_M_old[idx] += d_M_t[idx] * lambda; 
    });

    // 2. Compute d_k, d_v, d_g (This is tricky to parallelize efficiently)
    // d_L / d_k_{b,n,r} = sum_c ( d_L/d_M_{b,r,c} * d_M/d_k )
    // term = eta * g_{b,n} * v_{b,n,c}
    // so d_k_{b,n,r} = sum_c ( d_M_{b,r,c} * eta * g * v_c )
    //                = eta * g * sum_c ( d_M_{r,c} * v_c )
    // This looks like Matrix-Vector mult: d_M * v
    
    // Launch parallel over [Batch, NumCircuits, Dim]
    q.parallel_for(sycl::range<3>(batch, num_circuits, dim), [=](sycl::id<3> idx) {
        int b = idx[0];
        int n = idx[1];
        int d = idx[2]; // Dimension index (r for k, c for v)
        
        // Base pointers
        int m_base = b * dim * dim;
        int vec_base = b * num_circuits * dim + n * dim;
        int g_idx = b * num_circuits + n;
        
        float gate = g_sig[g_idx];
        float common_scale = eta * gate; // eta * g
        
        // Compute d_k[d] = scale * sum_c ( d_M[d, c] * v[c] )
        float sum_k = 0.0f;
        for(int c=0; c<dim; ++c) {
            sum_k += d_M_t[m_base + d * dim + c] * v[vec_base + c];
        }
        
        // Compute d_v[d] = scale * sum_r ( d_M[r, d] * k[r] )
        float sum_v = 0.0f;
        for(int r=0; r<dim; ++r) {
            sum_v += d_M_t[m_base + r * dim + d] * k[vec_base + r];
        }
        
        // Accumulate to gradients
        // d_k and d_v are gradients of BitLinear outputs
        d_k[vec_base + d] += common_scale * sum_k;
        d_v[vec_base + d] += common_scale * sum_v;
    });
    
    // 3. Compute d_g (Scalar per circuit)
    q.parallel_for(sycl::range<2>(batch, num_circuits), [=](sycl::id<2> idx) {
        int b = idx[0];
        int n = idx[1];
        
        int m_base = b * dim * dim;
        int vec_base = b * num_circuits * dim + n * dim;
        
        // d_g = eta * sum_{r,c} ( d_M[r,c] * k[r] * v[c] )
        float sum = 0.0f;
        for(int r=0; r<dim; ++r) {
            float k_val = k[vec_base + r];
            for(int c=0; c<dim; ++c) {
                float v_val = v[vec_base + c];
                sum += d_M_t[m_base + r * dim + c] * k_val * v_val;
            }
        }
        
        d_g_sig[b * num_circuits + n] += eta * sum;
    });
}

// v1.9: Sigmoid 反向傳播
// dx = dy * y * (1 - y)
// y is sigmoid(x) result
void sigmoid_bwd(sycl::queue &q, const float *grad_y, const float *y, float *grad_x, int size) {
    q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
        float s = y[idx];
        grad_x[idx] = grad_y[idx] * s * (1.0f - s);
    });
}

// v2.0: AdamW 優化器步驟
// m = beta1 * m + (1 - beta1) * grad
// v = beta2 * v + (1 - beta2) * grad * grad
// w = w - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * w)
// where m_hat = m / (1 - beta1^t), v_hat = v / (1 - beta2^t)
void adamw_step(sycl::queue &q, float *weights, float *grads, float *m, float *v, 
                int size, float lr, float beta1, float beta2, float eps, float weight_decay, int step) {
    
    // 計算偏差修正係數
    // 注意: pow 是 host 端計算傳入 kernel，還是 device 端?
    // 為了效率，在 host 算好傳入，或者讓每個 thread 算 (浪費)。
    // 這裡簡化：假設 step 在 host 端處理好修正係數，或者我們在這裡算。
    // 為了 API 簡單，我們在 Kernel 裡算 (雖然有點慢)。
    // 其實 beta1_pow = pow(beta1, step) 可以當參數傳入。
    // 讓我們改為傳入 `bias_correction1` 和 `bias_correction2`。
    // 但為了保持簽名簡單，我們先用 step。
    
    q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
        int i = idx[0];
        float g = grads[i];
        
        // Update moments
        float m_curr = m[i];
        float v_curr = v[i];
        
        m_curr = beta1 * m_curr + (1.0f - beta1) * g;
        v_curr = beta2 * v_curr + (1.0f - beta2) * g * g;
        
        // Write back moments
        m[i] = m_curr;
        v[i] = v_curr;
        
        // Bias correction
        float beta1_pow = sycl::pown(beta1, step);
        float beta2_pow = sycl::pown(beta2, step);
        
        float m_hat = m_curr / (1.0f - beta1_pow);
        float v_hat = v_curr / (1.0f - beta2_pow);
        
        // Update weights
        float w = weights[i];
        w = w - lr * (m_hat / (sycl::sqrt(v_hat) + eps) + weight_decay * w);
        weights[i] = w;
        
        // Clear gradient
        grads[i] = 0.0f;
    });
}

} // namespace kernels

