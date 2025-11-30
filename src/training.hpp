// src/training.hpp
#pragma once
#include "common.hpp"
#include "kernels.hpp"
#include <unordered_map>

struct AdamState {
    float *m = nullptr;
    float *v = nullptr;
    size_t size = 0;
    
    // Step counter for bias correction
    int step = 0; 
};

class TrainingTools
{
    sycl::queue &q;
    float *scratch_pad = nullptr;
    float *loss_device = nullptr;
    
    // Map data pointer (key) to Optimizer State
    // Since pointers from MemoryPool might be reused, this is dangerous if tensor dies.
    // But weights are long-lived.
    // We assume sgd_update is called only on persistent weight tensors.
    std::unordered_map<float*, AdamState> adam_states;

  public:
    TrainingTools(sycl::queue &queue) : q(queue)
    {
        // 支援最大 Batch Size 1024
        scratch_pad = sycl::malloc_shared<float>(1024 * 2, q); 
        loss_device = sycl::malloc_shared<float>(1024, q);
    }

    ~TrainingTools()
    {
        if (scratch_pad)
            sycl::free(scratch_pad, q);
        if (loss_device)
            sycl::free(loss_device, q);
            
        // Free optimizer states
        for(auto& kv : adam_states) {
            sycl::free(kv.second.m, q);
            sycl::free(kv.second.v, q);
        }
    }

    // 交叉熵損失與梯度 (批次版)
    // target_indices: [Batch]
    // 如果同步為真則回傳平均損失，否則回傳 -1.0f
    float cross_entropy_loss(Tensor &logits, const std::vector<int>& target_indices, Tensor &d_logits, int step, int log_interval = 100)
    {
        int batch = target_indices.size();
        int vocab_size = logits.size / batch;
        
        // 複製 targets 到 device (使用 scratch_pad 的後半部分作為臨時 int 緩衝區? 不安全)
        // 最好分配一個專用的 int buffer 或是使用 malloc_shared
        // 為了效能，我們使用 malloc_shared 並且不立即釋放 (依賴 MemoryPool 會更好，但這裡手動管理)
        int* d_targets = sycl::malloc_shared<int>(batch, q);
        q.memcpy(d_targets, target_indices.data(), batch * sizeof(int));
        
        // 使用 GPU 核心
        kernels::cross_entropy_fw_bw(q, logits.data, d_targets, d_logits.data, loss_device, batch, vocab_size, scratch_pad);
        
        // 釋放 targets (非同步)
        q.submit([=](sycl::handler &h) {
            h.host_task([=]() {
                sycl::free(d_targets, q);
            });
        });

        // 僅定期等待並回傳損失
        if ((step + 1) % log_interval == 0) {
            q.wait();
            // 計算平均損失 (在 Host 做，因為 batch 小)
            float total_loss = 0.0f;
            for(int i=0; i<batch; ++i) total_loss += loss_device[i];
            return total_loss / batch;
        }
        
        return -1.0f; 
    }

    // AdamW 優化器
    void sgd_update(Tensor &weights, float lr, float weight_decay = 0.01f, float grad_scale = 1.0f)
    {
        // 確保狀態存在
        if (adam_states.find(weights.data) == adam_states.end()) {
            AdamState state;
            state.size = weights.size;
            state.m = sycl::malloc_shared<float>(weights.size, q);
            state.v = sycl::malloc_shared<float>(weights.size, q);
            q.fill(state.m, 0.0f, weights.size);
            q.fill(state.v, 0.0f, weights.size);
            state.step = 0;
            adam_states[weights.data] = state;
        }
        
        AdamState& state = adam_states[weights.data];
        state.step++;
        
        // 預設參數
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-8f;
        
        // 呼叫 AdamW Kernel
        // 注意：我們將 grad_scale 應用於 lr 或 grad？
        // 為了符合習慣，grad 應該先縮放。
        // 為了避免額外的 kernel，我們修改 adamw_step kernel 接受 grad_scale？
        // 或是我們在外部縮放。
        // 簡化：將 grad_scale 乘入 lr。這不完全正確 (m/v 統計會受影響)。
        // 正確做法：修改 adamw_step 支援 grad scale。
        // 為了不修改 kernel，我們這裡先假設 grad_scale=1.0 或在之前已經處理。
        // 其實 MiniLLM 傳入的 scale 是 1/accum_steps。這應該直接作用於梯度。
        // 但我們的 kernel 假設 grad 是原始的。
        // **修正**: 我們修改 LR。對於 SGD: w -= lr * (grad * scale)。
        // 對於 Adam，m = b1*m + (1-b1)*g。如果 g 縮放了，m 也縮放了。
        // 最終 update ~ m_hat ~ g * scale。
        // 所以調整 lr 是不夠的，因為 epsilon 項的存在。
        // 但是如果 scale 是常數，它會被 sqrt(v) 抵消一部分。
        // 最佳方案：在計算梯度時已經縮放，或者在 adam 內部縮放。
        // 讓我們簡單地將梯度預乘 scale (需要一個 kernel)。
        // 或者... 這裡用一個簡單的 kernel 先縮放梯度？
        // 為了效能，直接把 scale 傳給 adamw_step 比較好。
        // 但 adamw_step 簽名固定。
        // 讓我們 Hack 一下：直接修改梯度 tensor (in-place scale)。
        if (std::abs(grad_scale - 1.0f) > 1e-6f) {
            q.parallel_for(sycl::range<1>(weights.size), [=, g=weights.grad](sycl::id<1> idx) {
                g[idx] *= grad_scale;
            });
        }

        kernels::adamw_step(q, weights.data, weights.grad, state.m, state.v, 
                            weights.size, lr, beta1, beta2, eps, weight_decay, state.step);
    }
};
