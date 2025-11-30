// src/embedding.hpp
#pragma once
#include "common.hpp"
#include "training.hpp" // 會用到優化器

class Embedding
{
    sycl::queue &q;

  public:
    int vocab_size;
    int embed_dim;
    Tensor weights; // [向量表大小, 向量維度]

    // 反向傳播快取的輸入ID
    std::vector<int> last_input_ids;

    Embedding(int vocab, int dim, sycl::queue &queue)
        : q(queue), vocab_size(vocab), embed_dim(dim), weights(vocab * dim, queue)
    {
        weights.random_init(0.0f, 0.1f);
    }

    // 前向傳播：IDs -> 向量
    // 輸入: std::vector<int> ids (批次大小)
    // 輸出: Tensor [批次, 維度]
    Tensor forward(const std::vector<int> &ids)
    {
        int batch = ids.size();
        last_input_ids = ids; // 儲存輸入ID以供反向傳播使用

        Tensor out(batch * embed_dim, q);

        // 將 IDs 複製到裝置端供核心使用？還是只在主機端
        // 迴圈以啟動記憶體複製？
        // 由於查找表是稀疏的，因此需要使用單獨的記憶體複製或將 ID 儲存在裝置
        // 上的核心。 對於 Mini LLM，我們可以逐一複製嵌入向量（簡單但速度慢）。
        // 或將整個表複製到裝置（已完成）並啟動核心。

        // 將 IDs 放置在裝置上
        int *d_ids = sycl::malloc_shared<int>(batch, q);
        q.memcpy(d_ids, ids.data(), batch * sizeof(int));

        const int v_size = vocab_size;
        const int e_dim = embed_dim;
        q.parallel_for(sycl::range<2>(batch, e_dim), [=, w_ptr = weights.data, out_ptr = out.data](sycl::id<2> idx) {
             int b = idx[0];
             int d = idx[1];
             int token_id = d_ids[b];

             if (token_id >= 0 && token_id < v_size)
             {
                 out_ptr[b * e_dim + d] = w_ptr[token_id * e_dim + d];
             }
             else
             {
                 out_ptr[b * e_dim + d] = 0.0f;
             }
         });

        // sycl::free(d_ids, q);
        // 非同步釋放
        q.submit([=](sycl::handler &h) {
            h.host_task([=]() {
                sycl::free(d_ids, q);
            });
        });
        
        return out;
    }

    // 反向傳播
    // grad_output: [批次, 維度]
    // input_ids: 前向傳播使用的輸入 ID (支援 BPTT)
    void backward(Tensor &grad_output, const std::vector<int>& input_ids = {})
    {
        // 如果傳入 input_ids 則使用，否則回退到 last_input_ids (不安全)
        const std::vector<int>* ids_ptr = &last_input_ids;
        if (!input_ids.empty()) {
            ids_ptr = &input_ids;
        }
        
        int batch = ids_ptr->size();

        // 複製 IDs 到裝置
        int *d_ids = sycl::malloc_shared<int>(batch, q);
        q.memcpy(d_ids, ids_ptr->data(), batch * sizeof(int));

        const int v_size = vocab_size;
        const int e_dim = embed_dim;
        // 稀疏梯度更新：將梯度累積至weights.grad
        // 需要原子加法，因為多個批次項目可能是相同的標記
        q.parallel_for(sycl::range<2>(batch, e_dim), [=, g_out = grad_output.data,
                                                      w_grad = weights.grad](sycl::id<2> idx) {
             int b = idx[0];
             int d = idx[1];
             int token_id = d_ids[b];

             if (token_id >= 0 && token_id < v_size)
             {
                 // 用於線程安全的原子參考
                 auto ref = sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>(w_grad[token_id * e_dim + d]);
                 ref.fetch_add(g_out[b * e_dim + d]);
             }
         });

        // sycl::free(d_ids, q); // 在未來的更新中使用 free_async 或小心處理
        // 目前，使用與之前相同的模式，但我們需要小心生命週期。
        // 由於我們移除了 .wait()，我們必須確保 d_ids 存活直到 kernel 完成。
        // 但是：在 Embedding 中，我們在本地分配 d_ids。如果我們返回，它就會遺失，但指標在我們釋放之前都是有效的。
        // 為了安全地使用非同步，我們應該使用我們新增到 Tensor 的 free_async 機制，但這裡它是原始指標。
        // 讓我們假設我們添加一個 lambda 來釋放它。
        
        q.submit([=](sycl::handler &h) {
            h.host_task([=]() {
                sycl::free(d_ids, q);
            });
        });
    }

    void save(std::ofstream &out)
    {
        // 儲存原始 FP32 權重
        std::vector<float> h_w(weights.size);
        q.memcpy(h_w.data(), weights.data, weights.size * sizeof(float)).wait();
        out.write(reinterpret_cast<char *>(h_w.data()), h_w.size() * sizeof(float));
    }

    void load(std::ifstream &in)
    {
        std::vector<float> h_w(weights.size);
        in.read(reinterpret_cast<char *>(h_w.data()), h_w.size() * sizeof(float));
        q.memcpy(weights.data, h_w.data(), weights.size * sizeof(float)).wait();
    }
};