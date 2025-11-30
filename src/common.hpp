// src/common.hpp
#pragma once
#include <algorithm> // for std::max
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>
#include <vector>
#include <map>
#include <stack>
#include <mutex>
#include <memory>

// 強健的記憶體池 (Robust Memory Pool)
class MemoryPool
{
    sycl::queue &q;
    // 大小 -> 空閒指標堆疊
    std::map<size_t, std::stack<float*>> pool;
    std::mutex mtx; 

  public:
    MemoryPool(sycl::queue &queue) : q(queue) {}

    ~MemoryPool()
    {
        std::lock_guard<std::mutex> lock(mtx);
        for (auto &entry : pool)
        {
            while (!entry.second.empty())
            {
                sycl::free(entry.second.top(), q);
                entry.second.pop();
            }
        }
    }

    float *allocate(size_t size)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (pool.count(size) && !pool[size].empty())
        {
            float *ptr = pool[size].top();
            pool[size].pop();
            return ptr;
        }
        return sycl::malloc_shared<float>(size, q);
    }

    void deallocate(float *ptr, size_t size)
    {
        if (!ptr) return;
        std::lock_guard<std::mutex> lock(mtx);
        pool[size].push(ptr);
        // 注意: 不立即釋放，供重用
    }
};

// 為了在不修改所有 Tensor 建構子的情況下使用 Pool，我們使用全域單例模式的變體。
// 但每個 Tensor 綁定一個 Queue。如果有多個 Queue，全域單例會出問題。
// 解決方案：每個 Tensor 持有一個 shared_ptr<MemoryPool>。
// 或者：我們假設此應用程式只有一個主要 Context/Queue。
// 為了最佳實作，我們在 Tensor 中添加 static map<context, pool>？太複雜。
// 簡單方案：在創建 MiniLLM 時傳入 Pool，然後 Tensor 接受 Pool 指標。
// 但這需要修改所有簽名。
// **折衷最佳方案**: 
// Tensor 預設使用 malloc/free。
// 提供一個 `Tensor::set_memory_pool(MemoryPool* p)` 靜態方法？
// 不，這不線程安全。
// 讓我們修改 Tensor 建構子來接受 `MemoryPool*` (可選)。
// 為了方便，我們提供一個全域存取點。

struct Tensor;

// 全域 Pool 管理器 (簡單版)
class GlobalPoolManager {
public:
    static MemoryPool* get_pool(sycl::queue& q) {
        // 由於 sycl::context 沒有直接的指標存取或比較運算子，
        // 且通常只有一個 context，我們使用簡單的靜態變數。
        // 如果需要支援多 context，可以使用 vector 線性搜尋。
        
        static std::vector<std::pair<sycl::context, std::unique_ptr<MemoryPool>>> pools;
        static std::mutex m;
        std::lock_guard<std::mutex> lock(m);
        
        sycl::context ctx = q.get_context();
        
        for(auto& p : pools) {
            if(p.first == ctx) {
                return p.second.get();
            }
        }
        
        pools.push_back({ctx, std::make_unique<MemoryPool>(q)});
        return pools.back().second.get();
    }
};

struct Tensor
{
    float *data = nullptr;
    float *grad = nullptr;
    size_t size = 0;
    sycl::queue q; 
    
    // 優化器狀態 (Intrusive, 為了方便 TrainingTools 管理)
    // void* opt_state = nullptr; // 棄用，改由 TrainingTools 管理 map

    Tensor(size_t s, sycl::queue queue) : size(s), q(queue)
    {
        MemoryPool* pool = GlobalPoolManager::get_pool(q);
        data = pool->allocate(size);
        grad = pool->allocate(size);
        q.fill(data, 0.0f, size);
        q.fill(grad, 0.0f, size);
    }

    void free_async(float *ptr, size_t s)
    {
        if (!ptr) return;
        // 我們需要確保 kernel 完成後才歸還給 pool。
        // Pool 的 deallocate 是同步的 (push to stack)。
        // 如果立即 push，下一個 allocate 拿到它並立即寫入，可能會 race condition (如果前一個 kernel 還在讀)。
        // SYCL in-order queue 解決了這個問題！
        // 只要我們在同一個 queue 上操作，依賴關係會保證順序。
        // 所以我們可以立即歸還。
        
        MemoryPool* pool = GlobalPoolManager::get_pool(q);
        pool->deallocate(ptr, s);
    }

    ~Tensor()
    {
        free_async(data, size);
        free_async(grad, size);
    }

    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;

    Tensor(Tensor &&other) noexcept : q(other.q), size(other.size), data(other.data), grad(other.grad)
    {
        other.data = nullptr;
        other.grad = nullptr;
        other.size = 0;
    }

    Tensor &operator=(Tensor &&other) noexcept
    {
        if (this != &other)
        {
            free_async(data, size);
            free_async(grad, size);

            q = other.q;
            size = other.size;
            data = other.data;
            grad = other.grad;

            other.data = nullptr;
            other.grad = nullptr;
            other.size = 0;
        }
        return *this;
    }

    void random_init(float mean = 0.0f, float std = 1.0f)
    {
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(mean, std);
        for (size_t i = 0; i < size; ++i)
            data[i] = dist(gen);
    }

    void zero_grad()
    {
        q.fill(grad, 0.0f, size);
    }
};