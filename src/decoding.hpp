#pragma once
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "common.hpp"

class AntiCollapse
{
    sycl::queue &q;
    float penalty_factor;
    int vocab_size;

  public:
    AntiCollapse(sycl::queue &queue, float penalty, int size) : q(queue), penalty_factor(penalty), vocab_size(size)
    {
    }

    void apply_penalty(Tensor &logits, const std::vector<int> &history)
    {
        if (history.empty())
            return;

        // 複製到主機進行簡單處理
        std::vector<float> h_logits(logits.size);
        q.memcpy(h_logits.data(), logits.data, logits.size * sizeof(float)).wait();

        for (int token_id : history)
        {
            if (token_id >= 0 && token_id < vocab_size && token_id < logits.size)
            {
                // 應用懲罰
                if (h_logits[token_id] < 0)
                {
                    h_logits[token_id] *= penalty_factor;
                }
                else
                {
                    h_logits[token_id] /= penalty_factor;
                }
            }
        }

        // 複製回裝置
        q.memcpy(logits.data, h_logits.data(), logits.size * sizeof(float)).wait();
    }

    int sample(Tensor &logits)
    {
        // 複製到主機
        std::vector<float> h_logits(logits.size);
        q.memcpy(h_logits.data(), logits.data, logits.size * sizeof(float)).wait();

        // Softmax
        float max_val = -1e9;
        for (float v : h_logits)
            max_val = std::max(max_val, v);

        std::vector<float> probs(logits.size);
        float sum_exp = 0.0f;
        for (size_t i = 0; i < h_logits.size(); ++i)
        {
            probs[i] = std::exp(h_logits[i] - max_val);
            sum_exp += probs[i];
        }

        // 採樣
        static std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<float> dist(0.0f, sum_exp);
        float r = dist(gen);

        float cum = 0.0f;
        for (size_t i = 0; i < probs.size(); ++i)
        {
            cum += probs[i];
            if (r <= cum)
                return i;
        }
        return (int)probs.size() - 1;
    }
};
