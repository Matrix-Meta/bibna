// src/common.hpp
#pragma once
#include <algorithm>  // for std::max
#include <cmath>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>
#include <vector>

struct Tensor {
  float* data = nullptr;
  float* grad = nullptr;
  size_t size = 0;
  sycl::queue q;  // [Fix 1] 改為數值型別 (Value Type)，不再是參考

  // 建構子
  Tensor(size_t s, sycl::queue queue) : size(s), q(queue) {
    data = sycl::malloc_shared<float>(size, q);
    grad = sycl::malloc_shared<float>(size, q);
    q.fill(data, 0.0f, size).wait();
    q.fill(grad, 0.0f, size).wait();
  }

  // 解構子
  ~Tensor() {
    if (data) sycl::free(data, q);
    if (grad) sycl::free(grad, q);
  }

  // [Fix 2] 禁止複製 (防止 Double Free)
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;

  // [Fix 3] 實作移動建構子 (Move Constructor)
  Tensor(Tensor&& other) noexcept
      : q(other.q), size(other.size), data(other.data), grad(other.grad) {
    // 竊取資源後，將來源歸零
    other.data = nullptr;
    other.grad = nullptr;
    other.size = 0;
  }

  // [Fix 4] 實作移動賦值 (Move Assignment)
  // 這是解決 s_state = result.second 報錯的關鍵
  Tensor& operator=(Tensor&& other) noexcept {
    if (this != &other) {
      // 1. 釋放自己當前的記憶體
      if (data) sycl::free(data, q);
      if (grad) sycl::free(grad, q);

      // 2. 竊取對方的資源
      q = other.q;
      size = other.size;
      data = other.data;
      grad = other.grad;

      // 3. 將對方歸零 (防止對方解構時釋放記憶體)
      other.data = nullptr;
      other.grad = nullptr;
      other.size = 0;
    }
    return *this;
  }

  // 輔助函式
  void random_init(float mean = 0.0f, float std = 1.0f) {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(mean, std);
    for (size_t i = 0; i < size; ++i) data[i] = dist(gen);
  }

  void zero_grad() { q.fill(grad, 0.0f, size).wait(); }
};