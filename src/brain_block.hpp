#pragma once
#include <utility>
#include <vector>

#include "cortical_hub.hpp"
#include "micro_circuit.hpp"

class BrainBlock {
  sycl::queue& q;
  int num_circuits;
  int dim;
  std::vector<MicroCircuit> circuits;
  CorticalHub hub;

 public:
  BrainBlock(int n_circuits, int d, sycl::queue& queue)
      : q(queue),
        num_circuits(n_circuits),
        dim(d),
        hub(d, d, n_circuits, queue) {
    // Create MicroCircuits
    // Assuming hidden_dim = dim for simplicity
    for (int i = 0; i < num_circuits; ++i) {
      circuits.emplace_back(d, d, queue);
    }
  }

  std::vector<Tensor> init_states(int batch) {
    hub.init_memory(batch);

    std::vector<Tensor> states;
    states.reserve(num_circuits);
    for (int i = 0; i < num_circuits; ++i) {
      // State for each MicroCircuit: [Batch, Hidden]
      // Tensor constructor allocates memory but doesn't initialize value if not
      // specified? MicroCircuit uses 0-init for recurrent state usually.
      Tensor s(batch * dim, q);
      q.fill(s.data, 0.0f, s.size).wait();
      states.push_back(std::move(s));
    }
    return states;
  }

  // Returns {logits, next_states}
  std::pair<Tensor, std::vector<Tensor>> forward(Tensor& input,
                                                 std::vector<Tensor>& states) {
    int batch = input.size / dim;
    std::vector<Tensor> next_states;
    std::vector<Tensor> mc_outputs;

    next_states.reserve(num_circuits);
    mc_outputs.reserve(num_circuits);

    // 1. MicroCircuits Forward
    for (int i = 0; i < num_circuits; ++i) {
      // forward_step returns {output, new_state}
      auto result = circuits[i].forward_step(input, states[i]);
      mc_outputs.push_back(std::move(result.first));
      next_states.push_back(std::move(result.second));
    }

    // 2. Prepare input for Hub
    // Hub expects [Batch, NumCircuits, Dim] (conceptually) flattened.
    // We need to aggregate mc_outputs.
    // For simplicity and to support batch=1 correctly (as in generation_test),
    // we concatenate them.
    // Note: If batch > 1, this simple concat produces [NumCircuits, Batch,
    // Dim]. Ideally needs transpose to [Batch, NumCircuits, Dim].
    Tensor hub_input(batch * num_circuits * dim, q);
    size_t chunk_size = batch * dim;
    for (int i = 0; i < num_circuits; ++i) {
      q.memcpy(hub_input.data + i * chunk_size, mc_outputs[i].data,
               chunk_size * sizeof(float));
    }
    q.wait();  // Ensure copy finished

    // 3. Hub Memory Update
    hub.update_memory(hub_input);

    // 4. Aggregate Outputs
    // Simple mean pooling of MicroCircuit outputs
    Tensor output(batch * dim, q);
    q.fill(output.data, 0.0f, output.size).wait();

    for (int i = 0; i < num_circuits; ++i) {
      float* out_ptr = output.data;
      float* mc_ptr = mc_outputs[i].data;
      q.parallel_for(sycl::range<1>(output.size), [=](sycl::id<1> idx) {
         out_ptr[idx] += mc_ptr[idx];
       }).wait();
    }

    // Normalize
    float scale = 1.0f / num_circuits;
    float* out_ptr = output.data;
    q.parallel_for(sycl::range<1>(output.size), [=](sycl::id<1> idx) {
       out_ptr[idx] *= scale;
     }).wait();

    return {std::move(output), std::move(next_states)};
  }
};