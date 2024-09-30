#pragma once
// #include "utils/log.h"
#include <tcnn.h>
#include <common_device.h>
#include <gpu_matrix.h>

#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

using namespace tcnn;

template <uint32_t N_DIMS>
__device__ uint32_t fast_hash(const uvec<N_DIMS>& pos_grid)
{
	static_assert(N_DIMS == 3, "fast_hash can only hash 3 dimensions.");
	return get_index(pos_grid[0], pos_grid[1], pos_grid[2]);
}

template <uint32_t N_DIMS>
__device__ uint32_t grid_index(const uint32_t hashmap_size, const uint32_t grid_resolution, const uvec<N_DIMS>& pos_grid)
{
	uint32_t stride = 1;
	uint32_t index = 0;

	// The second part of the loop condition is needed to avoid integer overflows in finer levels.
	TCNN_PRAGMA_UNROLL
	for (uint32_t dim = 0; dim < N_DIMS && stride <= hashmap_size; ++dim) {
		index += pos_grid[dim] * stride;
		stride *= grid_resolution;
	}

	if (hashmap_size < stride) {
		index = fast_hash<N_DIMS>(pos_grid);
	}

	return index % hashmap_size;
}

template <typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL>
__global__ void kernel_grid(
	const uint32_t num_elements,
	const uint32_t num_grid_features,
	const uint32_t* __restrict__ offset_table,
	const uint32_t base_resolution,
	const float log2_per_level_scale,
	float max_level,
	const T* __restrict__ grid,
	MatrixView<const float> positions_in,
	T* __restrict__ encoded_positions,
	float *__restrict__ dy_dx)
{
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_elements) return;

	const uint32_t level = blockIdx.y; // <- the level is the same for all threads

	max_level = (max_level * num_grid_features) / N_FEATURES_PER_LEVEL;

	if (level >= max_level + 1e-3f) {
		if (encoded_positions) {
			TCNN_PRAGMA_UNROLL
			for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
				encoded_positions[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = (T)0.0f;
			}
		}

		// Gradient is zero for zeroed-out dimensions.
		if (dy_dx) {
			TCNN_PRAGMA_UNROLL
			for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
				((vec<N_POS_DIMS>*)dy_dx)[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = {0.0f};
			}
		}

		return;
	}

	grid += offset_table[level] * N_FEATURES_PER_LEVEL;
	const uint32_t hashmap_size = offset_table[level + 1] - offset_table[level];

	const float scale = grid_scale(level, log2_per_level_scale, base_resolution);
	const uint32_t resolution = grid_resolution(scale);

	float pos[N_POS_DIMS];
	float pos_derivative[N_POS_DIMS];
	uvec<N_POS_DIMS> pos_grid;

	TCNN_PRAGMA_UNROLL
	for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
		pos_fract(positions_in(dim, i), &pos[dim], &pos_derivative[dim], &pos_grid[dim], scale, identity_fun, identity_derivative);
	}

	auto grid_val = [&](const uvec<N_POS_DIMS>& local_pos) {
		const uint32_t index = grid_index<N_POS_DIMS>(hashmap_size, resolution, local_pos) * N_FEATURES_PER_LEVEL;
		return *(tvec<T, N_FEATURES_PER_LEVEL, PARAMS_ALIGNED ? sizeof(T) * N_FEATURES_PER_LEVEL : sizeof(T)>*)&grid[index];
	};

	if (encoded_positions) {
		// N-linear interpolation
		tvec<T, N_FEATURES_PER_LEVEL, PARAMS_ALIGNED ? sizeof(T) * N_FEATURES_PER_LEVEL : sizeof(T)> result = {};

		TCNN_PRAGMA_UNROLL
		for (uint32_t idx = 0; idx < (1 << N_POS_DIMS); ++idx) {
			float weight = 1;
			uvec<N_POS_DIMS> pos_grid_local;

			TCNN_PRAGMA_UNROLL
			for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
				if ((idx & (1<<dim)) == 0) {
					weight *= 1 - pos[dim];
					pos_grid_local[dim] = pos_grid[dim];
				} else {
					weight *= pos[dim];
					pos_grid_local[dim] = pos_grid[dim] + 1;
				}
			}

			result = fma((T)weight, grid_val(pos_grid_local), result);
		}

		TCNN_PRAGMA_UNROLL
		for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
			encoded_positions[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = result[f];
		}
	}

	// Gradient
	if (dy_dx) {
		vec<N_POS_DIMS> grads[N_FEATURES_PER_LEVEL] = {0.0f};

		TCNN_PRAGMA_UNROLL
		for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim) {
			TCNN_PRAGMA_UNROLL
			for (uint32_t idx = 0; idx < (1 << (N_POS_DIMS-1)); ++idx) {
				float weight = scale;
				uvec<N_POS_DIMS> pos_grid_local;

				TCNN_PRAGMA_UNROLL
				for (uint32_t non_grad_dim = 0; non_grad_dim < N_POS_DIMS-1; ++non_grad_dim) {
					const uint32_t dim = non_grad_dim >= grad_dim ? (non_grad_dim+1) : non_grad_dim;

					if ((idx & (1<<non_grad_dim)) == 0) {
						weight *= 1 - pos[dim];
						pos_grid_local[dim] = pos_grid[dim];
					} else {
						weight *= pos[dim];
						pos_grid_local[dim] = pos_grid[dim] + 1;
					}
				}

				pos_grid_local[grad_dim] = pos_grid[grad_dim];
				auto val_left = grid_val(pos_grid_local);
				pos_grid_local[grad_dim] = pos_grid[grad_dim] + 1;
				auto val_right = grid_val(pos_grid_local);

				TCNN_PRAGMA_UNROLL
				for (uint32_t feature = 0; feature < N_FEATURES_PER_LEVEL; ++feature) {
					grads[feature][grad_dim] += weight * ((float)val_right[feature] - (float)val_left[feature]) * pos_derivative[grad_dim];
				}
			}
		}

		TCNN_PRAGMA_UNROLL
		for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
			((vec<N_POS_DIMS>*)dy_dx)[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = grads[f];
		}
	}
}

template <typename T, typename GRAD_T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, uint32_t N_FEATURES_PER_THREAD>
__global__ void kernel_grid_backward(
	const uint32_t num_elements,
	const uint32_t num_grid_features,
	const uint32_t* __restrict__ offset_table,
	const uint32_t base_resolution,
	const float log2_per_level_scale,
	float max_level,
	GRAD_T* __restrict__ grid_gradient,
	MatrixView<const float> positions_in,
	const T* __restrict__ dL_dy)
{
	const uint32_t i = ((blockIdx.x * blockDim.x + threadIdx.x) * N_FEATURES_PER_THREAD) / N_FEATURES_PER_LEVEL;
	if (i >= num_elements) return;

	const uint32_t level = blockIdx.y ; // <- the level is the same for all threads.
	const uint32_t feature = (blockIdx.x * blockDim.x + threadIdx.x) * N_FEATURES_PER_THREAD - i * N_FEATURES_PER_LEVEL;

	max_level = (max_level * num_grid_features) / N_FEATURES_PER_LEVEL;

	if (level > max_level + 1e-3f) {
		return;
	}

	grid_gradient += offset_table[level] * N_FEATURES_PER_LEVEL;
	const uint32_t hashmap_size = offset_table[level + 1] - offset_table[level];

	const float scale = grid_scale(level, log2_per_level_scale, base_resolution);
	const uint32_t resolution = grid_resolution(scale);

	auto add_grid_gradient = [&](const uvec<N_POS_DIMS>& local_pos, const tvec<GRAD_T, N_FEATURES_PER_THREAD>& grad, const float weight) {
		uint32_t index = grid_index<N_POS_DIMS>(hashmap_size, resolution, local_pos) * N_FEATURES_PER_LEVEL + feature;
		atomic_add_gmem(grid_gradient + index, (GRAD_T)weight * grad);
	};

	float pos[N_POS_DIMS];
	uvec<N_POS_DIMS> pos_grid;

	TCNN_PRAGMA_UNROLL
	for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
		pos_fract(positions_in(dim, i), &pos[dim], &pos_grid[dim], scale, identity_fun);
	}

	tvec<T, N_FEATURES_PER_THREAD> grad;

	TCNN_PRAGMA_UNROLL
	for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; ++f) {
		grad[f] = dL_dy[i + (level * N_FEATURES_PER_LEVEL + feature + f) * num_elements];
	}

	// N-linear interpolation
	TCNN_PRAGMA_UNROLL
	for (uint32_t idx = 0; idx < (1 << N_POS_DIMS); ++idx) {
		float weight = 1;
		uvec<N_POS_DIMS> pos_grid_local;

		TCNN_PRAGMA_UNROLL
		for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
			if ((idx & (1<<dim)) == 0) {
				weight *= 1 - pos[dim];
				pos_grid_local[dim] = pos_grid[dim];
			} else {
				weight *= pos[dim];
				pos_grid_local[dim] = pos_grid[dim] + 1;
			}
		}

		add_grid_gradient(pos_grid_local, grad, weight);
	}
}

template <typename T>
__global__ void transpose_encoded_position(
	const uint32_t n_elements,
	const T* __restrict__ encoded_positions,
	PitchedPtr<T> output
) {
	const uint32_t i = threadIdx.y + blockIdx.x * blockDim.y;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i;
	const uint32_t dim_idx = threadIdx.x;

	output(elem_idx)[dim_idx] = encoded_positions[elem_idx + n_elements * dim_idx];
}

template <typename T>
__global__ void transpose_gradients(
	const uint32_t n_elements,
	T* __restrict__ transposed_dL_dy,
	PitchedPtr<const T> dL_dy
) {
	const uint32_t i = threadIdx.y + blockIdx.x * blockDim.y;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i;
	const uint32_t dim_idx = threadIdx.x;

	transposed_dL_dy[elem_idx + n_elements * dim_idx] = dL_dy(elem_idx)[dim_idx];
}
