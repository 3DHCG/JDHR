import os
import jittor as jt
from jittor import Function
from math import log2, pow, ceil
jt.flags.use_cuda = 1
proj_options = { f"FLAGS: --extended-lambda --expt-relaxed-constexpr": 1 }
proj_options[f"FLAGS: -DTCNN_MIN_GPU_ARCH={jt.flags.cuda_archs[0]}"] = 1


def enlarge(x: jt.Var, size: int):
    if x.shape[0] < size:
        y = jt.empty([size],x.dtype)
        x.assign(y)


class GridEncode(Function):
    def div_round_up(self, val, divisor, type):
        if type == int:
            return int((val + divisor - 1) / divisor)
        assert(type == int)

    def __init__(self, hash_func_header, per_level_scale, n_pos_dims=3, n_features_per_level=2, n_levels=16, base_resolution=16, log2_hashmap_size=19,n_rays_per_batch=512,MAX_STEP=1024,using_fp16=False,loss_scale=128.0):
        self.hash_func_header = hash_func_header
        m_per_level_scale = per_level_scale
        log2_per_level_scale = 0.400000036      # same with std::log2
        n_features = n_features_per_level * n_levels
        m_n_levels = self.div_round_up(n_features, n_features_per_level, int)
        offsets_table_host = [0 for x in range(33)]
        offset = 0
        for i in range(m_n_levels):
            # scale = pow(2, (i*log2(m_per_level_scale))) * base_resolution - 1.0
            scale = pow(2, (i*log2_per_level_scale)) * base_resolution - 1.0
            resolution = ceil(scale) + 1
            params_in_level = int(resolution)**int(n_pos_dims)
            params_in_level = self.div_round_up(params_in_level, 8, int)*8
            params_in_level = min(
                params_in_level, (1 << log2_hashmap_size))
            offsets_table_host[i] = offset
            offset += params_in_level
        offsets_table_host[m_n_levels] = offset
        m_n_params = offsets_table_host[m_n_levels] * n_features_per_level
        self.m_n_params=m_n_params
        self.m_hashmap_offsets_table = jt.empty([n_levels+1], 'int32')
        for i in range(m_n_levels+1):
            self.m_hashmap_offsets_table[i] = offsets_table_host[i]
        self.N_POS_DIMS = n_pos_dims
        self.N_FEATURES_PER_LEVEL = n_features_per_level
        self.m_n_features = n_features
        self.m_n_levels = n_levels
        self.m_base_resolution = base_resolution
        self.m_per_level_scale = m_per_level_scale
        self.m_max_level = 1000.0
        self.m_n_output_dims = n_features
        self.n_rays_per_batch=n_rays_per_batch
        self.MAX_STEP=MAX_STEP
        self.num_elements = self.n_rays_per_batch * self.MAX_STEP
        self.m_positions = jt.empty([self.num_elements * n_pos_dims], 'float')  # temp memory to save input.
        self.grad_type='float32'
        if using_fp16:
            self.grad_type='float16'
        self.loss_scale = loss_scale
        self.m_encoded_positions = jt.empty(
            [self.num_elements * n_features], self.grad_type)   # temp memory to save transposed output / output_gradient.
        self.m_grid_gradient = jt.empty([m_n_params], self.grad_type)
        header_path = os.path.join(os.path.dirname(__file__), 'op_header')
        proj_options[f"FLAGS: -I{header_path}"] = 1

    def execute(self, x, m_grid):
        self.num_elements=x.shape[0]
        assert(m_grid.dtype==self.grad_type)
        assert(self.m_encoded_positions.dtype==self.grad_type)
        self.m_positions = x.detach()
        output = jt.empty([self.num_elements,self.m_n_output_dims], self.grad_type)
        output, self.m_encoded_positions = jt.code([ self.m_hashmap_offsets_table, x, m_grid],[output, self.m_encoded_positions], 
        cuda_header=self.hash_func_header+'#include "HashEncode.h"', cuda_src=f"""
        #define grad_t in2_type
        @alias(m_encoded_positions, out1)
        @alias(hashmap_offsets_table, in0)
        cudaStream_t stream=0;
        const unsigned int num_elements=in1_shape0;
        if(num_elements==0){{
            return ;
        }}
		const int N_POS_DIMS={self.N_POS_DIMS};
        const int N_FEATURES_PER_LEVEL={self.N_FEATURES_PER_LEVEL};
        const int m_n_features={self.m_n_features};

        static constexpr uint32_t N_THREADS_HASHGRID = 512;
		const dim3 blocks_hashgrid = {{ div_round_up(num_elements, N_THREADS_HASHGRID), {self.m_n_levels}, 1 }};
        grad_t* m_grid = (grad_t*)in2_p;
        float* dy_dx = nullptr;
		kernel_grid<grad_t, N_POS_DIMS, N_FEATURES_PER_LEVEL><<<blocks_hashgrid, N_THREADS_HASHGRID, 0, stream>>>(
			num_elements,
            m_n_features,
			(uint32_t*)hashmap_offsets_table_p,
			{self.m_base_resolution},
			0.400000036f,
			{self.m_max_level},
            (grad_t*)m_grid,
			GPUMatrixDynamic<float>(in1_p, N_POS_DIMS, num_elements).view(),
			m_encoded_positions_p,
			dy_dx
		);
     
        const dim3 threads_transpose = {{ {self.m_n_levels * self.N_FEATURES_PER_LEVEL}, 8, 1 }};
		const uint32_t blocks_transpose = div_round_up(num_elements, threads_transpose.y);
       
	    transpose_encoded_position<grad_t><<<blocks_transpose, threads_transpose, 0, stream>>>(
			num_elements,
			m_encoded_positions_p,
			GPUMatrixDynamic<grad_t>(out0_p, out0_shape1, num_elements).pitched_ptr()
		);
        """)
        output.compile_options = proj_options
        self.m_encoded_positions = self.m_encoded_positions.detach()
        return output

    def grad(self, grad_x):
        if self.grad_type == "float16":
            grad_x = grad_x * self.loss_scale
            grad_x = grad_x.float16()
        self.m_grid_gradient, self.m_encoded_positions =\
            jt.code([self.m_positions, self.m_hashmap_offsets_table, grad_x], [self.m_grid_gradient, self.m_encoded_positions], 
            cuda_header=self.hash_func_header+'#include"HashEncode.h"', cuda_src=f"""
        #define grad_t in2_type
        @alias(m_positions, in0)
        @alias(m_encoded_positions, out1)
        @alias(hashmap_offsets_table, in1)
        @alias(m_grid_gradient, out0)
        const unsigned int num_elements = in2_shape0;
        if (num_elements==0) {{
            return;
        }}
        const unsigned int N_FEATURES_PER_LEVEL={self.N_FEATURES_PER_LEVEL};
        cudaStream_t stream = 0;
	    const dim3 threads_transpose = {{  {self.m_n_levels} * N_FEATURES_PER_LEVEL, 8, 1}};
        const uint32_t blocks_transpose = div_round_up(num_elements, threads_transpose.y);
        transpose_gradients<grad_t><<<blocks_transpose, threads_transpose, 0, stream>>>(
            num_elements,
            m_encoded_positions_p,
            GPUMatrixDynamic<const grad_t>(in2_p, in2_shape1, num_elements).pitched_ptr()
        );

        CUDA_CHECK_THROW(cudaMemsetAsync(m_grid_gradient_p, 0, out0->size, stream));
        static constexpr uint32_t N_THREADS_HASHGRID = 256;
        static constexpr uint32_t N_FEATURES_PER_THREAD = std::min(2u, N_FEATURES_PER_LEVEL);
        const dim3 blocks_hashgrid = {{div_round_up(num_elements * N_FEATURES_PER_LEVEL / N_FEATURES_PER_THREAD, N_THREADS_HASHGRID), {self.m_n_levels}, 1}};
   
        kernel_grid_backward<grad_t, grad_t, {self.N_POS_DIMS}, N_FEATURES_PER_LEVEL, N_FEATURES_PER_THREAD><<<blocks_hashgrid, N_THREADS_HASHGRID, 0, stream>>>(
            num_elements,
            {self.m_n_features},
            (uint32_t*)hashmap_offsets_table_p,
            {self.m_base_resolution},
            0.400000036f,
            {self.m_max_level},
            m_grid_gradient_p,
            GPUMatrixDynamic<const float>(m_positions_p, {self.N_POS_DIMS}, num_elements).view(),	    // positions SoA
            (grad_t*)m_encoded_positions_p      // gradients SoA
        );

        """)

        self.m_grid_gradient.compile_options = proj_options
        self.m_positions = self.m_positions.detach()
        self.m_encoded_positions = self.m_encoded_positions.detach()
    
        if self.grad_type == "float16":
            self.m_grid_gradient = self.m_grid_gradient / self.loss_scale
        return None, self.m_grid_gradient
