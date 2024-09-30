import os
import jittor as jt
from jittor import Function
import numpy as np

from jdhr.engine import EMBEDDERS

proj_options = { f"FLAGS: --extended-lambda --expt-relaxed-constexpr": 1 }
proj_options[f"FLAGS: -DTCNN_MIN_GPU_ARCH={jt.flags.cuda_archs[0]}"] = 1

@EMBEDDERS.register_module()
class TcnnDirEmbedderjt(Function):
    def __init__(self) :
        #self.cfg = get_cfg()
        using_fp16 = False#self.cfg.fp16
        self.num_elements=4194304
        self.m_n_padded_output_dims=16
        self.m_sh_degree=4
        self.m_n_to_pad=0
        if using_fp16:
            self.grad_type='float16'
        else:
            self.grad_type='float32'
        header_path = os.path.join(os.path.dirname(__file__), 'op_header')
        proj_options[f"FLAGS: -I{header_path}"]=1
        self.out_dim=self.m_n_padded_output_dims

    def execute(self,x) :
        #print("x.",x.shape)
        B,P,_=x.shape
        x=x.view(-1,3)
        self.num_elements=x.shape[0]

        output=jt.code((self.num_elements,16),self.grad_type,[x],cuda_header='#include "SphericalEncode.h"',cuda_src=f"""

       #define grad_t out_type

        uint32_t num_elements=in0_shape0;
        uint32_t m_n_padded_output_dims={self.m_n_padded_output_dims};
        uint32_t m_sh_degree={self.m_sh_degree};
        uint32_t m_n_to_pad={self.m_n_to_pad};

        cudaStream_t stream=0;

        PitchedPtr<const float> inputs={{in0_p,in0_shape1}};
		PitchedPtr<grad_t> outputs={{out_p,out_shape1}};
		float* dy_dx = nullptr;
        linear_kernel(kernel_sh<grad_t>, 0, stream,
			num_elements,
			m_sh_degree,
			m_n_to_pad,
			inputs,
            outputs,
			dy_dx
		);
        """)
        output.compile_options=proj_options
        output=output.view(B,P,-1)
        #print("output.shape",output.shape)
        return output

    def grad(self,grad_x):
        return None
