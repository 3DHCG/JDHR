import os 
import jittor as jt
from jittor import Function, exp, log
from typing import Tuple



header_path = os.path.join(os.path.dirname(__file__), 'cuda_rasterizer')
glm_path = os.path.join(os.path.dirname(__file__),'third_party','glm')
proj_options = {f'FLAGS: -I"{header_path}" -I"{glm_path}" -l"CudaRasterizer" -L"{os.path.dirname(__file__)}"':1}
# proj_options = {f'FLAGS: -I"./cuda_rasterizer" -I"./third_paart/glm" -l"CudaRasterizer" -L"./"':1}
jt.flags.use_cuda = 1
cuda_header = """
#include <math.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
//#include "config.h"
//#include "rasterizer.h"
#include "rasterizer_impl.h"

#include <fstream>
#include <string>
#include <functional>
std::function<char*(size_t N)> resizeFunctional(jittor::Var*& t) {
    auto lambda = [&t](size_t N) {
        t->set_shape({(long long)N});
		return reinterpret_cast<char*>(t->ptr<char>());
        //return t->ptr<char>();
    };
    return lambda;
}
"""

def mark_visible(means3D,viewmatrix,projmatrix):
    present = jt.zeros([means3D.shape[0]],dtype=jt.bool)
    out = jt.code(inputs=[means3D,viewmatrix,projmatrix],outputs=[present]
                  ,cuda_header=cuda_header,cuda_src='''
        @alias(means3D, in0)
        @alias(viewmatrix, in1)
        @alias(projmatrix,in2)
        @alias(present,out0)
        const int P = means3D_shape0;
        if(P != 0)
        {
            CudaRasterizer::Rasterizer::markVisible(P,
                means3D_p,
                viewmatrix_p,
                projmatrix_p,
                present_p;
        }
    ''')
    out.compile_options = proj_options
    return out

def bool_value(t):
    return 'true' if t else 'false' 

def compute_buffer_size(means3D,image_width,image_height):
    P = means3D.size(0)
    geom_size = jt.zeros([1,],'int64')
    img_size = jt.zeros([1,],'int64')
    
    geom_size,img_size = jt.code(outputs=[geom_size,img_size],inputs=[means3D],
        cuda_header=cuda_header,cuda_src=f'''
        @alias(geom_size, out0)
        @alias(img_size, out1)
        
        const int P = in0_shape0;
        size_t a = CudaRasterizer::required<CudaRasterizer::GeometryState>(P);
        cudaMemcpy(geom_size->ptr<size_t>(),&a,sizeof(size_t),cudaMemcpyHostToDevice);//,cudaMemcpyHostToDevice
        a = CudaRasterizer::required<CudaRasterizer::ImageState>({image_width} * {image_height});
        cudaMemcpy(img_size->ptr<size_t>(),&a,sizeof(size_t),cudaMemcpyHostToDevice);  //,cudaMemcpyHostToDevice      
        //  a = CudaRasterizer::required<CudaRasterizer::BinningState>(P * 16);      
        //  cudaMemcpy(binning_size->ptr<size_t>(),&a,sizeof(size_t),cudaMemcpyHostToDevice);        
        
    ''')
    geom_size.compile_options = proj_options
    return geom_size[0].item(),img_size[0].item()

def RasterizePointsCUDA(
    background:jt.Var,
    means3D:jt.Var,
    colors:jt.Var,
    opacity:jt.Var,
    radius:jt.Var,
    scale_modifier:float,
    viewmatrix:jt.Var,
    projmatrix:jt.Var,
    tan_fovx:float,
    tan_fovy:float,
    image_height:int,
    image_width:int,
    sh:jt.Var,
    degree:int,
    campos:jt.Var,
    prefiltered:bool,
    debug:bool
) -> Tuple[int, jt.Var, jt.Var, jt.Var, jt.Var, jt.Var,jt.Var,jt.Var]:
    
    image_height=int(image_height)
    image_width=int(image_width)
    geom_size,img_size = compute_buffer_size(means3D,image_width,image_height)
    
    
    with jt.flag_scope(compile_options=proj_options):
        geomBuffer = jt.array(jt.zeros([geom_size],dtype='uint8'))
        rendered = jt.array(jt.zeros([1],dtype='int32'))
        binning_size = jt.array(jt.zeros([1],dtype='int64'))
        radii = jt.array(jt.zeros([means3D.shape[0]],dtype='int32'))
        
        rendered,binning_size,radii = jt.code(
            outputs=[rendered,binning_size,radii],inputs=[background,means3D,colors,opacity,radius,viewmatrix,projmatrix,sh,campos,geomBuffer],
            data = {
                'image_height':image_height,
                'image_width':image_width,
                'scale_modifier':scale_modifier,
                'tan_fovx':tan_fovx,
                'tan_fovy':tan_fovy,  
                'degree':degree ,
            },
            cuda_header=cuda_header,
            cuda_src=f'''
                @alias(background, in0)
                @alias(means3D, in1)
                @alias(colors, in2)
                @alias(opacity, in3)
                @alias(radius, in4)
                @alias(viewmatrix, in5)
                @alias(projmatrix, in6)
                @alias(sh, in7)
                @alias(campos, in8)
                @alias(geomBuffer, in9)
                
                @alias(rendered, out0)
                @alias(binning_size, out1)
                @alias(radii, out2)
                
                const int P = means3D_shape0;
                const int H = data["image_height"];
                const int W = data["image_width"];
                if(P != 0){{
                    int M = 0;
                    if(sh_shape0 != 0)
                    {{
                        M = sh_shape1;
                    }}
                    if(colors_shape0 == 0) colors_p = nullptr;
                    int num_rendered = CudaRasterizer::Rasterizer::forward_0(
                        geomBuffer->ptr<char>(),
                        P, data["degree"], M,
                        background_p,
                        W, H,
                        means3D_p,
                        sh_p,
                        colors_p,
                        opacity_p,
                        radius_p,
                        data["scale_modifier"],
                        viewmatrix_p,
                        projmatrix_p,
                        campos_p,
                        data["tan_fovx"],
                        data["tan_fovy"],
                        {bool_value(prefiltered)},
                        radii_p,
                        {bool_value(debug)});
                    cudaMemcpy(rendered->ptr<int>(),&num_rendered,sizeof(int),cudaMemcpyHostToDevice);
                    size_t a = CudaRasterizer::required<CudaRasterizer::BinningState>(num_rendered);   
                    cudaMemcpy(binning_size->ptr<size_t>(),&a,sizeof(size_t),cudaMemcpyHostToDevice);
                }}
            '''
        )
        binningBuffer = jt.array(jt.zeros([binning_size[0].item()],dtype='uint8'))
        imageBuffer = jt.array(jt.zeros([img_size],dtype='uint8'))
        out_color = jt.array(jt.zeros([3,image_height,image_width],dtype='float32'))
        out_depth = jt.array(jt.zeros([1,image_height,image_width],dtype='float32'))
        out_alpha = jt.array(jt.zeros([1,image_height,image_width],dtype='float32'))
        
        binningBuffer,imageBuffer,out_color,out_depth,out_alpha = jt.code(
            outputs=[binningBuffer,imageBuffer,out_color,out_depth,out_alpha],
            inputs=[background,means3D,colors,opacity,radius,viewmatrix,projmatrix,sh,campos,geomBuffer,radii],
            data = {
                'image_height':image_height,
                'image_width':image_width,
                'scale_modifier':scale_modifier,
                'tan_fovx':tan_fovx,
                'tan_fovy':tan_fovy,  
                'degree':degree ,
                'num_rendered':rendered[0].item(),
            },
            cuda_header=cuda_header,
            cuda_src=f'''
                @alias(background, in0)
                @alias(means3D, in1)
                @alias(colors, in2)
                @alias(opacity, in3)
                @alias(radius, in4)
                @alias(viewmatrix, in5)
                @alias(projmatrix, in6)
                @alias(sh, in7)
                @alias(campos, in8)
                @alias(geomBuffer, in9)
                @alias(radii, in10)
                
                @alias(binningBuffer, out0)
                @alias(imageBuffer, out1)
                @alias(out_color, out2)
                @alias(out_depth, out3)
                @alias(out_alpha, out4)
                const int P = means3D_shape0;
                const int H = data["image_height"];
                const int W = data["image_width"];
                
                if(P != 0){{
                    int M = 0;
                    if(sh_shape0 != 0)
                    {{
                        M = sh_shape1;
                    }}
                    if(radii_shape0 == 0) radii_p = nullptr;
                    if(colors_shape0 == 0) colors_p = nullptr;
                    CudaRasterizer::Rasterizer::forward_1(
                        geomBuffer->ptr<char>(),    
                        binningBuffer->ptr<char>(),
                        imageBuffer->ptr<char>(),
                        P, data["degree"], M, data["num_rendered"],
                        background_p,
                        W, H,
                        means3D_p,
                        sh_p,
                        colors_p,
                        opacity_p,
                        radius_p,
                        data["scale_modifier"],
                        viewmatrix_p,
                        projmatrix_p,
                        campos_p,
                        data["tan_fovx"],
                        data["tan_fovy"],
                        {bool_value(prefiltered)},
                        out_color_p,
                        out_depth_p,
                        out_alpha_p,
                        radii_p,
                        {bool_value(debug)}
                    );
                }}
            '''
        )
    geomBuffer = geomBuffer.detach()
    binningBuffer = binningBuffer.detach()
    imageBuffer = imageBuffer.detach()
    return rendered[0].item(), out_color, out_depth,out_alpha,radii, geomBuffer, binningBuffer, imageBuffer

def RasterizePointsBackwardCUDA(
    background:jt.Var,
    means3D:jt.Var,
    radii:jt.Var,
    colors:jt.Var,
    radius:jt.Var,
    opacities:jt.Var,
    scale_modifier:float,
    viewmatrix:jt.Var,
    projmatrix:jt.Var,
    tan_fovx:float,
    tan_fovy:float,
    dL_dout_color:jt.Var,
    dL_dout_depth:jt.Var,
    dL_dout_alpha:jt.Var,
    sh:jt.Var,
    degree:int,
    campos:jt.Var,
    geomBuffer:jt.Var,
    R:int, 
    binningBuffer:jt.Var,
    imageBuffer:jt.Var,
    out_alpha:jt.Var,
    debug:bool
) -> Tuple[jt.Var, jt.Var, jt.Var, jt.Var, jt.Var, jt.Var]:
    
    P = means3D.size(0)
    M = sh.size(1) if sh.size(0)!= 0 else 0
     
    dL_dmeans2D=jt.zeros([P,3],dtype='float32')
    dL_dcolors= jt.zeros([P,3],dtype='float32')
    dL_dopacity= jt.zeros([P,1],dtype='float32')
    dL_dmeans3D=jt.zeros([P,3],dtype='float32')
    dL_dsh= jt.zeros([P,M,3],dtype='float32')
    dL_dradius= jt.zeros([P,1],dtype='float32')
    #just for storing intermediate results
    dL_dradius2D= jt.zeros([P,1],dtype='float32')
    dL_ddepths= jt.zeros([P,1],dtype='float32')

    dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dsh, dL_dradius,dL_dradius2D,dL_ddepths = jt.code(
        inputs=[background,means3D,radii,colors,radius,viewmatrix,projmatrix,dL_dout_color,dL_dout_depth,dL_dout_alpha,
        sh,campos,geomBuffer,binningBuffer,imageBuffer,out_alpha,opacities],
        outputs=[dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dsh, dL_dradius,dL_dradius2D,dL_ddepths],
        data={
                "R":R,
                "scale_modifier":scale_modifier,
                'tan_fovx':tan_fovx,
                "tan_fovy":tan_fovy,
                "degree":degree
            },
        cuda_header=cuda_header,
        cuda_src=f'''
            @alias(background, in0)
            @alias(means3D, in1)
            @alias(radii, in2)
            @alias(colors, in3)
            @alias(radius, in4)
            @alias(viewmatrix, in5)
            @alias(projmatrix, in6)
            @alias(dL_dout_color, in7)
            @alias(dL_dout_depth, in8)
            @alias(dL_dout_alpha, in9)
            @alias(sh, in10)
            @alias(campos, in11)
            @alias(geomBuffer, in12)
            @alias(binningBuffer, in13)
            @alias(imageBuffer, in14)
            @alias(out_alpha, in15)
            @alias(opacities, in16)
            
            @alias(dL_dmeans2D, out0)
            @alias(dL_dcolors, out1)
            @alias(dL_dopacity, out2)
            @alias(dL_dmeans3D, out3)
            @alias(dL_dsh, out4)
            @alias(dL_dradius, out5)
            @alias(dL_dradius2D, out6)
            @alias(dL_ddepths, out7)
            
            const int P = means3D_shape0;
            const int H = dL_dout_color_shape1;
            const int W = dL_dout_color_shape2;
            int M = 0;
            if(sh_shape0 != 0){{
                M = sh_shape1;
            }}
            
            
            
            if(P != 0){{
                if(radii_shape0 == 0) radii_p = nullptr;
                if(colors_shape0 == 0) colors_p = nullptr;
                
                CudaRasterizer::Rasterizer::backward(P, data["degree"], M, data["R"],
                background_p,
                W, H, 
                means3D_p,
                sh_p,
                colors_p,
                radius_p,
                data["scale_modifier"],
                opacities_p,
                viewmatrix_p,
                projmatrix_p,
                campos_p,
                data["tan_fovx"],
                data["tan_fovy"],
                radii_p,
                geomBuffer->ptr<char>(),
                binningBuffer->ptr<char>(),
                imageBuffer->ptr<char>(),
                out_alpha_p,
                dL_dout_color_p,
                dL_dout_depth_p,
                dL_dout_alpha_p,
                dL_dmeans2D_p,
                dL_dradius2D_p,  
                dL_dopacity_p,
                dL_dcolors_p,
                dL_ddepths_p,
                dL_dmeans3D_p,
                dL_dsh_p,
                dL_dradius_p,
                {bool_value(debug)});
            }}
        '''
    )
    dL_dmeans2D.compile_options = proj_options
    return dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dsh, dL_dradius



