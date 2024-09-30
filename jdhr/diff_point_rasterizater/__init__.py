from typing import NamedTuple
from jittor import nn
import jittor as jt 
from . import rasterize_points

class PointRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : jt.Var
    scale_modifier : float
    viewmatrix : jt.Var
    projmatrix : jt.Var
    sh_degree : int
    campos : jt.Var
    prefiltered : bool
    debug : bool


class _RasterizePoints(jt.Function):
    
    def save_for_backward(self,*args):
        self.saved_tensors = args
    
    def execute(self,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        radius,
        raster_settings,):
        
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            radius,
            raster_settings.scale_modifier,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )
        
        if raster_settings.debug:
            try:
                num_rendered, color, depth, alpha, radii, geomBuffer, binningBuffer, imgBuffer = rasterize_points.RasterizePointsCUDA(*args)
            except Exception as ex:
                jt.save(args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, depth, alpha, radii, geomBuffer, binningBuffer, imgBuffer = rasterize_points.RasterizePointsCUDA(*args)
        self.raster_settings = raster_settings
        self.num_rendered = num_rendered
        self.save_for_backward(colors_precomp, means3D, radius, opacities, radii, sh, geomBuffer, binningBuffer, imgBuffer, alpha)
        return color, depth, alpha, radii

    def grad(self,grad_out_color, grad_out_depth, grad_out_alpha, _):
        num_rendered = self.num_rendered
        raster_settings = self.raster_settings
        colors_precomp, means3D, radius, opacities, radii, sh, geomBuffer, binningBuffer, imgBuffer, alpha = self.saved_tensors
        if grad_out_depth is None:grad_out_depth=jt.zeros(grad_out_alpha.shape,dtype=jt.float)
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                radius, 
                opacities, 
                raster_settings.scale_modifier,  
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                grad_out_depth,
                grad_out_alpha,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                alpha,
                raster_settings.debug)
        
        if raster_settings.debug:
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_sh, grad_radius = rasterize_points.RasterizePointsBackwardCUDA(*args)
            except Exception as ex:
                jt.save(args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_sh, grad_radius = rasterize_points.RasterizePointsBackwardCUDA(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_radius,
            None,
        )
        del self.saved_tensors
        return grads
    
class PointRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings
        self.rasterizeFunc = _RasterizePoints()

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with jt.no_grad():
            raster_settings = self.raster_settings
            visible = rasterize_points.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible
    
    def execute(self, means3D, means2D, opacities, shs=None, colors_precomp=None, radius=None):
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if shs is None:
            shs = jt.array([])
        if colors_precomp is None:
            colors_precomp = jt.array([])

        if radius is None:
            radius = jt.array([])
        
        return self.rasterizeFunc(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            radius, 
            raster_settings, 
        )
