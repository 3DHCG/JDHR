import cv2
#import torch
import jittor as jt
import trimesh
import numpy as np
import open3d as o3d
from jittor import nn
from PIL import Image
from plyfile import PlyData, PlyElement

from jdhr.utils.console_utils import *
from jdhr.utils.chunk_utils import multi_gather
from jdhr.utils.ray_utils import create_meshgrid
from jdhr.utils.math_utils import affine_inverse, affine_padding
from jdhr.utils.tsdf_utils import TSDF, TSDFFuser


def voxel_reconstruction(pcd: jt.Var, voxel_size: float = 0.01):
    if isinstance(pcd, np.ndarray): pcd = jt.array(pcd)  # for a consistent api

    # Convert torch tensor to Open3D PointCloud
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.view(-1, 3).detach().cpu().numpy())

    # Create VoxelGrid from PointCloud
    o3d_vox = o3d.geometry.VoxelGrid.create_from_point_cloud(o3d_pcd, voxel_size=voxel_size)

    # Extract dense grid from VoxelGrid using get_voxel
    voxels = o3d_vox.get_voxels()
    max_index = np.array([vox.grid_index for vox in voxels]).max(axis=0)  # !: for-loop
    dense_grid = np.zeros((max_index[0] + 1, max_index[1] + 1, max_index[2] + 1))

    for vox in voxels:  # !: for-loop
        dense_grid[vox.grid_index[0], vox.grid_index[1], vox.grid_index[2]] = 1

    # Use marching cubes to obtain mesh from dense grid
    vertices, triangles = mcubes.marching_cubes(dense_grid, 0.5)
    vertices = vertices * voxel_size + o3d_vox.origin  # resizing
    return vertices, triangles


def filter_global_points(points: dotdict[str, jt.Var]):
    from jdhr.utils.fcds_utils import remove_outlier

    def gather_from_inds(ind: jt.Var, scalars: dotdict()):
        return dotdict({k: multi_gather(v, ind[..., None]) for k, v in scalars.items()})

    # Remove NaNs in point positions
    ind = ((points.pts.isnan()).logical_not())[..., 0].nonzero()[..., 0]  # P,
    points = gather_from_inds(ind, points)

    # Remove low density points
    ind = (points.occ > 0.01)[..., 0].nonzero()[..., 0]  # P,
    points = gather_from_inds(ind, points)

    # # Remove statistic outliers (il_ind -> inlier indices)
    # ind = remove_outlier(points.pts[None], K=50, std_ratio=4.0, return_inds=True)[0]  # P,
    # points = gather_from_inds(ind, points)

    return points

# *************************************
# Our PyTorch implementation
# *************************************


def depth_reprojection(dpt_ref: jt.Var,  # B, H, W
                       ixt_ref: jt.Var,  # B, 3, 3
                       ext_ref: jt.Var,  # B, 4, 4
                       dpt_src: jt.Var,  # B, S, H, W
                       ixt_src: jt.Var,  # B, S, 3, 3
                       ext_src: jt.Var,  # B, S, 4, 4
                       ):
    sh = dpt_ref.shape[:-2]
    H, W = dpt_ref.shape[-2:]

    # step1. project reference pixels to the source view
    # reference view x, y
    ij_ref = create_meshgrid(H, W, dtype=dpt_ref.dtype)
    xy_ref = ij_ref.flip(-1)
    for _ in range(len(sh)): xy_ref = xy_ref[None]  # add dimension
    xy_ref = xy_ref.view(*sh, -1, 2)  # B, HW, 2
    x_ref, y_ref = xy_ref.split(1, dim=-1)  # B, HW, 1

    xy1_ref = jt.cat([xy_ref, jt.ones_like(xy_ref[..., :1])], dim=-1)  # B, HW, 3
    # reference 3D space
    xyz_ref = xy1_ref @ ixt_ref.inverse().transpose(-2,-1) * dpt_ref.view(*sh, H * W, 1)  # B, HW, 3

    # source 3D space
    xyz1_ref = jt.cat([xyz_ref, jt.ones_like(xyz_ref[..., :1])], dim=-1)  # B, HW, 4
    xyz_src = ((xyz1_ref @ affine_inverse(ext_ref).transpose(-2,-1)).unsqueeze(-3) @ ext_src.transpose(-2,-1))[..., :3]  # B, S, HW, 3
    # source view x, y
    K_xyz_src = xyz_src @ ixt_src.transpose(-2,-1)  # B, S, HW, 3
    xy_src = K_xyz_src[..., :2] / K_xyz_src[..., 2:3]  # homography reprojection, B, S, HW, 2

    # step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[..., 0].view(*sh, -1, H, W)  # B, S, H, W
    y_src = xy_src[..., 1].view(*sh, -1, H, W)  # B, S, H, W

    dpt_input = dpt_src.unsqueeze(-3)  # B, S, 1, H, W
    xy_grid = jt.stack([x_src / W * 2 - 1, y_src / H * 2 - 1], dim=-1)  # B, S, H, W, 2
    bs = dpt_input.shape[:-3]  # BS
    dpt_input = dpt_input.view(-1, *dpt_input.shape[-3:])  # BS, H, W, 2
    xy_grid = xy_grid.view(-1, *xy_grid.shape[-3:])  # BS, 1, H, W
    sampled_depth_src = nn.grid_sample(dpt_input, xy_grid, padding_mode='border')  # BS, 1, H, W # sampled src depth map
    sampled_depth_src = sampled_depth_src.view(*bs, H, W)  # B, S, H, W
    mask_projected = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xy1_src = jt.cat([xy_src, jt.ones_like(xy_src[..., :1])], dim=-1)  # B, S, HW, 3
    xyz_src = (xy1_src @ jt.linalg.inv(ixt_src).transpose(-2,-1)) * sampled_depth_src.view(*sh, -1, H * W, 1)  # B, S, HW, 3

    # reference 3D space
    xyz1_src = jt.cat([xyz_src, jt.ones_like(xyz_src[..., :1])], dim=-1)  # B, S, HW, 4
    xyz_reprojected = ((xyz1_src @ affine_inverse(ext_src).transpose(-2,-1)) @ ext_ref.transpose(-2,-1).unsqueeze(-3))[..., :3]  # B, S, HW, 3

    # source view x, y, depth
    depth_reprojected = xyz_reprojected[..., 2].view(*sh, -1, H, W)  # source depth in ref view space, B, S, H, W
    K_xyz_reprojected = xyz_reprojected @ ixt_ref.unsqueeze(-3).transpose(-2,-1)  # B, S, HW, 3  # source xyz in ref screen space
    xy_reprojected = K_xyz_reprojected[..., :2] / K_xyz_reprojected[..., 2:3]  # homography
    x_reprojected = xy_reprojected[..., 0].view(*sh, -1, H, W)  # source point in ref screen space, x: B, S, H, W
    y_reprojected = xy_reprojected[..., 1].view(*sh, -1, H, W)  # source point in ref screen space, y: B, S, H, W

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src, mask_projected


def depth_geometry_consistency(
    dpt_ref: jt.Var,  # B, H, W
    ixt_ref: jt.Var,  # B, 3, 3
    ext_ref: jt.Var,  # B, 4, 4
    dpt_src: jt.Var,  # B, S, H, W
    ixt_src: jt.Var,  # B, S, 3, 3
    ext_src: jt.Var,  # B, S, 4, 4

    geo_abs_thresh: float = 0.5,
    geo_rel_thresh: float = 0.01,
):
    # Assumption: same sized image
    # Assumption: zero depth -> no depth, should be masked out as photometrically inconsistent
    sh = dpt_ref.shape[:-2]
    H, W = dpt_ref.shape[-2:]

    # Step1. project reference pixels to the source view
    # Reference view x, y
    ij_ref = create_meshgrid(H, W, dtype=dpt_ref.dtype)
    xy_ref = ij_ref.flip(-1)
    for _ in range(len(sh)): xy_ref = xy_ref[None]  # add dimension
    xy_ref = xy_ref.view(*sh, H, W, 2)  # B, H, W, 2
    x_ref, y_ref = xy_ref.unbind(-1)  # B, H, W
    x_ref: jt.Var  # add type annotation
    y_ref: jt.Var  # add type annotation

    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src, mask_projected = depth_reprojection(
        dpt_ref, ixt_ref, ext_ref,
        dpt_src, ixt_src, ext_src)

    # Check |p_reproj-p_1| < 1
    dist = jt.sqrt((x2d_reprojected - x_ref.unsqueeze(-3)) ** 2 + (y2d_reprojected - y_ref.unsqueeze(-3)) ** 2)  # B, S, H, W

    # Check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = jt.abs(depth_reprojected - dpt_ref.unsqueeze(-3))  # unprojected depth difference
    relative_depth_diff = depth_diff / dpt_ref  # relative unprojected depth difference  # B, S, H, W

    mask = jt.logical_and(dist < geo_abs_thresh, relative_depth_diff < geo_rel_thresh)  # smaller than 0.5 pix, relative smaller than 0.01
    mask = jt.logical_and(mask, mask_projected)
    depth_reprojected = jt.ternary(mask, depth_reprojected, 0)

    return mask, depth_reprojected, x2d_src, y2d_src


def compute_consistency(
    dpt_ref: jt.Var,  # B, H, W
    ixt_ref: jt.Var,  # B, 3, 3
    ext_ref: jt.Var,  # B, 4, 4
    dpt_src: jt.Var,  # B, S, H, W
    ixt_src: jt.Var,  # B, S, 3, 3
    ext_src: jt.Var,  # B, S, 4, 4

    geo_abs_thresh: float = 0.5,
    geo_rel_thresh: float = 0.01,
    geo_sum_thresh: float = 0.75,
    pho_abs_thresh: float = 0.0,
):
    # Perform actual geometry consistency check
    geo_mask, depth_reprojected, x2d_src, y2d_src = depth_geometry_consistency(
        dpt_ref, ixt_ref, ext_ref, dpt_src, ixt_src, ext_src,
        geo_abs_thresh, geo_rel_thresh
    )  # [B, H, W]; [B, S, H, W]; [B, S, H, W]; [B, S, H, W]

    # Aggregate the projected mask
    geo_mask_sum = geo_mask.sum(-3)  # B, H, W
    depth_est_averaged = (depth_reprojected.sum(-3) + dpt_ref) / (geo_mask_sum + 1)  # average depth values, B, H, W

    # At least 3 source views matched
    geo_mask = geo_mask_sum >= (geo_sum_thresh * dpt_src.shape[-3])  # a pixel is considered valid when at least 3 sources matches up
    photo_mask = dpt_ref > pho_abs_thresh
    final_mask = jt.logical_and(photo_mask, geo_mask)  # B, H, W

    return depth_est_averaged, photo_mask, geo_mask, final_mask


# *************************************
# Original implementation in cvp-mvsnet
# *************************************
"""
Copied from: https://github.com/JiayuYANG/CVP-MVSNet
"""


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

    return intrinsics, extrinsics


def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            data.append((ref_view, src_views))
    return data

# read an image


def read_img(filename):
    img = Image.open(filename)
    # Crop image (Hard code dtu image size here)
    left = 0
    top = 0
    right = 1600
    bottom = 1184
    img = img.crop((left, top, right, bottom))

    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.uint8)
    return np_img


# read a binary mask
def read_mask(filename):
    return read_img(filename) > 0.5


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


def save_pfm(filename, image, scale=1):

    if not os.path.exists(os.path.dirname(filename)):
        # try:
        os.makedirs(os.path.dirname(filename))
        # except OSError as exc: # Guard against race condition
        #     if exc.errno != errno.EEXIST:
        #         raise

    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()


def write_depth_img(filename, depth, depth_min, depth_max):

    if not os.path.exists(os.path.dirname(filename)):
        # try:
        os.makedirs(os.path.dirname(filename))
        # except OSError as exc: # Guard against race condition
        #     if exc.errno != errno.EEXIST:
        #         raise

    image = Image.fromarray((depth - depth_min) / (depth_max - depth_min) * 255).convert("L")
    image.save(filename)
    return 1


# project the reference point cloud into the source view, then project back


def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    # step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))  # int
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])  # int
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))  # view space ref xyz 2, HW
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]  # ref xyz in src view space 2, HW
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)  # ref xyz in src screen space
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]  # homography reprojection, 2, HW

    # step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)  # H, W
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)  # H, W
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)  # sampled src depth map
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))  # unproject back to src view space, 3, HW
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]  # unprojected points back to ref view space, 3, HW
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)  # source depth in ref view space, H, W
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)  # source xyz in ref screen space, 3, HW
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]  # homography
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)  # source point in ref screen space, x: H, W
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)  # source point in ref screen space, y: H, W

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                                                                 depth_src, intrinsics_src, extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)  # unprojected depth difference
    relative_depth_diff = depth_diff / depth_ref  # relative unprojected depth difference

    mask = np.logical_and(dist < 0.5, relative_depth_diff < 0.01)  # smaller than 0.5 pix, relative smaller than 0.01
    depth_reprojected[mask.logical_not()] = 0  # those are valid points

    return mask, depth_reprojected, x2d_src, y2d_src


def filter_depth(dataset_root, scan, out_folder, plyfilename):

    print("Starting fusion for:" + out_folder)

    # the pair file
    pair_file = os.path.join(dataset_root, 'Cameras/pair.txt')
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)
    nviews = len(pair_data)

    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:
        # load the camera parameters
        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(dataset_root, 'Cameras/{:0>8}_cam.txt'.format(ref_view)))

        # load the reference image
        ref_img = read_img(os.path.join(dataset_root, "Rectified", scan, 'rect_{:03d}_3_r5000.png'.format(ref_view + 1)))  # Image start from 1.
        # load the estimated depth of the reference view
        ref_depth_est, scale = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))
        # load the photometric mask of the reference view
        confidence, scale = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))
        photo_mask = confidence > 0.9

        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []

        # compute the geometric mask
        geo_mask_sum = 0
        for src_view in src_views:
            # camera parameters of the source view
            src_intrinsics, src_extrinsics = read_camera_parameters(
                os.path.join(dataset_root, 'Cameras/{:0>8}_cam.txt'.format(src_view)))

            # the estimated depth of the source view
            src_depth_est, scale = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))

            geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                                                        src_depth_est, src_intrinsics, src_extrinsics)

            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)
            all_srcview_x.append(x2d_src)
            all_srcview_y.append(y2d_src)
            all_srcview_geomask.append(geo_mask)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)  # average depth values
        # at least 3 source views matched
        geo_mask = geo_mask_sum >= 3  # a pixel is considered valid when at least 3 sources matches up
        final_mask = np.logical_and(photo_mask, geo_mask)

        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)

        print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(scan, ref_view,
                                                                                    photo_mask.mean(),
                                                                                    geo_mask.mean(), final_mask.mean()))

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        # valid_points = np.logical_and(final_mask, ~used_mask[ref_view])
        valid_points = final_mask
        print("valid_points", valid_points.mean())
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        ref_img = np.array(ref_img)

        color = ref_img[valid_points]

        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]  # convert valid depths to world space
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color).astype(np.uint8))

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    print("Saving the final model to", plyfilename)
    PlyData([el], comments=['Model created by CVP-MVSNet.']).write(plyfilename)
    print("Model saved.")


# *************************************
# Fusion implementation in Simple-Recon
# *************************************
"""
Copied from: https://github.com/nianticlabs/simplerecon/blob/main/tools/fusers_helper.py
"""
"""
    This is borrowed from https://github.com/alexrich021/3dvnet/blob/main/mv3d/eval/pointcloudfusion_custom.py
"""


def reverse_imagenet_normalize(image):
    """ Reverses ImageNet normalization in an input image. """

    image = TF.normalize(tensor=image,
                         mean=(-2.11790393, -2.03571429, -1.80444444),
                         std=(4.36681223, 4.46428571, 4.44444444))
    return image


class DepthFuser():
    def __init__(
        self,
        gt_path="",
        fusion_resolution=0.04,
        max_fusion_depth=3.0,
        fuse_color=False
    ):
        self.fusion_resolution = fusion_resolution
        self.max_fusion_depth = max_fusion_depth


class OurFuser(DepthFuser):
    """
    This is the fuser used for scores in the SimpleRecon paper. Note that
    unlike open3d's fuser this implementation does not do voxel hashing. If a
    path to a known mehs reconstruction is provided, this function will limit
    bounds to that mesh's extent, otherwise it'll use a wide volume to prevent
    clipping.

    It's best not to use this fuser unless you need to recreate numbers from the
    paper.

    """

    def __init__(
        self,
        gt_path="",
        fusion_resolution=0.04,
        max_fusion_depth=3,
        fuse_color=False,
    ):
        super().__init__(
            gt_path,
            fusion_resolution,
            max_fusion_depth,
            fuse_color,
        )

        if gt_path is not None:
            gt_mesh = trimesh.load(gt_path, force='mesh')
            tsdf_pred = TSDF.from_mesh(gt_mesh, voxel_size=fusion_resolution)
        else:
            bounds = {}
            bounds["xmin"] = -10.0
            bounds["xmax"] = 10.0
            bounds["ymin"] = -10.0
            bounds["ymax"] = 10.0
            bounds["zmin"] = -10.0
            bounds["zmax"] = 10.0

            tsdf_pred = TSDF.from_bounds(bounds, voxel_size=fusion_resolution)

        self.tsdf_fuser_pred = TSDFFuser(tsdf_pred, max_depth=max_fusion_depth)

    def fuse_frames(self, depths_b1hw, K_b44,
                    cam_T_world_b44,
                    color_b3hw):
        self.tsdf_fuser_pred.integrate_depth(
            depth_b1hw=depths_b1hw.half(),
            cam_T_world_T_b44=cam_T_world_b44.half(),
            K_b44=K_b44.half(),
        )

    def export_mesh(self, path, export_single_mesh=True):
        _ = trimesh.exchange.export.export_mesh(
            self.tsdf_fuser_pred.tsdf.to_mesh(
                export_single_mesh=export_single_mesh),
            path,
        )

    def get_mesh(self, export_single_mesh=True, convert_to_trimesh=True):
        return self.tsdf_fuser_pred.tsdf.to_mesh(
            export_single_mesh=export_single_mesh)


class Open3DFuser(DepthFuser):
    """
    Wrapper class for the open3d fuser.

    This wrapper does not support fusion of tensors with higher than batch 1.
    """

    def __init__(
        self,
        gt_path="",
        fusion_resolution=0.04,
        max_fusion_depth=3,
        fuse_color=False,
        use_upsample_depth=False,
    ):
        super().__init__(
            gt_path,
            fusion_resolution,
            max_fusion_depth,
            fuse_color,
        )

        self.fuse_color = fuse_color
        self.use_upsample_depth = use_upsample_depth
        self.fusion_max_depth = max_fusion_depth

        voxel_size = fusion_resolution * 100
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=float(voxel_size) / 100,
            sdf_trunc=3 * float(voxel_size) / 100,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

    def fuse_frames(
        self,
        depths_b1hw,
        K_b44,
        cam_T_world_b44,
        color_b3hw,
    ):

        width = depths_b1hw.shape[-1]
        height = depths_b1hw.shape[-2]

        if self.fuse_color:
            color_b3hw = jt.nn.interpolate(
                color_b3hw,
                size=(height, width),
            )
            color_b3hw = reverse_imagenet_normalize(color_b3hw)

        for batch_index in range(depths_b1hw.shape[0]):
            if self.fuse_color:
                image_i = color_b3hw[batch_index].permute(1, 2, 0)

                color_im = (image_i * 255).cpu().numpy().astype(
                    np.uint8
                ).copy(order='C')
            else:
                # mesh will now be grey
                color_im = 0.7 * jt.ones_like(
                    depths_b1hw[batch_index]
                ).squeeze().cpu().clone().numpy()
                color_im = np.repeat(
                    color_im[:, :, np.newaxis] * 255,
                    3,
                    axis=2
                ).astype(np.uint8)

            depth_pred = depths_b1hw[batch_index].squeeze().cpu().clone().numpy()
            depth_pred = o3d.geometry.Image(depth_pred)
            color_im = o3d.geometry.Image(color_im)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_im,
                depth_pred,
                depth_scale=1.0,
                depth_trunc=self.fusion_max_depth,
                convert_rgb_to_intensity=False,
            )
            cam_intr = K_b44[batch_index].cpu().clone().numpy()
            cam_T_world_44 = cam_T_world_b44[batch_index].cpu().clone().numpy()

            self.volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(
                    width=width,
                    height=height, fx=cam_intr[0, 0],
                    fy=cam_intr[1, 1],
                    cx=cam_intr[0, 2],
                    cy=cam_intr[1, 2]
                ),
                cam_T_world_44,
            )

    def export_mesh(self, path, use_marching_cubes_mask=None):
        o3d.io.write_triangle_mesh(path, self.volume.extract_triangle_mesh())

    def get_mesh(self, export_single_mesh=None, convert_to_trimesh=False):
        mesh = self.volume.extract_triangle_mesh()

        if convert_to_trimesh:
            mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.triangles)

        return mesh


def get_fuser(opts, scan):
    """Returns the depth fuser required. Our fuser doesn't allow for """

    if opts.dataset == "scannet":
        gt_path = ScannetDataset.get_gt_mesh_path(opts.dataset_path,
                                                  opts.split, scan)
    else:
        gt_path = None

    if opts.depth_fuser == "ours":
        if opts.fuse_color:
            print("WARNING: fusing color using 'ours' fuser is not supported, "
                  "Color will not be fused.")

        return OurFuser(
            gt_path=gt_path,
            fusion_resolution=opts.fusion_resolution,
            max_fusion_depth=opts.fusion_max_depth,
            fuse_color=False,
        )
    if opts.depth_fuser == "open3d":
        return Open3DFuser(
            gt_path=gt_path,
            fusion_resolution=opts.fusion_resolution,
            max_fusion_depth=opts.fusion_max_depth,
            fuse_color=opts.fuse_color,
        )
    else:
        raise ValueError("Unrecognized fuser!")
