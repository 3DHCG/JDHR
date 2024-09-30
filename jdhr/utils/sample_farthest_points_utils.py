import jittor as jt
import numpy as np

def sample_farthest_points(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    #import ipdb; ipdb.set_trace()
    #device = xyz.device
    B, N, C = xyz.shape
    centroids = jt.zeros((B, npoint))
    distance = jt.ones((B, N)) * 1e10
    
    farthest = np.random.randint(0, N, B, dtype='l')
    batch_indices = np.arange(B, dtype='l')
    farthest = jt.array(farthest)
    batch_indices = jt.array(batch_indices) 
    # jt.sync_all(True)
    # print (xyz.shape, farthest.shape, batch_indices.shape, centroids.shape, distance.shape)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :]
        centroid = centroid.view(B, 1, 3) 

        dist = jt.sum((xyz - centroid.repeat(1, N, 1)) ** 2, 2) 
        mask = dist < distance
        # distance = mask.ternary(distance, dist)
        # print (mask.size())

        if mask.sum().data[0] > 0: 
            distance[mask] = dist[mask] # bug if mask.sum() == 0 

        farthest = jt.argmax(distance, 1)[0]
        # print (farthest)
        # print (farthest.shape)
    # B, N, C = xyz.size() 
    # sample_list = random.sample(range(0, N), npoint)
    # centroids = jt.zeros((1, npoint)) 
    # centroids[0,:] = jt.array(sample_list)
    # centroids = centroids.view(1, -1).repeat(B, 1)
    # x_center = x[:,sample_list, :]
    return centroids