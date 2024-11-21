import torch
import copy
import pickle
import sklearn
import math

import numpy as np
import pyransac3d as pyrsc

from numba import jit
from scipy import spatial
from pathlib import Path
from pytorch3d.ops.knn import knn_gather, knn_points

from scipy.spatial.transform import Rotation as R

from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils


def apply_transform(pts, transformation, box=False, mode='left'):
    if len(pts) == 0:
        return pts
    
    einsum = np.einsum
    if torch.is_tensor(pts):
        pts_ = pts.clone().detach()
        einsum = torch.einsum
        pts_wo_h = torch.concat((pts_[:, :3], pts_.new_ones((len(pts_), 1))), axis=1)
    else:
        pts_ = copy.deepcopy(pts)
        pts_wo_h = np.hstack((pts_[:, :3], np.ones((len(pts_), 1))))

    if mode == 'left':
        pts_[..., :3] = einsum('ij,kj->ki', transformation, pts_wo_h)[..., :3]
    elif mode == 'right':
        pts_[..., :3] = einsum('ij,jk->ik', pts_wo_h, transformation)[..., :3]
    else:
        raise NotImplementedError

    if box:
        rot_mat = transformation[:3, :3] if not torch.is_tensor(transformation) else transformation[:3, :3].cpu().numpy()
        yaw = R.from_matrix(rot_mat).as_euler('xyz')[-1]
        pts_[..., 6] += yaw

    return pts_


def mask_ground_points_patchwork_pp(points, patchwork_pp, z_offset=0.0):
    pts = np.concatenate([points[..., :4].copy(), np.arange(points.shape[0])[..., None]], axis=-1)
    pts[..., 2] -= z_offset # semantic kitti sensor height
   
    patchwork_pp.estimateGround(pts)
    ground = patchwork_pp.getGround()
    
    return ground[..., -1].astype(int)


def load_frame_data(file_path: Path) -> dict:
    frame_data = {}
    if file_path.exists():
        with open(file_path, 'rb') as f:
            frame_data = pickle.load(f)
    return frame_data


def save_frame_data(data, file_path: Path) -> bool:
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
        
    return True


def count_neighbors(pts_buffer, seek=0, skip_frames=1, max_neighbor_point_dist=0.3, max_neighbor_points=1000, **kwargs):
    skip = skip_frames + 1
    neighbor_count = []
    pts_query = pts_buffer[seek]
    idx_list = list(range(len(pts_buffer)))
    # idx_list.pop(seek)
    
    for i in idx_list[::skip]:
        pts_target = pts_buffer[i]
        idx, empty_ball_mask = pointnet2_utils.ball_query(max_neighbor_point_dist, max_neighbor_points, 
                                                          pts_target, torch.tensor([pts_target.shape[0]]).int().cuda(),
                                                          pts_query, torch.tensor([pts_query.shape[0]]).int().cuda())
        count = torch.count_nonzero(idx.squeeze() != idx.squeeze()[..., 0][:, None], dim=1)
        count[~empty_ball_mask] += 1
        count = count.cpu().numpy()
        if i == seek:
            count -= 1
        neighbor_count.append(count)
    neighbor_count = np.stack(neighbor_count).T
    
    return neighbor_count


def count_neighbors_inter_frame(points, max_neighbor_point_dist=0.1, max_neighbor_points=100):
    points_t = torch.from_numpy(points[..., :3]).cuda()
    idx, empty_ball_mask = pointnet2_utils.ball_query(max_neighbor_point_dist, max_neighbor_points, 
                                                          points_t, torch.tensor([points_t.shape[0]]).int().cuda(),
                                                          points_t, torch.tensor([points_t.shape[0]]).int().cuda())
    count = torch.count_nonzero(idx.squeeze() != idx.squeeze()[..., 0][:, None], dim=1)
    count[~empty_ball_mask] += 1
    return count.cpu().numpy()


def compute_ephe_score(count, ephe_type="entropy"):
    N = count.shape[1]
    if ephe_type == "entropy":
        P = count / (np.expand_dims(count.sum(axis=1), -1) + 1e-8)
        H = (-P * np.log(P + 1e-8)).sum(axis=1) / np.log(N)
    else:
        raise NotImplementedError()
    return H


def calculate_entropy_scores(frame_buffer, seek=0, **kwargs):
    count = count_neighbors(frame_buffer, seek=seek, **kwargs)
    H = compute_ephe_score(count)
    return H


def poly_area_2d(pts):
    lines = np.hstack([pts, np.roll(pts, -1, axis=0)])
    area = 0.5 * abs(sum(x1 * y2 - x2 * y1 for x1 , y1, x2, y2 in lines))
    return area


def get_lowest_point_rect(ptc, xy_center, l, w, rz):
    ptc_xy = ptc[:, [0, 1]] - xy_center
    rot = np.array([
        [np.cos(rz), -np.sin(rz)],
        [np.sin(rz), np.cos(rz)]
    ])
    ptc_xy = ptc_xy @ rot.T
    mask = (ptc_xy[:, 0] > -l/2) & \
        (ptc_xy[:, 0] < l/2) & \
        (ptc_xy[:, 1] > -w/2) & \
        (ptc_xy[:, 1] < w/2)
    zs = ptc[mask, 2]
    return zs.min()


def get_box_heights(points, boxes):
    boxes_new = boxes.copy()
    box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()
    for i in range(len(boxes)):
        box_points = points[box_idxs_of_pts == i]
        if len(box_points) > 0:
            min_z = box_points[:, 2].min()
            max_z = box_points[:, 2].max()
            height = max_z - min_z
            boxes_new[i, 2] = min_z + height / 2
            boxes_new[i, 5] = height
    return boxes_new


@jit(nopython=True)
def min_axis_zero_2d(x, y):
    for i in range(len(x)):
        if x[i] < y[i]:
            y[i] = x[i]
    return y


@jit(nopython=True)
def check_all_angles(cluster_points, delta, delta_zero):
    max_beta = -math.inf
    choose_angle = None
    for angle in np.arange(0, 90 + delta, delta):
        angle = angle / 180. * np.pi
        components = np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ], dtype=np.float32)
        projection = np.dot(cluster_points, components.T)
        min_x, max_x = projection[:,0].min(), projection[:,0].max()
        min_y, max_y = projection[:,1].min(), projection[:,1].max()
        Dx = min_axis_zero_2d(projection[:, 0] - min_x, max_x - projection[:, 0])
        Dy = min_axis_zero_2d(projection[:, 1] - min_y, max_y - projection[:, 1])
        beta = min_axis_zero_2d(Dx, Dy)
        beta = np.maximum(beta, delta_zero)
        beta = 1 / beta
        beta = beta.sum()
        if beta > max_beta:
            max_beta = beta
            choose_angle = angle
            
    return choose_angle


def closeness_rectangle(cluster_points, delta=2, delta_zero=1e-2):
    angle = check_all_angles(cluster_points, delta, delta_zero)
    
    components = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ], dtype=np.float32)
    projection = np.dot(cluster_points, components.T)

    min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
    min_y, max_y = projection[:, 1].min(), projection[:, 1].max()

    if (max_x - min_x) < (max_y - min_y):
        angle += np.pi / 2
        components = np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ], dtype=np.float32)
        projection = np.dot(cluster_points, components.T)
        # projection = np.tensordot(cluster_ptc, components, axes=(1, 0))
        min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
        min_y, max_y = projection[:, 1].min(), projection[:, 1].max()

    area = (max_x - min_x) * (max_y - min_y)

    rval = np.array([
        [max_x, min_y],
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
    ], dtype=np.float32)
    rval = np.dot(rval, components)
    
    return rval, angle, area
    

def variance_rectangle(cluster_ptc, delta=0.1):
    max_var = -float('inf')
    choose_angle = None
    for angle in np.arange(0, 90+delta, delta):
        angle = angle / 180. * np.pi
        components = np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ])
        projection = cluster_ptc @ components.T
        min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
        min_y, max_y = projection[:, 1].min(), projection[:, 1].max()
        Dx = np.vstack((projection[:, 0] - min_x,
                       max_x - projection[:, 0])).min(axis=0)
        Dy = np.vstack((projection[:, 1] - min_y,
                       max_y - projection[:, 1])).min(axis=0)
        Ex = Dx[Dx < Dy]
        Ey = Dy[Dy < Dx]
        var = 0
        if (Dx < Dy).sum() > 0:
            var += -np.var(Ex)
        if (Dy < Dx).sum() > 0:
            var += -np.var(Ey)
        # print(angle, var)
        if var > max_var:
            max_var = var
            choose_angle = angle
    # print(choose_angle, max_var)
    angle = choose_angle
    components = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])
    projection = cluster_ptc @ components.T
    min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
    min_y, max_y = projection[:, 1].min(), projection[:, 1].max()

    if (max_x - min_x) < (max_y - min_y):
        angle = choose_angle + np.pi / 2
        components = np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ])
        projection = cluster_ptc @ components.T
        min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
        min_y, max_y = projection[:, 1].min(), projection[:, 1].max()

    area = (max_x - min_x) * (max_y - min_y)

    rval = np.array([
        [max_x, min_y],
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
    ])
    rval = rval @ components
    return rval, angle, area


def PCA_rectangle(cluster_ptc):
    components = sklearn.decomposition.PCA(
        n_components=2).fit(cluster_ptc).components_
    on_component_ptc = cluster_ptc @ components.T
    min_x, max_x = on_component_ptc[:, 0].min(), on_component_ptc[:, 0].max()
    min_y, max_y = on_component_ptc[:, 1].min(), on_component_ptc[:, 1].max()
    area = (max_x - min_x) * (max_y - min_y)

    rval = np.array([
        [max_x, min_y],
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
    ])
    rval = rval @ components
    angle = np.arctan2(components[0, 1], components[0, 0])
    return rval, angle, area


def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.
    https://stackoverflow.com/questions/13542855/algorithm-to-find-the-minimum-area-rectangle-for-given-points-in-order-to-comput
    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    pi2 = np.pi/2.

    # get the convex hull for the points
    try:
        hull_points = points[spatial.ConvexHull(points).vertices]
    except:
        corners = np.ones((4, 2)) * np.mean(points[:, :2], axis=0)[:2]
        corners += np.array([[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05]])
        rz = 0
        return corners, rz, 0

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval, angles[best_idx], areas[best_idx]


def fit_plane(points, plane_distance_threshold=0.2, threshold=0.1, max_iteration=100):
    plane1 = pyrsc.Plane()
    plane2 = pyrsc.Plane()
    best_eq, best_inliers = plane1.fit(points[:, :3], 0.1, maxIteration=max_iteration)
    # refinement
    plane_model, inliers = plane2.fit(points[best_inliers][:, :3], threshold, maxIteration=max_iteration)
    plane_model = np.array(plane_model)
    if plane_model[2] < 0:
        plane_model *= -1

    angle = np.arccos((np.dot([0, 0, 1], plane_model[:3])) / (np.linalg.norm(plane_model[:3]) * np.linalg.norm([0, 0, 1])))
    
    return plane_model, angle


def transform_cluster_points_to_origin(points):
    # rotations for image coordinates
    rot1 = R.from_euler('z', np.pi / 2.)
    rot2 = R.from_euler('x', np.pi)
    # get view direction
    pts_ = points.copy()
    center_pos = np.median(pts_[..., :3], axis=0)
    angle = np.arctan2(center_pos[1], center_pos[0])
    rot3 = R.from_euler('z', -angle)
    # shift to origin
    pts_[..., :2] -= center_pos[:2]
    # rotate to view direction
    pts_ = rot3.apply(pts_)
    # shift one meter into x direction
    pts_[..., 0] -= 1
    # pts = pts[..., [2,1,0]]
    # transform to image coordinates
    pts_ = np.stack([pts_[:, 2], pts_[:, 1], pts_[:, 0]], axis=1)
    rot = np.eye(4)
    rot[:3, :3] = (rot2.as_matrix() @ rot1.as_matrix())
    pts_ = apply_transform(pts_, rot)
    
    return pts_


def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)
    Returns:
    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    # d_x0, d_x1, d_y0, d_y1 = x - x0.type_as(x), x1.type_as(x) - x, y - y0.type_as(y), y1.type_as(y) - y
    # wa = d_x1 * d_y1
    # wb = d_x1 * d_y0
    # wc = d_x0 * d_y1
    # wd = d_x0 * d_y0

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(
        torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans


def interpolate_from_bev_features(keypoints, bev_features, batch_size, bev_stride, voxel_size, voxel=False, voxel_range=None):
    if not voxel:
        x_idxs = (keypoints[:, :, 0] - voxel_range[0]) / voxel_size[0]
        y_idxs = (keypoints[:, :, 1] - voxel_range[1]) / voxel_size[1]
        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride
    else:
        x_idxs = keypoints[:, :, 2].type_as(bev_features)
        y_idxs = keypoints[:, :, 1].type_as(bev_features)

    point_bev_features_list = []
    for k in range(batch_size):
        cur_x_idxs = x_idxs[k]
        cur_y_idxs = y_idxs[k]
        cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
        point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
        point_bev_features_list.append(point_bev_features.unsqueeze(dim=0))

    point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (B, N, C0)
    return point_bev_features


def chamfer_distance(points_1, points_2, smallest_first=True, threshold=0.2):
    if len(points_1) > len(points_2) and smallest_first:
        p1 = torch.from_numpy(points_2[..., :3]).cuda().unsqueeze(dim=0)
        p2 = torch.from_numpy(points_1[..., :3]).cuda().unsqueeze(dim=0)
    else:
        p1 = torch.from_numpy(points_1[..., :3]).cuda().unsqueeze(dim=0)
        p2 = torch.from_numpy(points_2[..., :3]).cuda().unsqueeze(dim=0)
    
    x_nn = knn_points(p1, p2, K=1)
    idx_p2 = x_nn.idx.squeeze()
    y_nn = knn_points(p2.index_select(1, idx_p2), p1, K=1)
    
    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_x = cham_x[cham_x < threshold]
    cham_y = y_nn.dists[..., 0]  # (N, P2)
    cham_y = cham_y[cham_y < threshold]
    
    return (np.mean(cham_x.cpu().numpy()) + np.mean(cham_y.cpu().numpy())) / 2


def knn(points_source, points_target, K=1):
    sp = torch.from_numpy(points_source[..., :3]).cuda().unsqueeze(dim=0)
    tp = torch.from_numpy(points_target[..., :3]).cuda().unsqueeze(dim=0)
    x_nn = knn_points(sp, tp, K=K)
    dists = x_nn.dists.squeeze().cpu().numpy()
    indices = x_nn.idx.squeeze().cpu().numpy()
    
    return dists, indices

def knn_labels(points, label_points, labels, probabilities=None, dist_threshold=0.2, K=1):
    dists, indices = knn(points, label_points, K=K)
    point_labels = labels[indices]
    point_probabilities = probabilities[indices] if probabilities is not None else None
    if len(points) > 1:
        point_labels[dists > dist_threshold] = -1
    else:
        point_labels = -1 if dists > dist_threshold else point_labels
    return point_labels, point_probabilities


def points_in_boxes(points, boxes):    
    with torch.no_grad():
        box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                        torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                        torch.from_numpy(boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
                    ).long().squeeze(dim=0).detach().cpu().numpy()
    return box_idxs_of_pts


def bin_angles(angles, n_bins=45):
    """
    Accepts angles in the full range from 0 to 2*pi, but bins them only if they are
    effectively within 0 to pi/2 by normalizing to this range.
    
    Args:
    angles (list of float): List of angles to bin (in radians, can be from 0 to 2*pi).
    n_bins (int): Number of bins to divide the range from 0 to pi/2 into. Default is 10.
    
    Returns:
    list of int: List with the count of angles in each bin, considering normalized range.
    """
    # Define the target range from 0 to pi/2
    bin_edges = np.linspace(0, np.pi, n_bins + 1)
    
    # Initialize the bins count to zero
    bin_counts = [0] * n_bins
    angles_bin = [[] for i in range(n_bins)]
    
    # Count the angles in each bin after normalization
    for angle in angles:
        # Normalize the angle to the range [0, pi/2]
        normalized_angle = angle % (2 * np.pi)  # Wrap the angle within [0, 2*pi)
        if normalized_angle > np.pi:
            # Further normalization if angle is beyond pi/2
            normalized_angle %= (np.pi)
        
        # Find which bin the normalized angle falls into
        bin_index = np.digitize(normalized_angle, bin_edges, right=False) - 1
        
        # Increment the corresponding bin count
        if 0 <= bin_index < n_bins:
            bin_counts[bin_index] += 1
            angles_bin[bin_index].append(normalized_angle)
    
    return bin_counts, angles_bin[np.argmax(bin_counts)]

    # Example usage
    angles_list = [0.1, 0.5, 3.5, 5.0, 6.0, 1.0, 2.0, 4.0, 5.5, 6.1]
    bin_counts = bin_angles_restricted_range(angles_list)