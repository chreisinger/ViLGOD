import torch
import numpy as np

from scipy import spatial
from scipy.optimize import linear_sum_assignment
from pcdet.ops.iou3d_nms import iou3d_nms_utils


def box_iou(boxes1, boxes2):
    try:
        boxes1_tensor = torch.tensor(boxes1[:, 0:7], dtype=torch.float).cuda()
        boxes2_tensor = torch.tensor(boxes2[:, 0:7], dtype=torch.float).cuda()
        iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(boxes1_tensor, boxes2_tensor)
        iou_matrix = iou_matrix.detach().cpu().numpy()
    except:
        boxes1_tensor = torch.tensor(boxes1[:, 0:7], dtype=torch.float).cuda()
        boxes2_tensor = torch.tensor(boxes2[:, 0:7], dtype=torch.float).cuda()
        iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(boxes1_tensor, boxes2_tensor)
        iou_matrix = iou_matrix.detach().cpu().numpy()
    return iou_matrix


def assign_detections_hungarian(detections, boxes, det_overlap_threshold=None, max_distance=None, weights=None, **kwargs):
    if len(detections) == 0 or len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    if det_overlap_threshold is not None:
        iou_matrix = box_iou(detections, boxes)
        cost_matrix = np.array(-iou_matrix)
    elif max_distance is not None:
        cost_matrix = spatial.distance.cdist(detections[:, 0:2], boxes[:, 0:2], 'euclidean')
        cost_matrix[cost_matrix > max_distance] = 1e7
    
    if weights is not None:
        cost_matrix = cost_matrix * weights
        
    matched_row_idx, matched_column_idx = linear_sum_assignment(cost_matrix)
    matched_indices = np.hstack((matched_row_idx[None].T, matched_column_idx[None].T))
    matched_indices = np.array(matched_indices)
    overlap = np.zeros(len(detections))
    
    if det_overlap_threshold is not None:
        overlap[matched_indices[..., 0]] = iou_matrix[matched_indices[..., 0], matched_indices[..., 1]]
        mask = overlap >= det_overlap_threshold
    elif max_distance is not None:
        overlap[matched_indices[..., 0]] = cost_matrix[matched_indices[..., 0], matched_indices[..., 1]]
        mask = overlap < max_distance
    
    # np.max(iou_matrix, axis=1) >= det_overlap_threshold

    return matched_indices, mask, overlap


def assign_detections_greedy(detections, boxes, det_overlap_threshold=None, max_distance=None, **kwargs):
    """ Greedy implementation adapted from
        https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking
    """

    if det_overlap_threshold is not None:
        raise NotImplementedError
    
    if len(detections) == 0 or len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    matched_indices = []

    cost_matrix = spatial.distance.cdist(detections[:, 0:2], boxes[:, 0:2], 'euclidean')
    # cost_matrix[cost_matrix > max_distance] = 1e7

    num_detections, num_tracks = cost_matrix.shape
    distance_1d = cost_matrix.reshape(-1)
    index_1d = np.argsort(distance_1d)
    index_2d = np.stack([index_1d // num_tracks, index_1d % num_tracks], axis=1)
    detection_id_matches_to_tracking_id = [-1] * num_detections
    tracking_id_matches_to_detection_id = [-1] * num_tracks
    for sort_i in range(index_2d.shape[0]):
        detection_id = int(index_2d[sort_i][0])
        tracking_id = int(index_2d[sort_i][1])
        if tracking_id_matches_to_detection_id[tracking_id] == -1 and \
                detection_id_matches_to_tracking_id[detection_id] == -1:
            tracking_id_matches_to_detection_id[tracking_id] = detection_id
            detection_id_matches_to_tracking_id[detection_id] = tracking_id
            matched_indices.append([detection_id, tracking_id])

    overlap = np.ones(len(detections)) * (max_distance + 1)
    mask = np.ones(len(detections), dtype=np.bool_)
    
    if len(matched_indices) > 0:
        matched_indices = np.array(matched_indices)
        overlap[matched_indices[..., 0]] = cost_matrix[matched_indices[..., 0], matched_indices[..., 1]]
        mask = overlap < max_distance
    else:
        matched_indices = np.empty((0, 2))

    return matched_indices, mask, overlap
