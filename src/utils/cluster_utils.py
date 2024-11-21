import sys
import numpy as np

from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
from hydra.utils import instantiate
from src.utils.pointcloud_utils import poly_area_2d
from scipy import spatial


def init(cluster_cfg):
    return instantiate(cluster_cfg)

def filter_by_number_points(points, min_points=0, max_points=999999, **kwargs):
    return (points.shape[0] >= min_points) & (points.shape[0] <= max_points)

def filter_by_aspect_ratio(points, min_aspect_ratio, max_aspect_ratio, **kwargs):
    size = points.max(axis=0) - points.min(axis=0)
    # max: applies always
    max_valid = (np.max(size[:2]) / np.min(size[:2])) <= max_aspect_ratio
    # min: applies only if size is smaller than 1.0 --> should exclude small instances, e.g., pedestrains
    min_valid = ((np.max(size[:2]) / np.min(size[:2])) >= min_aspect_ratio) | ((size[0] < 1.0) | (size[1] < 1.0))
    return min_valid & max_valid

def filter_by_volume(points, min_volume, **kwargs):
    if len(points) < 3:
        return False
    height = points[..., 2].max(axis=0) - points[..., 2].min(axis=0)
    hull_points = points[spatial.ConvexHull(points[..., :2]).vertices]
    area = poly_area_2d(hull_points[..., :2])
    volume = area * height
    valid = volume >= min_volume
    if kwargs.get('max_volume', None) is not None:
        valid &= volume <= kwargs.get('max_volume')
    return valid

def filter_by_area(points, min_area, **kwargs):
    if len(points) < 3:
        return False
    # get the convex hull for the points
    hull_points = points[spatial.ConvexHull(points[..., :2]).vertices]
    area = poly_area_2d(hull_points[..., :2])
    valid = area >= min_area
    if kwargs.get('max_area', None) is not None:
        valid &= area <= kwargs.get('max_area')
    return valid

def filter_by_height(height, min_height, max_height, **kwargs):
    return (height >= min_height) & (height <= max_height)

def distance_to_plane(points, plane_model, directional=False):
    d = points @ plane_model[:3] + plane_model[3]
    if not directional:
        d = np.abs(d)
    d /= np.sqrt((plane_model[:3]**2).sum())
    return d

def filter_by_plane_distance(points, plane_model, max_min_height, min_max_height, **kwargs):
    distance_to_ground = distance_to_plane(points, plane_model, directional=True)
    return (distance_to_ground.min() <= max_min_height) & (distance_to_ground.max() >= min_max_height)

def filter_by_ephemeral_score(ephemeral_scores, percentile, min_percentile_pp_score, **kwargs):
    # small values indicate that the cluster is moving --> percentile > xx indicates static clusters
    return not (np.percentile(ephemeral_scores, percentile) > min_percentile_pp_score) 
    
def validate_cluster(points, filters, filters_active, **kwargs):
    and_valid = []
    and_required_valid = []
    or_valid = []
    for filter in filters:
        if getattr(sys.modules[__name__], filter['name'], False) and filter['name'] in filters_active:
            valid = getattr(sys.modules[__name__], filter['name'])(points=points, **filter['args'], **kwargs)
            if filter['args'].get('logic') == 'and':
                if filter['args'].get('required', False):
                    and_required_valid.append(valid)
                else:
                    and_valid.append(valid)
            elif filter['args'].get('logic') == 'or':
                or_valid.append(valid)
        elif filter['name'] not in filters_active:
            pass
        else:
            logger = kwargs.get('logger', None)
            warning_filter_not_found = f"Filter {filter['name']} not found!"
            if logger is not None:
                logger.warning(warning_filter_not_found)
            else:
                print(warning_filter_not_found)
    return (np.all(and_valid) or np.any(or_valid)) and np.all(and_required_valid)

def filter_clusters(points, labels, filters, filters_active, **kwargs):
    labels_ = labels.copy()
    filter_mask = []
    l_ids, _ = np.unique(labels, return_counts=True)
    for l_id in l_ids:
        if l_id != -1:
            is_valid = validate_cluster(points[labels == l_id, :3], filters, filters_active,  **kwargs)
            if not is_valid:
                labels_[labels_ == l_id] = -1
            filter_mask.append(is_valid)
    return labels_, filter_mask

def filter_detection(detection, filters, filters_active, **kwargs):
    detection.valid = validate_cluster(detection.cluster_points[..., :3], filters, filters_active,  **kwargs)