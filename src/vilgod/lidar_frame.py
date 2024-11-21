import numpy as np
import pandas as pd
import pickle as pickle

from time import time
from pathlib import Path
from easydict import EasyDict
from copy import deepcopy

from src.utils import common_utils, pointcloud_utils, eval_utils, cluster_utils
from src.dataclass.evaluation import ClusterResult
from src.dataclass.objects import BoundingBox3D, Detection


class LidarFrame():
    def __init__(self, name, fnr, points, annos, pose, ref_pose, cfg, logger) -> None:
        # frame infos
        self.fnr = fnr
        self.cfg = cfg
        self.pose = pose
        self.annos = annos
        self.logger = logger
        # unique frame id
        self.frame_id = name + '_' + str(self.fnr)
        self.transform_to_ref = np.linalg.inv(ref_pose) @ self.pose
        self.transform_to_ego = np.linalg.inv(self.pose) @ ref_pose
        # frame data
        self._points = points
        self._points_ref = None
        self._points_ref_wo_ground = None
        self._ground_point_indices = None
        self._entropy_scores = None
        self._entropy_indices = None
        self._ground_plane_model_ref = None
        
        self._detections: list[Detection] = []
        self._gt_detection_index_mapping = {}
        self._gt_cluster_mapping = {}
    
    @property
    def serialize(self):
        parameter_list = ['_detections', '_ground_point_indices', 
                          '_entropy_scores', '_entropy_indices', 
                          '_gt_cluster_mapping']
        frame_data = {}
        # serialize frame-level data
        for p in parameter_list:
            if hasattr(self, p):
                if p == '_detections' and self.__getattribute__(p) is not None:
                    if p not in frame_data:
                        frame_data[p] = []
                    for detection in self.__getattribute__(p):
                        detection_data = detection.serialize
                        if detection_data is not None:
                            frame_data[p].append(detection_data)
                elif self.__getattribute__(p) is not None:
                    frame_data[p] = self.__getattribute__(p)
        # return serialized frame data
        return frame_data
    
    @property
    def points(self):
        return self._points
    
    @property
    def points_ref(self):
        if self._points_ref is None: 
            self._points_ref = pointcloud_utils.apply_transform(self._points, self.transform_to_ref)
        return self._points_ref
    
    @property
    def points_wo_ground(self):
        return self._points[self.non_ground_mask] if self.non_ground_mask is not None else None
    
    @property
    def points_ref_wo_ground(self):
        if self._points_ref_wo_ground is None:
            self._points_ref_wo_ground = self.points_ref[self.non_ground_mask] if self.non_ground_mask is not None else None
        return self._points_ref_wo_ground
    
    @property
    def ground_mask(self):
        ground_mask = None
        if self._ground_point_indices is not None:
            ground_mask = np.zeros_like(self.points[..., 0], dtype=np.bool_)
            ground_mask[self._ground_point_indices] = True
        return ground_mask
    
    @property
    def non_ground_mask(self):
        non_ground_mask = None
        if self.ground_mask is not None:
            non_ground_mask = ~self.ground_mask
        return non_ground_mask
    
    @property
    def ground_plane_model_ref(self):
        if self._ground_plane_model_ref is None:
            ground_plane_model_ref = None
            if self.ground_mask is not None:
                ground_plane_model_ref = pointcloud_utils.fit_plane(self.points_ref[self.ground_mask])[0]
            else:
                ground_plane_model_ref = pointcloud_utils.fit_plane(self.points_ref)[0]
                self.logger.warn(f"No ground mask found for frame {self.frame_id}, estimation on whole point cloud.")
            self._ground_plane_model_ref = ground_plane_model_ref
        else:
            ground_plane_model_ref = self._ground_plane_model_ref
        
        return ground_plane_model_ref
    
    @property
    def entropy_scores(self):
        # usining entropy scores only below 0.9 saves about 75% of the memory
        entropy_scores = None
        if self._entropy_scores is not None:
            entropy_scores = np.ones_like(self.points_ref_wo_ground[..., 0], dtype=np.float32)
            entropy_scores[self._entropy_indices] = self._entropy_scores
        return entropy_scores
    
    @property
    def detections(self):
        return self._detections

    def sync_lidar_frame(self, data):
        detections = None
        for k, v in data.items():
            if hasattr(self, k):
                if k == '_detections':
                    detections = v
                else:
                    self.__setattr__(k, v)
                    
        if detections is not None:
            self.sync_detections(detections)
        
    def sync_detections(self, detections):
        for detection in detections:
            cluster_points = self.points_ref_wo_ground[detection['cluster_points_index']]
            cluster_points_entropy = None
            if self.entropy_scores is not None:
                cluster_points_entropy = self.entropy_scores[detection['cluster_points_index']]
            new_detection = Detection(cluster_id=detection['cluster_id'], 
                                      cluster_points=cluster_points, 
                                      cluster_points_index=detection['cluster_points_index'],
                                      cluster_points_entropy=cluster_points_entropy)
            new_detection.sync_detection(detection)
            self._detections.append(new_detection)
    
    def clear_detections(self):
        self._detections = []
        self._gt_detection_index_mapping = {}
        self._gt_cluster_mapping = {}
    
    def generate_detections(self, indices, probabilities=None, proposals=None, names=None, assign_gt=False, entropy_scores=None):
        # generate detections out of proposals
        if indices is None and proposals is not None:
            if len(proposals) == 0:
                return
            proposals_ref = pointcloud_utils.apply_transform(proposals, self.transform_to_ref, box=True)
            indices = pointcloud_utils.points_in_boxes(self.points_ref, proposals_ref)
            cluster_ids = np.sort(pd.unique(indices[indices != -1]))
        # mask cluster points with low probability
        if probabilities is not None:
            probability_mask = probabilities < self.cfg.preprocessor.clustering.propability_threshold
            indices[probability_mask] = -1
        # filter all cluster ids
        cluster_ids = np.sort(pd.unique(indices[indices != -1]))
        # assign clusters to gt detections
        if assign_gt and len(self.detections) > 0:
            self._gt_cluster_mapping = {}
            # find all gt indices for points not on ground
            gt_indices = np.ones(len(self.points), dtype=np.int32) * -1
            for d in self.detections:
                gt_indices[d.cluster_points_index] = d.cluster_id
            # remove indices for ground points --> valid for clusters, not for proposals (detections are done on whole point cloud)
            if proposals is None:
                gt_indices = gt_indices[self.non_ground_mask]
            # assign clusters if point overlap with gt, otherwise create new detection
            new_detections = []
            new_cluster_id = np.max(gt_indices) + 1
            for cid in cluster_ids:
                cluster_points_index = np.where(indices == cid)[0]
                if np.count_nonzero(gt_indices[cluster_points_index] + 1) > 0:
                    idx, count = np.unique(gt_indices[cluster_points_index], return_counts=True)
                    # find max overlap gt
                    idx_max = idx[np.argmax(count)]
                    if idx_max == -1:
                        idx_max = idx[np.argmax(count[1:]) + 1]
                    gt_detection = self._detections[self._gt_detection_index_mapping[idx_max]]
                    assert gt_detection.cluster_id == idx_max, "Cluster id does not match with position"
                    # if gt detection has already been assigned, create clone (multiple clusters per gt)
                    if gt_detection.gt_assigned:
                        gt_detection = deepcopy(gt_detection)
                        gt_detection.cluster_id = new_cluster_id
                        new_cluster_id += 1
                        new_detections.append(gt_detection)
                    # mark detection as assigned
                    gt_detection.gt_assigned = True
                    gt_detection.gt = False
                    # update gt detection
                    gt_detection.cluster_points_index_fp = cluster_points_index[gt_indices[cluster_points_index] != idx_max]
                    gt_detection.cluster_points_index_fn = np.setdiff1d(np.where(gt_indices == idx_max)[0], cluster_points_index)
                    gt_detection.cluster_points_index = cluster_points_index
                    # for proposals, use whole point cloud
                    if proposals is None:
                        gt_detection.cluster_points = self.points_ref_wo_ground[cluster_points_index]
                    else:
                        gt_detection.cluster_points = self.points_ref[cluster_points_index]
                        gt_detection.update_bounding_box(proposals_ref[..., :7][cid])
                    if names is not None:
                        gt_detection.add_object_entry('object_class', 'proposal', names[cid])
                    gt_detection.cluster_center = gt_detection.cluster_points.mean(axis=0)
                    # add cluster id to gt mapping
                    if gt_detection.gt_id not in self._gt_cluster_mapping:
                        self._gt_cluster_mapping[gt_detection.gt_id] = []
                    self._gt_cluster_mapping[gt_detection.gt_id].append(gt_detection.cluster_id)
                else:
                    cluster_points = self.points_ref_wo_ground[cluster_points_index] if proposals is None else self.points_ref[cluster_points_index]
                    new_detection = Detection(cluster_id=new_cluster_id,
                                            cluster_points=cluster_points, 
                                            cluster_points_index=cluster_points_index)
                    if proposals is not None:
                        new_detection.update_bounding_box(proposals_ref[..., :7][cid])
                    if names is not None:
                        new_detection.add_object_entry('object_class', 'proposal', names[cid])
                    new_detections.append(new_detection)
                    new_cluster_id += 1
            # add non-gt clusters to detections
            self._detections.extend(new_detections)
        else:
            # add detection from cluster without gt assignment
            for cid in cluster_ids:
                cluster_points_index = np.where(indices == cid)[0]
                cluster_points = self.points_ref_wo_ground[cluster_points_index] if proposals is None else self.points_ref[cluster_points_index]
                new_detection = Detection(cluster_id=cid,
                                          cluster_points=cluster_points, 
                                          cluster_points_index=cluster_points_index)
                if entropy_scores is not None:
                    cluster_cfg = self.cfg.preprocessor.clustering.entropy_score_filter
                    moving = cluster_utils.filter_by_ephemeral_score(entropy_scores[cluster_points_index], 
                                                                     percentile=cluster_cfg.percentile, 
                                                                     min_percentile_pp_score=cluster_cfg.min_percentile_pp_score)
                    new_detection.static = not moving
                if proposals is not None:
                    new_detection.update_bounding_box(proposals_ref[..., :7][cid])
                if names is not None:
                    new_detection.add_object_entry('object_class', 'proposal', names[cid])
                self._detections.append(new_detection)

    def remove_invalid_detections(self):
        self._detections = [d for d in self.detections if d.is_valid]

    def update_ground_indices(self, indices):
        self._ground_point_indices = indices

    def update_entropy_scores(self, scores, indices):
        self._entropy_scores = scores
        self._entropy_indices = indices
            
    def update_object_classes(self, class_names, class_names_detailed, class_scores, 
                              cluster_update_list, key='class_key', aggregation='voting', depth_images=None):
        idx = 0
        for d_idx, detection in enumerate(self.detections):
            if cluster_update_list[d_idx]:
                detection.add_object_entry('object_class_predictions', key, class_names[idx])
                detection.add_object_entry('object_class_predictions_detailed', key, class_names_detailed[idx])
                detection.add_object_entry('object_class_predictions_score', key, class_scores[idx])
                
                if aggregation == 'voting':
                    names, counts = np.unique(class_names[idx], return_counts=True)
                    if sum((counts[np.argmax(counts)]) == counts) > 1:
                        name_max = None
                        max_score = 0
                        for name in names:
                            score = np.mean(class_scores[idx][class_names[idx]==name])
                            if score > max_score:
                                max_score = score
                                name_max = name
                        name = name_max
                        score = max_score
                    else:
                        name = names[np.argmax(counts)]
                        score = np.mean(class_scores[idx][class_names[idx]==name])
                    detection.add_object_entry('object_class', key, name)
                    detection.add_object_entry('object_class_score', key, score)
                else:
                    raise NotImplementedError
                    
                if depth_images is not None:
                    detection.depth_image = depth_images[idx]
                idx += 1