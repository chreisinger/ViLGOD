import torch
import pickle
import scipy
import kornia

import numpy as np
import cv2 as cv
from pathlib import Path
from tqdm import tqdm
from time import time
from PIL import Image
from copy import deepcopy
from hdbscan import HDBSCAN
from sklearn.metrics import pairwise
from functools import partial

from pcdet.ops.iou3d_nms import iou3d_nms_utils

from src.utils import common_utils, pointcloud_utils, cluster_utils
from src.utils.mv_utils import RealisticProjection
from src.vilgod.lidar_frame import LidarFrame
from src.vilgod.tracker import Tracker

class ZeroShotDetector():
    def __init__(self, dataset, name, cfg, logger, cluster_model,
                 clip_model):
        self.cfg = cfg
        self.name = name
        self.dataset = dataset
        
        self.lenght = dataset.sequence_length
        self.logger = logger
        self.lidar_frame_list: list[LidarFrame] = []
        self.progress_bar = tqdm(total=self.lenght, desc=f"Processing sequence: {self.name}")
        self.tracker = None
        self.projection_model = RealisticProjection(cfg.preprocessor.lidar_image_projection)
        self.cluster_model = cluster_model
        self.clip_model = clip_model
        
        # init paths
        self.sequence_data_dir_path = Path(self.cfg.paths.sequence_data)
        # load all lidar frames
        self.init_lidar_frames()
        # synchronize already processed data
        try:
            self.sync_lidar_frames(mode='load')
        except:
            pass
        self.logger.info(f"Loaded {len(self.lidar_frame_list)} lidar frames")
        self.detection_3d_result_list = []
        self.cls_key = None
    
    def __del__(self):
        del self.detection_model
        del self.clip_model
        del self.projection_model
            
    def process(self):
        self.logger.info(f"Processing sequence: {self.name}")
        
        # Active task list-ordered execution
        available_tasks = [task.name for task in self.cfg.pipeline]
        for task_name in self.cfg.pipeline_active:
            if task_name in available_tasks:
                task_idx = available_tasks.index(task_name)
                getattr(self, task_name)(**self.cfg.pipeline[task_idx]['args'])
            else:
                self.logger.warn(f'{task_name} NOT FOUND!!!')
        self.logger.info(f"Finished processing sequence: {self.name}")
            
    def init_lidar_frames(self):
        # check if output dir exists -> create otherwise
        if common_utils.check_and_create_dir(self.sequence_data_dir_path):
            self.logger.info(f"Created directory: {self.sequence_data_dir_path}")
        self.reset_progress_bar('Load lidar frames')
        # prepare annos stats
        annos_stats = {}
        for class_name in self.dataset.class_names:
            annos_stats[class_name] = 0
            annos_stats[f'{class_name}_moving'] = 0
        # inti lidar frames
        for fnr in range(self.dataset.sequence_length):
            annos = self.dataset.get_annos(fnr)
            self.lidar_frame_list.append(
                LidarFrame(
                    self.name, fnr, 
                    self.dataset.get_lidar_points(fnr), 
                    annos, 
                    self.dataset.sequence_infos[fnr]['pose'],
                    self.dataset.sequence_infos[0]['pose'], 
                    self.cfg, self.logger
                )
            )
            # fill annos stats
            for name, moving in zip(annos['gt_names'], annos['moving']):
                annos_stats[name] += 1
                if moving:
                    annos_stats[f'{name}_moving'] += 1
                    
            self.progress_bar.update(1)
        # log annos stats    
        for k, v in annos_stats.items():
            self.logger.info(f"{k}: {v}")
    
    def sync_lidar_frames(self, mode='save'):
        self.reset_progress_bar('Synchronize')
        file_name = f"{self.name}{self.cfg.postfix.sequence_data}"
        sequence_data_file_path = self.sequence_data_dir_path / file_name
        
        if mode == 'save':
            sequence_data = [lidar_frame.serialize for lidar_frame in self.lidar_frame_list]
            with open(sequence_data_file_path, 'wb') as fp:
                pickle.dump(sequence_data, fp)
            self.progress_bar.update(self.lenght)
        elif mode == 'load':
            if sequence_data_file_path.exists():
                with open(sequence_data_file_path, 'rb') as fp:
                    sequence_data = pickle.load(fp)
                for fnr, frame_data in enumerate(sequence_data):
                    self.lidar_frame_list[fnr].sync_lidar_frame(frame_data)
                    self.progress_bar.update(1)
        else:
            raise NotImplementedError(f"Mode {mode} not implemented!")
    
    def reset_progress_bar(self, description):
        self.progress_bar.reset(total=self.lenght)
        self.progress_bar.set_description(f"[{self.name}] {description}")
    
    def mask_ground_points(self, min_range, z_offset, **kwargs):
        # https://github.com/url-kaist/patchwork-plusplus
        # Lee et al.,
        # Patchwork++: Fast and robust ground segmentation solving 
        #              partial under-segmentation using 3D point cloud. 
        # IROS 2022.
        import pypatchworkpp
        # init patchwork++
        params = pypatchworkpp.Parameters()
        params.verbose = False
        params.min_range = min_range
        patchwork_pp = pypatchworkpp.patchworkpp(params)
        self.reset_progress_bar('Mask ground points')
        update = False
        for lidar_frame in self.lidar_frame_list:
            if lidar_frame.ground_mask is None:
                ground_point_indices = pointcloud_utils.mask_ground_points_patchwork_pp(lidar_frame.points, patchwork_pp, z_offset)
                lidar_frame.update_ground_indices(ground_point_indices)
                update = True
            self.progress_bar.update(1)
            
        if update:
            self.sync_lidar_frames()
            
    def calculate_entropy_scores(self, n_neighbouring_frames, **kwargs):
        self.reset_progress_bar('Calculate entropy scores')
        seek = -1
        frame_buffer = []
        
        include_ground_points = kwargs.get('include_ground_points', False)
        
        # skip if entropy scores are already calculated
        if np.count_nonzero([lidar_frame._entropy_scores is None for lidar_frame in self.lidar_frame_list]) == 0:
            self.progress_bar.update(self.lenght)
            return
        
        for fnr in range(self.lenght):
            if len(frame_buffer) == 0:
                for n_idx in range(n_neighbouring_frames):
                    if include_ground_points:
                        points = self.lidar_frame_list[n_idx].points_ref[..., :3]
                    else:
                        points = self.lidar_frame_list[n_idx].points_ref_wo_ground[..., :3]
                        
                    frame_buffer.append(torch.from_numpy(points).cuda())
                    
            if fnr > 0 and fnr <= (self.lenght - n_neighbouring_frames):
                if include_ground_points:
                    points = self.lidar_frame_list[fnr + n_neighbouring_frames - 1].points_ref[..., :3]
                else:
                    points = self.lidar_frame_list[fnr + n_neighbouring_frames - 1].points_ref_wo_ground[..., :3]
                frame_buffer.append(torch.from_numpy(points).cuda())
                drop_frame = frame_buffer.pop(0)
                drop_frame = drop_frame.detach().cpu()
            else:
                seek += 1
                
            if self.lidar_frame_list[fnr].entropy_scores is None or kwargs.get('force', False):
                entropy_scores = pointcloud_utils.calculate_entropy_scores(frame_buffer, seek, **kwargs)
                # reduce entropy scores (below 0.9); saves about 75% of the memory without loosing relevant information
                self.lidar_frame_list[fnr].update_entropy_scores(entropy_scores[entropy_scores < 0.9],
                                                                 np.where(entropy_scores < 0.9)[0])
            self.progress_bar.update(1)
            
        del frame_buffer
        torch.cuda.empty_cache()
        self.sync_lidar_frames()

    def spatial_clustering(self, **kwargs):
        self.reset_progress_bar('Spatial clustering')
        updated = False
        for lidar_frame in self.lidar_frame_list:
            condition_zero = np.count_nonzero(np.array([len(lf.detections) for lf in self.lidar_frame_list]) == 0) > 0
            condition_only_gt = np.count_nonzero([not d.gt for d in lidar_frame.detections]) == 0
            condition_force = kwargs.get('force', False)
            condition = condition_zero or condition_only_gt or condition_force
            if lidar_frame.points_ref_wo_ground is not None and condition:
                updated = True
                n_frames = kwargs.get('n_frames', 1)
                if n_frames > 1:
                    point_list = []
                    range_ = list(range(min(lidar_frame.fnr, len(self.lidar_frame_list) - n_frames), 
                                    min(lidar_frame.fnr + n_frames, len(self.lidar_frame_list))))
                    # spatial clustering with multiple frames
                    for f_idx_rel, f_idx in enumerate(range_):
                        len_ = len(self.lidar_frame_list[f_idx].points_ref_wo_ground)
                        points = self.lidar_frame_list[f_idx].points_ref_wo_ground[..., :3]
                        points_mask = np.ones(len_, dtype=bool)
                        if True: # lidar_frame.fnr != f_idx:
                            counts = pointcloud_utils.count_neighbors_inter_frame(points, 0.2)
                            entropy_mask = self.lidar_frame_list[f_idx].entropy_scores < 0.6 # moving points
                            moving_points = self.lidar_frame_list[f_idx].points_ref_wo_ground[entropy_mask]
                            dists = pointcloud_utils.knn(moving_points, moving_points, K=4)[0][..., 1:]
                            dists_mask = np.sum(dists < 0.1, axis=1) > 1
                            points_indices = np.random.choice(len_, int((len_ / len(range_))), replace=False)
                            points_mask = np.zeros(len_, dtype=bool)
                            points_mask[points_indices] = True
                            points_mask[counts < 2] = False
                            points_mask[entropy_mask] = False
                            points_mask[entropy_mask] |= dists_mask
                        cluster_input = np.concatenate([points[points_mask], 
                                                        self.lidar_frame_list[f_idx].entropy_scores[points_mask, None],
                                                        np.ones((len(points[points_mask]), 1)) * (f_idx_rel * 0.1)], axis=1)
                        point_list.append(cluster_input)
                        
                    points_seq = np.concatenate(point_list, dtype=np.float32)

                    cluster_info = self.cluster_model.fit(points_seq)
                    labels, probabilities = pointcloud_utils.knn_labels(
                        lidar_frame.points_ref_wo_ground, 
                        points_seq, 
                        cluster_info.labels_, 
                        cluster_info.probabilities_
                    )
                    
                # std spatial clustering
                else:
                    points_ref_wo_ground = lidar_frame.points_ref_wo_ground[..., :3]
                    # cluster_input = np.concatenate([points_ref_wo_ground, lidar_frame.entropy_scores[..., None]], axis=1)
                    cluster_info = self.cluster_model.fit(points_ref_wo_ground)    
                    labels = cluster_info.labels_
                    probabilities = cluster_info.probabilities_       
                
                if condition_force and not condition_zero and not condition_only_gt:
                    lidar_frame.clear_detections()
                    
                lidar_frame.generate_detections(labels, probabilities, assign_gt=False, entropy_scores=lidar_frame.entropy_scores)
                
            self.progress_bar.update()
        if updated:
            self.sync_lidar_frames()

    def filter_detections(self, **kwargs):
        self.logger.info('Check filter detections required')
        # check if filtered objects within sequence
        filtered_detections = False
        for lidar_frame in self.lidar_frame_list:
            for det in lidar_frame.detections:
                if not det.valid:
                    if kwargs.get('force', False):
                        det.valid = True
                    else:
                        filtered_detections = True
                        break
            if filtered_detections:
                break
            
        # run filtering if no filtered detection can be found or action is forced
        if not filtered_detections:
            self.reset_progress_bar('Filter detections')
            filters = []
            # assemble filters
            for filter in self.cfg.preprocessor.clustering.filters:
                if filter['name'] in self.cfg.preprocessor.clustering.filters_active:
                    if getattr(cluster_utils, filter['name'], False):
                        filter_ = partial(getattr(cluster_utils, filter['name']), **filter['args'])
                        # list of filters: [filter, filter_name, logic, required]
                        filters.append([filter_, filter['name'], filter['args'].get('logic'), filter['args'].get('required', False)])
            
            # filter all detections       
            for lidar_frame in self.lidar_frame_list:
                for det in lidar_frame.detections:
                    det.filter(filters, plane_model=lidar_frame.ground_plane_model_ref)

                self.progress_bar.update()
            self.sync_lidar_frames()
            
        else:
            self.logger.info('Filtered detections found. No filtering required.')
    
    def track_clusters(self, **kwargs):
        self.logger.info('Check track clusters required')
        # check if any detection is assigned to a track
        tracked_detections = False
        for lidar_frame in self.lidar_frame_list:
            for det in lidar_frame.detections:
                if det.tid != -1:
                    if kwargs.get('force', False):
                        det.tid = -1
                    else:
                        tracked_detections = True
                        break
            if tracked_detections:
                break
    
        # run tracking if no tracked detection can be found or action is forced
        if not tracked_detections:
            self.reset_progress_bar('Track clusters')
            valid_only = kwargs.get('valid_only', False)
            self.tracker = Tracker(self.name, self.cfg.preprocessor.tracking.cluster)
            for lidar_frame in self.lidar_frame_list:
                detections = [d for d in lidar_frame.detections if d.valid] if valid_only else lidar_frame.detections
                self.tracker.next(detections, lidar_frame.fnr)
                self.progress_bar.update(1)
                
            self.tracker.finish()
            self.sync_lidar_frames()
        else:
            self.logger.info('Detections are already tracked')
        
    def classification(self, image_size, aggregation='voting', **kwargs):
        self.reset_progress_bar('Classification')
        image_out_path = Path('../output_images') / self.name
        common_utils.check_and_create_dir(image_out_path)
        # reset results -> e.g. ground truth results
        self.result_list = []
        length_total = 0
        classify_gt = False # TODO: reimplement
        valid_only = kwargs.get('valid_only', False)
        missing_only = kwargs.get('missing_only', False)
        force = kwargs.get('force', False)
        key_ = kwargs.get('key', 'clip')
         
        classified_detections = False
        missing_detections = False
        
        if not force:
            for lidar_frame in self.lidar_frame_list:
                for det in lidar_frame.detections:
                    if det.object_class is not None and key_ in det.object_class:
                        classified_detections = True
                        if not missing_detections:
                            break
                    elif (det.object_class is None or key_ not in det.object_class) and missing_only:
                        missing_detections = True
                        break
                if classified_detections or missing_detections:
                    break
        else:
            for lidar_frame in self.lidar_frame_list:
                for det in lidar_frame.detections:
                    if det.object_class is not None and key_ in det.object_class:
                        del det.object_class[key_]
        
        if not classified_detections or missing_detections:
            # evaluate for objects in frames
            for lidar_frame in self.lidar_frame_list:
                depth_image_list = []
                cluster_update_list = []                
                detections = []
                
                if valid_only:
                    for d in lidar_frame.detections:
                        if missing_detections:
                            if d.valid and (d.object_class is None or key_ not in d.object_class):
                                detections.append(d)
                                cluster_update_list.append(True)

                            else:
                                cluster_update_list.append(False)
                        else:
                            if d.valid:
                                detections.append(d)
                                cluster_update_list.append(True)
                            else:
                                cluster_update_list.append(False)
                else:
                    detections = lidar_frame.detections
                    cluster_update_list = [True] * len(detections)
                        
                for d_idx, det in enumerate(detections):
                    if (det.gt and classify_gt) or (not det.gt and not classified_detections):
                        cluster_points = det.cluster_points[..., :3]
                        cluster_points_ego = pointcloud_utils.apply_transform(cluster_points, lidar_frame.transform_to_ego)
                        cluster_points_transformed = pointcloud_utils.transform_cluster_points_to_origin(cluster_points_ego)
                        cluster_points_transformed_t = torch.from_numpy(cluster_points_transformed).float().cuda().unsqueeze(0)
                        depth_image = self.projection_model.get_img(cluster_points_transformed_t)
                        depth_image_list.append(depth_image)
                        cluster_update_list[d_idx] &= True
                    else:
                        cluster_update_list[d_idx] &= False
                        
                length = len(depth_image_list)
                length_total += length
                
                if length > 0:
                    depth_images = torch.cat(depth_image_list, dim=0).cuda()
                    depth_images = torch.nn.functional.interpolate(depth_images, size=(image_size, image_size), 
                                                                mode='bilinear', align_corners=True)
                    depth_images = depth_images.permute(0, 3, 2, 1).detach().cpu().numpy()
                    input_image_list = [Image.fromarray(np.uint8(img * 255)) for img in depth_images]
                    cls_result_list_detailed, score_result_list = self.clip_model.predict_clip_labels(input_image_list)
                    
                    cls_result_list = [self.cfg.preprocessor.clip.class_mapping[cls] for cls in cls_result_list_detailed]
                    cls_result_list = np.stack(cls_result_list).reshape((length, -1))
                    cls_result_list_detailed = np.stack(cls_result_list_detailed).reshape(length, -1)
                    score_result_list = np.stack(score_result_list).reshape(length, -1)
                    lidar_frame.update_object_classes(cls_result_list, cls_result_list_detailed, score_result_list, cluster_update_list, key=key_,
                                                        aggregation=aggregation, depth_images=input_image_list[::cls_result_list.shape[1]])
                    
                self.progress_bar.update(1)
            self.sync_lidar_frames()

    def fit_bounding_boxes_simple(self, method, **kwargs):
        box_fitted = False
        force = kwargs.get('force', False)
        if not force:
            for lidar_frame in self.lidar_frame_list:
                for det in lidar_frame.detections:
                    if det.bounding_box is not None:
                        box_fitted = True
                        break
                if box_fitted:
                    break
        else:
            for lidar_frame in self.lidar_frame_list:
                for det in lidar_frame.detections:
                    det._bounding_box = None
        
        if not box_fitted:
            self.reset_progress_bar('Fit bounding boxes')
            valid_only = kwargs.get('valid_only', False)
            fg_only = kwargs.get('fg_only', False)
            classification_key = kwargs.get('classification_key', None)
            # if no tracking involved, just fit bounding boxes for each detection in each frame
            if self.tracker is None or len(self.tracker.tracks_valid) == 0:
                for lidar_frame in self.lidar_frame_list:
                    if len(lidar_frame.detections) > 0:
                        for detection in lidar_frame.detections:
                            if (valid_only and detection.valid) or (not valid_only):
                                if (fg_only and classification_key is not None and classification_key in detection.object_class and detection.object_class[classification_key] in self.dataset.class_names) or (not fg_only):
                                    cluster_points = detection.cluster_points
                                    corners, rz, area = getattr(pointcloud_utils, method['name'])(cluster_points[:, :2], **method['args'])
                                    l = np.linalg.norm(corners[0] - corners[1])
                                    w = np.linalg.norm(corners[0] - corners[-1])
                                    c = (corners[0] + corners[2]) / 2
                                    if w > l:
                                        l, w = w, l
                                        rz += np.pi / 2
                                    # assumed height
                                    height = cluster_points[:, 2].max() - cluster_points[:, 2].min()
                                    box = np.array([c[0], c[1], cluster_points[:, 2].min() + height / 2, l, w, height + 0.3, rz])
                                    detection.update_bounding_box(box)
                    self.progress_bar.update(1)
            # if tracks available, handle static and moving objects differently
            else:
                for track in self.tracker.tracks_valid:
                    possibly_moving = False
                    for det in track.detections:
                        if not det.static:
                            possibly_moving = True
                            break
                    
                    # static boxes -> simple fit
                    if not possibly_moving:
                        for detection in track.detections:
                            cluster_points = detection.cluster_points
                            
                            corners, rz, area = getattr(pointcloud_utils, method['name'])(cluster_points[:, :2], **method['args'])
                                
                            l = np.linalg.norm(corners[0] - corners[1])
                            w = np.linalg.norm(corners[0] - corners[-1])
                            c = (corners[0] + corners[2]) / 2
                            if w > l:
                                l, w = w, l
                                rz += np.pi / 2
                            # assumed height
                            height = cluster_points[:, 2].max() - cluster_points[:, 2].min()
                            box = np.array([c[0], c[1], cluster_points[:, 2].min() + height / 2, l, w, height + 0.3, rz])
                            detection.update_bounding_box(box)
                    # possibly moving boxes -> fit with motion infomation
                    else:
                        def calc_motion_vectors(cluster_points):
                            from src.utils.common_utils import angle_between_vectors
                            centers_xy = []
                            center_indices = []
                            
                            for p_idx, points in enumerate(cluster_points):
                                if points.shape[0] > 0:
                                    centers_xy.append(np.median(points[..., :2], axis=0))
                                    center_indices.append(p_idx)
                            centers_xy = np.array(centers_xy)

                            # calculate directiion vector for each cluster
                            motion_vectors = []
                            motion_vectors_index = []
                            vector_far = None
                            for c_idx, centers in enumerate(centers_xy):
                                c_idx_far = min(c_idx + 10 - 1, len(centers_xy) - 1)
                                # far vector gives intention on where the cluster is moving
                                vector_far_ = np.array([centers_xy[c_idx_far, 0] - centers[0], 
                                                        centers_xy[c_idx_far, 1] - centers[1]])
                                # find far vector with cluster that is at least 0.5 meters away
                                if np.linalg.norm(vector_far_) < 0.5 and vector_far is None:
                                    idx_counter = 1
                                    # this should handle the case for a cluster slowly or static in the beginning
                                    while np.linalg.norm(vector_far_) < 0.5 and (c_idx_far + idx_counter) < len(centers_xy):
                                        vector_far_ = np.array([
                                            centers_xy[c_idx_far + idx_counter, 0] - centers[0], 
                                            centers_xy[c_idx_far + idx_counter, 1] - centers[1]])
                                        idx_counter += 1
                                    if np.linalg.norm(vector_far_) >= 0.5:
                                        vector_far = vector_far_
                                elif np.linalg.norm(vector_far_) < 0.5:
                                    # keep last far vector
                                    # this should handle the case for a cluster slowly or static in the middle / end
                                    pass # vector_far = vector_far
                                else:
                                    # vector_var_ is fine
                                    vector_far = vector_far_
                                
                                # in case there is a vector that points at least 0.5 meters away
                                if vector_far is not None:
                                    vectors = []
                                    mean_vector_norm = 0
                                    # find all vectors that point in a similar direction as vector_far
                                    for i in range(c_idx + 1, c_idx_far):
                                        vector_next = np.array([centers_xy[i, 0] - centers[0], 
                                                                centers_xy[i, 1] - centers[1]])
                                        if angle_between_vectors(vector_far, vector_next) < 60 and np.linalg.norm(vector_next) > 0.3:
                                            vectors.append(vector_next * (0.95 ** (i + 1)))
                                            mean_vector_norm += (0.9 ** (i + 1))
                                    
                                    # mean all similar vectors
                                    if len(vectors) > 0:
                                        motion_vectors_valid = True
                                        mean_vector = np.mean(vectors, axis=0) / mean_vector_norm
                                        if len(motion_vectors) > 0:
                                            # if directions already exist, interpolate between last direction and current direction
                                            mean_vector = mean_vector * 0.5 + motion_vectors[-1] * 0.5
                                            motion_vectors.append(mean_vector)
                                            motion_vectors_index.append(center_indices[c_idx])
                                        else:
                                            motion_vectors.append(mean_vector)
                                            motion_vectors_index.append(center_indices[c_idx])
                                    elif len(motion_vectors) > 0:
                                        # no current vector fits, propagate last direction
                                        motion_vectors.append(motion_vectors[-1])
                                        motion_vectors_index.append(center_indices[c_idx])
                                    elif vector_far is not None:
                                        # no current vector fits, and no last direction exists, propagate far vector
                                        motion_vectors.append(vector_far)
                                        motion_vectors_index.append(center_indices[c_idx])
                                    else:
                                        # no calculation possible
                                        return [], []
                                else:
                                    return [], []
                                
                            return motion_vectors, motion_vectors_index
                                    
                        cluster_points = [d.cluster_points for d in track.detections]
                        motion_vectors, motion_vectors_index = calc_motion_vectors(cluster_points)
                        from scipy.spatial.transform import Rotation as R
                        boxes = []
                        corner_list = []
                        # fit bounding box to each cluster base on direction vector
                        for c_idx, direction in enumerate(motion_vectors):
                            angle = np.arctan2(direction[1], direction[0])
                            rot_mat = R.from_euler('z', angle, degrees=False).as_matrix()
                            center =  np.median(cluster_points[c_idx][..., :3], axis=0)
                            # project points into center and align with x axis
                            pts = cluster_points[c_idx][..., :3] - center
                            projection = np.dot(pts, rot_mat)
                            # calculate min and max x and y values
                            min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
                            min_y, max_y = projection[:, 1].min(), projection[:, 1].max()
                            # represent box as 4 corners
                            rval = np.array([
                                [max_x, min_y],
                                [min_x, min_y],
                                [min_x, max_y],
                                [max_x, max_y],
                            ], dtype=np.float32)
                            # backproject corners into original coordinate system
                            corners = np.dot(rval, rot_mat[:2, :2].T)
                            corners += center[:2]
                            # calculate width, length and center of box
                            w = np.linalg.norm(corners[0] - corners[1])
                            l = np.linalg.norm(corners[0] - corners[-1])
                            c = (corners[0] + corners[2]) / 2
                            corner_list.append(corners)
                            # assumed height
                            height = cluster_points[c_idx][:, 2].max() - cluster_points[c_idx][:, 2].min()
                            box = np.array([c[0], c[1], cluster_points[c_idx][:, 2].min() + height / 2, w, l, height, angle])
                            boxes.append(box)
                        
                        if len(boxes) > 0:
                            boxes = np.array(boxes)                                
                            # TODO: change ref box to median box of top k boxes
                            k = 3
                            k_closest_idxs = np.argsort([len(cp) for cp in cluster_points])[-k:]
                            k_closest_boxes = boxes[k_closest_idxs]
                            heights = np.array([np.max(cp[..., 2]) for cp in cluster_points])
                            # tob_k_boxes = boxes[np.argsort([len(p) for p in self.accumulated_detections.points[cluster_id]])[-k:]]
                            # tob_k_median_box = np.median(tob_k_boxes, axis=0)
                            k_closest_median_box = np.median(k_closest_boxes, axis=0)
                                                    
                            # closest corner: corners[np.linalg.norm(corners, axis=1).argmin()]
                            corner_list_ego = [pointcloud_utils.apply_transform(np.concatenate([corner_list[c_idx], np.zeros((4, 1))], axis=1), 
                                                                                self.lidar_frame_list[f_idx].transform_to_ego)[..., :2] for c_idx, f_idx in enumerate(track.frame_indices)]
                            closest_corner_idxs = np.array([np.linalg.norm(corners, axis=1).argmin() for corners in corner_list_ego])
                            closest_corners = np.array([corner_list[cc_idx][cc] for cc_idx, cc in enumerate(closest_corner_idxs)])
                            # closest corner point and old boxes for debugging
                            closest_corner_points = np.concatenate([np.array(closest_corners), boxes[..., 2][None].T], axis=1)
                            boxes_old = deepcopy(boxes)
                            
                            # align median box with closest corner to ego vehicle
                            for cc_idx, cc in enumerate(closest_corner_idxs):
                                diff_w = k_closest_median_box[3] - boxes[cc_idx, 3]
                                diff_l = k_closest_median_box[4] - boxes[cc_idx, 4]
                                angle = np.arctan2(motion_vectors[cc_idx][1], motion_vectors[cc_idx][0])
                                if cc == 0:
                                    # max x, min y
                                    boxes[cc_idx, 0] -= (diff_w / 2) * np.cos(angle)
                                    boxes[cc_idx, 1] -= (diff_w / 2) * np.sin(angle)

                                    boxes[cc_idx, 0] += (diff_l / 2) * np.sin(-angle)
                                    boxes[cc_idx, 1] += (diff_l / 2) * np.cos(-angle)
                                if cc == 1:
                                    # min x, min y
                                    boxes[cc_idx, 0] += (diff_w / 2) * np.cos(angle)
                                    boxes[cc_idx, 1] += (diff_w / 2) * np.sin(angle)

                                    boxes[cc_idx, 0] += (diff_l / 2) * np.sin(-angle)
                                    boxes[cc_idx, 1] += (diff_l / 2) * np.cos(-angle)
                                if cc == 2:
                                    # min x, max y
                                    boxes[cc_idx, 0] += (diff_w / 2) * np.cos(angle)
                                    boxes[cc_idx, 1] += (diff_w / 2) * np.sin(angle)

                                    boxes[cc_idx, 0] -= (diff_l / 2) * np.sin(-angle)
                                    boxes[cc_idx, 1] -= (diff_l / 2) * np.cos(-angle)
                                if cc == 3:
                                    # max x, max y
                                    boxes[cc_idx, 0] -= (diff_w / 2) * np.cos(angle)
                                    boxes[cc_idx, 1] -= (diff_w / 2) * np.sin(angle)

                                    boxes[cc_idx, 0] -= (diff_l / 2) * np.sin(-angle)
                                    boxes[cc_idx, 1] -= (diff_l / 2) * np.cos(-angle)
                            # set dimension to median dimension -> ignore height
                            boxes[..., 3:6] = k_closest_median_box[3:6]
                            boxes[..., 2] = heights - (k_closest_median_box[5] / 2)
                            
                            # if all points (or allmost all) are still covered by the new boxes -> otherwise the boxes are static
                            for b_idx, box in enumerate(boxes):
                                track.detections[b_idx].update_bounding_box(box)
                                track.detections[b_idx].static_track = False
                            track.static = False
                        else:
                            for d in track.detections:
                                d.static_track = True
                                cluster_points = d.cluster_points
                                corners, rz, area = getattr(pointcloud_utils, method['name'])(cluster_points[:, :2], **method['args'])
                                l = np.linalg.norm(corners[0] - corners[1])
                                w = np.linalg.norm(corners[0] - corners[-1])
                                c = (corners[0] + corners[2]) / 2
                                if w > l:
                                    l, w = w, l
                                    rz += np.pi / 2
                                # assumed height
                                height = cluster_points[:, 2].max() - cluster_points[:, 2].min()
                                box = np.array([c[0], c[1], cluster_points[:, 2].min() + height / 2, l, w, height + 0.3, rz])
                                d.update_bounding_box(box)
                                    
            self.sync_lidar_frames()
            
    def propagate_labels(self, **kwargs):
        self.reset_progress_bar('Propagate labels')
        min_length = kwargs.get('min_length', 5)
        cls_key = kwargs.get('classification_key', 'clip')
        self.cls_key = cls_key
        def check_box(bounding_box):
            l, w, h = bounding_box[3:6]
           
            if h > 0.8 and h <= 2.3 and w > 0.2 and w <= 1 and l > 0.2 and l <= 1:
                return 'Pedestrian'
            elif h > 1.4 and h <= 2 and w > 0.5 and w <= 1 and l > 1 and l <= 2.5:
                return 'Cyclist'
            elif w > 0.5 and w <= 3 and l > 0.5 and l <= 8.0 and h > 1 and h <= 3:
                return 'Vehicle'
            else:
                return 'Background'
        
        for track in self.tracker.tracks_valid:
            # set all tracks shorter than min_length = 5 to invalid
            if len(track) < min_length:
                for det in track.detections:
                    det.valid = False
                continue
                        
            max_score = 0
            class_name = 'Background'
            class_count = {}
            # find class name with max score and count of each class
            for d_idx, d in enumerate(track.detections):
                if d.track_prediction:
                    continue
                if d.object_class_score[cls_key] > max_score:
                    max_score = d.object_class_score[cls_key]
                    class_name = d.object_class[cls_key]
                    
                if d.object_class[cls_key] not in class_count:
                    class_count[d.object_class[cls_key]] = 1
                else:
                    class_count[d.object_class[cls_key]] += 1
                    
            # check overlap for moving objects
            if not track.static:
                boxes = deepcopy(np.array([d.bounding_box for d in track.detections]))
                # select largest box
                box_ref = boxes[np.argmax(np.prod(boxes[..., 3:5], axis=1))]
                box_ref[..., 2] = 0
                box_ref[..., 5] = 1
                boxes[..., 2] = 0
                boxes[..., 5] = 1
                boxes_tensor = torch.tensor(boxes[..., 0:7], dtype=torch.float).cuda()
                box_ref_tensor = torch.tensor(box_ref[None, ...], dtype=torch.float).cuda()
                iou = iou3d_nms_utils.boxes_iou3d_gpu(box_ref_tensor, boxes_tensor)
                if np.count_nonzero(iou.cpu().numpy()) == len(boxes):
                    track.static = True
                    for d in track.detections:
                        d.static_track = True
            
            # correct static box and filter too large boxes
            if track.static:
                boxes = []
                n_points = []
                k = 10
                for d_idx, d in enumerate(track.detections):
                    if d.track_prediction:
                        continue
                    boxes.append(d.bounding_box)
                    n_points.append(len(d.cluster_points))
                    
                if len(boxes) > 0:
                    boxes = np.array(boxes)[np.argsort(n_points)[::-1][:k]]
                    max_bins, angles = pointcloud_utils.bin_angles(boxes[..., 6])
                    
                    median_box = np.median(boxes, axis=0)
                    median_box[6] = np.mean(angles)
                
                    l, w, h = median_box[3:6]
                    if l < 0.2 or l > 20 or w < 0.2 or w > 3.5 or h < 0.5 or h > 4:
                        track.valid = False
                        for d_idx, d in enumerate(track.detections):
                            d.valid = False
                        continue
                    else:
                        for d_idx, d in enumerate(track.detections):
                            d.update_bounding_box(median_box)
            
            if not track.static:
                for d_idx, d in enumerate(track.detections):
                    if d.track_prediction:
                        continue
                    if class_name in self.dataset.class_names and (max_score >= 0.5 or (class_count[class_name]) / len(track.detections) >= 0.6):
                        d.object_class[cls_key] = class_name
                        d.object_class_score[cls_key] = max_score
                        track.class_label_corrected = True
                        track.class_label = class_name
                    elif class_name in self.dataset.class_names and (class_name == 'Cyclist' or class_name == 'Pedestrian') and (max_score >= 0.35 or (class_count[class_name]) / len(track.detections) >= 0.6):
                        d.object_class[cls_key] = class_name
                        d.object_class_score[cls_key] = 0.7
                        track.class_label_corrected = True
                        track.class_label = class_name
                    elif class_name == 'Background' and max_score >= 0.3:
                        d.object_class[cls_key] = class_name
                        d.object_class_score[cls_key] = max_score
                        track.class_label_corrected = True
                        track.class_label = class_name
                    else:
                        new_cls_label = check_box(d.bounding_box)
                        track.class_label_corrected_by_size = new_cls_label != d.object_class[cls_key]
                        track.class_label = new_cls_label
                        d.object_class[cls_key] = new_cls_label
                        d.object_class_score[cls_key] = 0.5
                        
                    d.static_track = False
                    # enlarge box by small margin
                    box = deepcopy(d.bounding_box)
                    box[3:5] += 0.3
                    d.update_bounding_box(box)
            else:
                for d_idx, d in enumerate(track.detections):
                    if d.track_prediction:
                        continue
                    if class_name in self.dataset.class_names and (max_score >= 0.5 or (class_count[class_name]) / len(track.detections) >= 0.6):
                        d.object_class[cls_key] = class_name
                        d.object_class_score[cls_key] = max_score
                        track.class_label_corrected = True
                        track.class_label = class_name
                    elif class_name == 'Background' and max_score >= 0.3:
                        d.object_class[cls_key] = 'Background' 
                        d.object_class_score[cls_key] = 1.0
                        track.class_label_corrected = True
                        track.class_label = class_name
                    else:
                        pass
                    
                    # enlarge box by small margin
                    box = deepcopy(d.bounding_box)
                    box[3:5] += 0.3
                    d.update_bounding_box(box)
        
        self.progress_bar.update(1)

    def evaluate_sequence(self, modes=['detection_3d'], logger=None, **kwargs):
        classification_key = kwargs.get('classification_key', 'clip')
                
        if 'detection_3d' in modes:
            self.reset_progress_bar('Evaluate Detection 3D')
            
            for lidar_frame in self.lidar_frame_list:
                boxes = []
                names = []
                scores = []
                moving = []
                for d in lidar_frame.detections:
                    if d.valid and d.object_class is not None and classification_key in d.object_class:
                        if d.object_class[classification_key] in self.dataset.class_names:
                            boxes.append(d.bounding_box)
                            scores.append(d.object_class_score[classification_key])
                            names.append(d.object_class[classification_key])
                            moving_ = True if (d.static_track is not None and not d.static_track) else False
                            moving.append(moving_)
                
                if len(boxes) > 0:
                    predicted_boxes = pointcloud_utils.apply_transform(np.array(boxes), lidar_frame.transform_to_ego, box=True)

                else:
                    predicted_boxes = np.zeros((0, 7))
                        
                self.detection_3d_result_list.append({
                    'boxes_lidar': predicted_boxes,
                    'name': np.array(names),
                    'score': np.array(scores),
                    'moving': np.array(moving)
                })
            