import numpy as np

from copy import deepcopy

from pcdet.datasets.argo2.argo2_dataset import Argo2Dataset as Argo2DatasetBase
from pcdet.utils import common_utils, box_utils

from src.utils import pointcloud_utils

class Argo2Dataset(Argo2DatasetBase):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, 
                 start_sequence=None, end_sequence=None):
        super().__init__(dataset_cfg, class_names, training, root_path, logger)
        self.index_mapping = []
        self.start_sequence = None
        self.end_sequence = None
        self._sequence_mapping = self.create_sequence_mapping(start_sequence, end_sequence)
        
        self._sequence_indices = None
        self._moving_track_ids = None
    
    @property
    def sequence_mapping(self):
        return self._sequence_mapping.copy()
    
    @property
    def sequence_names(self):
        return self.filter_sequence_names(list(self._sequence_mapping.keys()),
                                          self.start_sequence, 
                                          self.end_sequence)
 
    @property
    def sequence_length(self):
        return len(self._sequence_indices) if self._sequence_indices is not None else 0
    
    @property
    def sequence_indices(self):
        return self._sequence_indices.copy()

    @property
    def sequence_infos(self):
        return [self.infos[idx] for idx in self._sequence_indices]
    
    def next_sequence(self):
        for name in self.sequence_names:
            start_index = self.sequence_mapping[name]['start']
            seq_length = self.sequence_mapping[name]['length']
            self._sequence_indices = list(range(start_index, start_index + seq_length)) 
            frame_idxs = np.argsort([int(info['uuid'].split('/')[1]) for info in self.sequence_infos])
            self._sequence_indices = [self._sequence_indices[i] for i in frame_idxs]
            self.adapt_annos()
            tracks, _ = self.extract_moving_tracks()
            self._moving_track_ids = [k for k, v in tracks.items() if v['moving']]

            yield name

    def create_sequence_mapping(self, start=0, end=999):
        sequence_mapping = {}
        for iidx in range(len(self.infos)):
            sequence_name = self.infos[iidx]['uuid'].split('/')[0]
            sample_idx = self.infos[iidx]['sample_idx']
            spilit_id, seq_id, frame_id = sample_idx[0], sample_idx[1:4], sample_idx[4:7]
            if sequence_name not in sequence_mapping:
                sequence_mapping[sequence_name] = {'start': iidx, 'length': 1, 'indices': [int(frame_id)]}
            else:
                sequence_mapping[sequence_name]['length'] += 1
                sequence_mapping[sequence_name]['indices'].append(int(frame_id))
        
        self.start_sequence = start if (start is not None) and (start < len(sequence_mapping.keys())) else 0
        self.end_sequence = end if (end is not None) and (end <= len(sequence_mapping.keys())) else len(sequence_mapping.keys())
        self.logger.info(f'Using [{self.end_sequence - self.start_sequence}/{len(sequence_mapping.keys())}] sequences from {self.start_sequence} to {self.end_sequence}.')
        return sequence_mapping
    
    def set_split(self, split):
        super().set_split(split)
        self.argo2_infos = []
        mode = [k for k, v in self.dataset_cfg.DATA_SPLIT.items() if v == split][0]
        self.include_argo2_data(mode)
        self._sequence_mapping = self.create_sequence_mapping(self.start_sequence, self.end_sequence)

    def filter_sequence_names(self, sequence_names, sequence_start_idx=0, sequence_end_idx=0):
        n_sequences = sequence_end_idx - sequence_start_idx
        if n_sequences > 0:
            # from start to end
            sequence_names = sequence_names[sequence_start_idx:sequence_end_idx]
        elif sequence_start_idx > 0 and sequence_start_idx < len(sequence_names):
            # all sequences after start
            sequence_names = sequence_names[sequence_start_idx:]
    
        return sequence_names
    
    def adapt_annos(self):
        for f_idx in range(self.sequence_length):
            info = self.sequence_infos[f_idx]
            annos = info['annos']
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            annos['gt_boxes_lidar'] = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            for n_idx, name in enumerate(annos['name']):
                if name in self.dataset_cfg.CLASS_MAPPING:
                    annos['name'][n_idx] = self.dataset_cfg.CLASS_MAPPING[name]
                elif name in self.class_names:
                    pass
                else:
                    annos['name'][n_idx] = 'unknown'
    
    def get_annos(self, index, transformation=None, filtered=True):
        info = self.sequence_infos[index]
        annos_dict = {}
        if 'annos' in info:
            annos = info['annos']
            
            # get moving info for all objects
            if not filtered:
                annos_dict.update({
                    'gt_names': annos['name'],
                    'gt_boxes': annos['gt_boxes_lidar'],
                    'num_points_in_gt': annos.get('num_points_in_gt', None),
                    'obj_ids': annos['obj_ids']
                })
                
                return annos_dict
            
            # set moving info for all objects
            if self._moving_track_ids is not None:
                moving = np.array([oid in self._moving_track_ids for oid in annos['obj_ids']])
                info['annos']['moving'] = moving
            
            
            # filter objects as in training ##########################################################
            annos = common_utils.drop_info_with_name(annos, name='unknown')
            
            # filter empty boxes
            keep_mask = annos['num_points_in_gt'] >= 1
            for k, v in annos.items():
                annos[k] = v[keep_mask]
                
            self.sequence_infos[index]['annos'] = annos            

            if self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False):
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(annos['gt_boxes_lidar'])
            else:
                gt_boxes_lidar = annos['gt_boxes_lidar']

            if self.training and self.dataset_cfg.get('FILTER_EMPTY_BOXES_FOR_TRAIN', False) and len(annos['name']) > 0:
                mask = (annos['num_points_in_gt'] > 0)  # filter empty boxes
                mask &= [name in self.class_names for name in annos['name']]
                annos['name'] = annos['name'][mask]
                gt_boxes_lidar = gt_boxes_lidar[mask]
                annos['num_points_in_gt'] = annos['num_points_in_gt'][mask]
                annos['obj_ids'] = annos['obj_ids'][mask]

            if len(gt_boxes_lidar) > 0 and transformation is not None:
                gt_boxes_lidar = pointcloud_utils.apply_transform(gt_boxes_lidar, transformation, box=True)

            annos_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': gt_boxes_lidar,
                'num_points_in_gt': annos.get('num_points_in_gt', None),
                'obj_ids': annos['obj_ids']
            })
            
            if annos_dict.get('gt_boxes', None) is not None:
                selected = common_utils.keep_arrays_by_name(annos_dict['gt_names'], self.class_names)
                for k, v in annos_dict.items():
                    if isinstance(v, np.ndarray):
                        annos_dict[k] = v[selected]
                        
            if self._moving_track_ids is not None:
                annos_dict['moving'] = np.array([oid in self._moving_track_ids for oid in annos_dict['obj_ids']])                

        return annos_dict
    
    def get_lidar_points(self, index, transformation=None):
        info = self.sequence_infos[index]
        pts = self.get_lidar(info['sample_idx'])

        if transformation is not None:
            pts = pointcloud_utils.apply_transform(pts=pts, transformation=transformation)

        return pts
    
    def extract_moving_tracks(self, threshold=1.0):
        tracks = {}
        track_template = {'indices': [], 'gt_boxes': [], 'gt_boxes_ref': [], 'gt_names': [], 'num_points_in_gt': []}
        for f_idx in range(self.sequence_length):            
            annos_dict = self.get_annos(f_idx, transformation=None, filtered=False)
            for t_idx, tid in enumerate(annos_dict['obj_ids']):
                # if annos_dict['num_points_in_gt'][t_idx] > 5:
                if tid not in tracks:
                    tracks[tid] = deepcopy(track_template)
                tracks[tid]['indices'].append(f_idx)
                tracks[tid]['gt_boxes'].append(annos_dict['gt_boxes'][t_idx].copy())
                tracks[tid]['gt_names'].append(annos_dict['gt_names'][t_idx])
                tracks[tid]['num_points_in_gt'].append(annos_dict['num_points_in_gt'][t_idx])
        
        # iterate over tracks and find moving ones
        n_moving_objects = 0
        for key, track in tracks.items():
            tracks[key]['moving'] = False
            if len(track['indices']) > 1:
                ref_pose = self.sequence_infos[track['indices'][0]]['pose']
                ref_box = track['gt_boxes'][0].copy()
                tracks[key]['gt_boxes_ref'].append(ref_box)
                for i in range(len(track['indices']) - 1):
                    pose = self.sequence_infos[track['indices'][i+1]]['pose']
                    box = track['gt_boxes'][i+1].copy()
                    box[:7] = common_utils.apply_transform(np.array([box[:7]]), np.linalg.inv(ref_pose) @ pose, box=True)
                    tracks[key]['gt_boxes_ref'].append(box)
                    if np.linalg.norm(ref_box[:3] - box[:3]) > threshold:
                        tracks[key]['moving'] = True
                        tracks[key]['gt_boxes_ref'] = np.array(tracks[key]['gt_boxes_ref'])
                        n_moving_objects += len(track['gt_boxes'])
                        break
                    
        return tracks, n_moving_objects
    
    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def waymo_eval(eval_det_annos, eval_gt_annos):
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Currently, memory growth needs to be the same across GPUs
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = tf.config.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                except RuntimeError as e:
                    # Memory growth must be set before GPUs have been initialized
                    print(e)
                    
            from src.datasets.waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
            eval = OpenPCDetWaymoDetectionMetricsEstimator()
            
            ap_dict = eval.waymo_evaluation(
                eval_det_annos, eval_gt_annos, class_name=class_names, distance_thresh=1000, 
                fake_gt_infos=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False), cfg=eval_cfg
            )

            return ap_dict
        
        def argo2_eval(eval_det_annos, eval_gt_annos):
            """Evaluation in Argo2 protocol.

            Args:
                results (list[dict]): Testing results of the dataset.
                metric (str | list[str]): Metrics to be evaluated.
                    Default: 'waymo'. Another supported metric is 'Argo2'.
                logger (logging.Logger | str | None): Logger used for printing
                    related information during evaluation. Default: None.
                pklfile_prefix (str | None): The prefix of pkl files. It includes
                    the file path and the prefix of filename, e.g., "a/b/prefix".
                    If not specified, a temp file will be created. Default: None.
                submission_prefix (str | None): The prefix of submission datas.
                    If not specified, the submission data will not be generated.
                show (bool): Whether to visualize.
                    Default: False.
                out_dir (str): Path to save the visualization results.
                    Default: None.
                pipeline (list[dict], optional): raw data loading for showing.
                    Default: None.

            Returns:
                dict[str: float]: results of each evaluation metric
            """
            from av2.evaluation.detection.constants import CompetitionCategories
            from av2.evaluation.detection.utils import DetectionCfg
            from av2.evaluation.detection.eval import evaluate
            from av2.utils.io import read_feather

            dts = self.format_results(eval_det_annos, class_names)
            argo2_root = self.root_path
            val_anno_path = osp.join(argo2_root, 'val_anno.feather')
            gts = read_feather(Path(val_anno_path))
            gts = gts.set_index(["log_id", "timestamp_ns"]).sort_values("category")

            valid_uuids_gts = gts.index.tolist()
            valid_uuids_dts = dts.index.tolist()
            valid_uuids = set(valid_uuids_gts) & set(valid_uuids_dts)
            gts = gts.loc[list(valid_uuids)].sort_index()

            categories = set(x.value for x in CompetitionCategories)
            categories &= set(gts["category"].unique().tolist())

            dataset_dir = Path(argo2_root) / 'sensor' / 'val'
            cfg = DetectionCfg(
                dataset_dir=dataset_dir,
                categories=tuple(sorted(categories)),
                max_range_m=self.evaluate_range,
                eval_only_roi_instances=True,
            )

            # Evaluate using Argoverse detection API.
            eval_dts, eval_gts, metrics = evaluate(
                dts.reset_index(), gts.reset_index(), cfg
            )

            valid_categories = sorted(categories) + ["AVERAGE_METRICS"]
            ap_dict = {}
            for index, row in metrics.iterrows():
                ap_dict[index] = row.to_json()
            return metrics.loc[valid_categories], ap_dict
        
        eval_cfg = kwargs.get('eval_cfg', {})
        eval_range = kwargs.get('eval_range', self.point_cloud_range[[0, 1, 3, 4]])
        sampling_rate = kwargs.get('sampling_rate', 1)
        score_thresh = kwargs.get('score_thresh', 0.0) 

        eval_det_annos = deepcopy(det_annos)
        print(len(eval_det_annos))
        eval_det_annos = eval_det_annos[::sampling_rate]
        for anno in eval_det_annos:
            if len(anno['boxes_lidar']) > 0:
                if kwargs.get('bev', False):   
                    anno['boxes_lidar'][..., 2] = 0.0
                    anno['boxes_lidar'][..., 5] = 1.0
                if kwargs.get('class_agnostic', False):
                    anno['name'] = [class_names[0] for _ in range(len(anno['name']))]
                
                corners = box_utils.boxes_to_corners_3d(anno['boxes_lidar'])
                mask = (np.count_nonzero(((corners[..., :2] < eval_range[0:2]) | (corners[..., :2] > eval_range[2:4])).reshape(corners.shape[0], -1), axis=1) == 0)
                mask[anno['score'] < score_thresh] = False
                anno['boxes_lidar'] = np.array(anno['boxes_lidar'])[mask]
                anno['name'] = np.array(anno['name'])[mask]
                anno['score'] = np.array(anno['score'])[mask]
        
        if kwargs.get('sequence', False):
            eval_gt_annos = [deepcopy(info['annos']) for info in self.sequence_infos]
        else:
            indices = kwargs.get('indices', self.index_mapping)
            indices = indices if len(indices) > 0 else self.index_mapping
            if len(indices) == 0:
                indices = np.arange(len(self.argo2_infos))
            eval_gt_annos = [deepcopy(self.infos[idx]['annos']) for idx in indices]
            
        if kwargs.get('class_agnostic', False):
            for anno in eval_gt_annos:
                anno['name'] = np.array([class_names[0] if name in class_names else name for name in anno['name']])
        # print('len annos:', len(eval_gt_annos))
        # print('len indices:', len(indices), ' - ', indices)
        eval_gt_annos = eval_gt_annos[::sampling_rate]
        for a_idx, anno in enumerate(eval_gt_annos):
            if 'difficulty' not in anno or anno['difficulty'] is None:
                anno['difficulty'] = np.ones(len(anno['name']))
            anno = common_utils.drop_info_with_name(anno, name='unknown')
            if np.array(anno['gt_boxes_lidar']).shape[0] > 0:       
                assert np.array(anno['gt_boxes_lidar']).shape[0] == len(anno['name']), 'Number of boxes and number of names are not equal'
                corners = box_utils.boxes_to_corners_3d(np.array(anno['gt_boxes_lidar']))
                mask = (np.count_nonzero(((corners[..., :2] < eval_range[0:2]) | (corners[..., :2] > eval_range[2:4])).reshape(corners.shape[0], -1), axis=1) == 0)
                if kwargs.get('moving', False):
                    mask &= anno['moving']
                if kwargs.get('static', False):
                    mask &= ~anno['moving']
                
                eval_gt_annos[a_idx]['difficulty'] = np.array(anno['difficulty'])[mask]
                eval_gt_annos[a_idx]['gt_boxes_lidar'] = np.array(anno['gt_boxes_lidar'])[mask]
                eval_gt_annos[a_idx]['name'] = np.array(anno['name'])[mask]
                eval_gt_annos[a_idx]['num_points_in_gt'] = np.array(anno['num_points_in_gt'])[mask]
                
        
            if kwargs.get('bev', False):
                if len(anno['gt_boxes_lidar']) > 0:
                    eval_gt_annos[a_idx]['gt_boxes_lidar'][..., 2] = 0.0
                    eval_gt_annos[a_idx]['gt_boxes_lidar'][..., 5] = 1.0
                
        if kwargs.get('eval_metric', 'waymo') == 'waymo':
            ap_dict = waymo_eval(eval_det_annos, eval_gt_annos)
        elif kwargs.get('eval_metric', 'argo2') == 'argo2':
            raise NotImplementedError
            ap_dict = argo2_eval(eval_det_annos, eval_gt_annos)
        else:
            raise NotImplementedError

        return ap_dict
