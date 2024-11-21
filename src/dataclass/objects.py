import numpy as np

from dataclasses import dataclass, field
from scipy import spatial
from PIL import Image
from copy import deepcopy

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

from src.utils.common_utils import angle_between_vectors
from src.utils import eval_utils, pointcloud_utils, common_utils


@dataclass
class BoundingBox3D:
    center_x: float
    center_y: float
    center_z: float
    length: float
    width: float
    height: float
    orientation: float

    _array: np.array = field(init=False)

    def __post_init__(self):
        self._array = np.array([self.center_x, self.center_y, self.center_z,
                               self.length, self.width, self.height, self.orientation])

    @property
    def array(self):
        return self._array


@dataclass
class Detection:
    '''
    parameters:
    static: describes the motion state regarding epochal motion
    static_track: describes the motion state regarding track motion
    tid: track id of assigned track, -1 if not assigned
    '''
    cluster_id: int
    cluster_points: np.array
    cluster_points_index: np.array
    cluster_points_flow: np.array = None
    cluster_points_index_fp: np.array = None
    cluster_points_index_fn: np.array = None
    cluster_points_entropy: np.array = None
    cluster_center: np.array = field(init=False)
    _cluster_mass_center: np.array = field(init=False)
    cluster_feature: np.array = None
    match_distances: np.array = None
    matched_detections: 'list[Detection]' = field(default_factory=list)

    valid: bool = True
    static: bool = True
    static_track = None
    track_prediction: bool = False
    feature_score: float = None
    depth_image: Image = None
    n_matches: int = 0
    tid: int = -1
    filter_dict: dict = field(default_factory=dict)

    # dicts -> contain different experiments, e.g. gt, clustered, cluster+tracking, ...
    object_class: dict = None
    object_class_score: dict = None
    object_class_predictions: dict = None
    object_class_predictions_score: dict = None
    object_class_predictions_detailed: dict = None
    _bounding_box: BoundingBox3D = None

    gt: bool = False
    gt_cluster_id = None
    gt_id: str = None
    gt_assigned: bool = False
    gt_iou: float = 0.0
    gt_moving: bool = False
    _gt_bounding_box: BoundingBox3D = None

    def __post_init__(self):
        self.cluster_center = self.cluster_points.mean(axis=0)
        self._cluster_mass_center = np.median(self.cluster_points, axis=0)

    @property
    def serialize(self):
        detection_data = {}
        parameter_list = ['cluster_id', '_bounding_box', 'valid', 'static', 'gt_assigned',
                          'cluster_points_index', 'object_class_predictions', 'tid', 'static_track',
                          'object_class_predictions_detailed', 'object_class_predictions_score',
                          'object_class', 'object_class_score']

        for p in parameter_list:
            if hasattr(self, p):
                if p == '_bounding_box' and self.__getattribute__(p) is not None:
                    detection_data[p] = self.__getattribute__(p).array
                elif self.__getattribute__(p) is not None:
                    detection_data[p] = self.__getattribute__(p)

        # serialize only if not ground truth
        return None if (self.gt and not self.gt_assigned) else detection_data

    @property
    def bounding_box(self):
        if self.gt:
            return self._gt_bounding_box.array if self._gt_bounding_box is not None else None
        else:
            return self._bounding_box.array if self._bounding_box is not None else None
        
    @property
    def height(self):
        return np.max(self.cluster_points[..., 2]) - np.min(self.cluster_points[..., 2])
    
    @property
    def n_points(self):
        return len(self.cluster_points)
    
    @property
    def cluster_mass_center(self):
        self._cluster_mass_center = np.median(self.cluster_points, axis=0)
        return self._cluster_mass_center

    @property
    def is_valid(self):
        return self.valid or not self.static # or not self.static_track

    def add_object_entry(self, entry_name, key, data):
        assert entry_name in ['object_class', 'object_class_score', 'object_class_predictions',
                              'object_class_predictions_score', 'object_class_predictions_detailed']
        if self.__getattribute__(entry_name) is None:
            self.__setattr__(entry_name, {})
        self.__getattribute__(entry_name)[key] = data

    def sync_detection(self, detection_data):
        for k, v in detection_data.items():
            if hasattr(self, k):
                if k == '_bounding_box':
                    self._bounding_box = BoundingBox3D(*v)
                else:
                    self.__setattr__(k, v)

    def update_bounding_box(self, bounding_box: BoundingBox3D):
        if self.gt:
            self._gt_bounding_box = BoundingBox3D(*bounding_box)
        else:
            self._bounding_box = BoundingBox3D(*bounding_box)
            
    def update_bounding_box_size(self, size):
        if self.gt:
            self._gt_bounding_box = BoundingBox3D(self._gt_bounding_box.center_x, self._gt_bounding_box.center_y, self._gt_bounding_box.center_z,
                                                  size[0], size[1], size[2], self._gt_bounding_box.orientation)
        else:
            self._bounding_box = BoundingBox3D(self._bounding_box.center_x, self._bounding_box.center_y, self._bounding_box.center_z,
                                               size[0], size[1], size[2], self._bounding_box.orientation)
    
    def filter(self, filters, **kwargs):
        and_valid, or_valid, and_required_valid = [], [], []
        # equip filter arguments with entropy scores
        filter_arguments = {
            'ephemeral_scores': self.cluster_points_entropy,
            'height': self.height
        }
        for k, v in kwargs.items():
            filter_arguments[k] = v
        
        # list of filters: [filter, filter_name, logic, required]
        for filter, name, logic, required in filters:
            valid = filter(points=self.cluster_points[..., :3], **filter_arguments)
            self.filter_dict[name] = valid
            if logic == 'and' and required:
                and_required_valid.append(valid)
            elif logic == 'and':
                and_valid.append(valid)
            elif logic == 'or':
                or_valid.append(valid)
            else:
                raise ValueError(f"Logic for filter {filter['name']} not defined!")
            
        self.valid = (np.all(and_valid) or np.any(or_valid)) and np.all(and_required_valid)
            
    def assign_gt(self, valid_gt, gt_id, iou, gt_moving=False, track_moving=False):
        self.gt_assigned = valid_gt
        self.gt_id = gt_id if gt_id is not None else -1
        self.gt_iou = iou
        self.gt_moving = gt_moving
        self.static_track = not track_moving
    
    def merge_detections(self, detections):
        for d in detections:
            if d.cluster_id == self.cluster_id:
                continue
            if d.n_matches > self.n_matches:
                self.match_distances = d.match_distances
                self.n_matches = d.n_matches
            self.cluster_points = np.concatenate([self.cluster_points, d.cluster_points])
            self.cluster_points_index = np.concatenate([self.cluster_points_index, d.cluster_points_index])
        self.cluster_center = self.cluster_points.mean(axis=0)
            

@dataclass
class Track:
    track_id: int
    mode: str
    valid: bool = True
    active: bool = True
    first_frame: int = None
    last_frame: int = None
    static: bool = True
    class_label_corrected: bool = False
    class_label_corrected_by_size: bool = False
    class_label = 'Background'
    detections: 'list[Detection]' = field(default_factory=list)
    frame_indices: 'list[int]' = field(default_factory=list)
    _miss_count: int = 0
    _current_prediction: np.array = None
    kf: KalmanFilter = None
    covariances: 'list[np.array]' = field(default_factory=list)
    velocities: 'list[np.array]' = field(default_factory=list)

    def __post_init__(self):
        pass
    
    def __len__(self):
        return len(self.detections)

    @property
    def length(self):
        return len(self.detections)

    @property
    def feature(self):
        return self.detections[-1].cluster_feature

    @property
    def current_state(self):
        return self.detections[-1].bounding_box if self.mode == 'bounding_box' else self.detections[-1].cluster_mass_center

    @property
    def current_prediction(self):
        return self._current_prediction

    @property
    def n_missed(self):
        return self._miss_count

    @property
    def max_distance_clusters(self):
        centers = np.array([d.cluster_mass_center for d in self.detections])
        distance_matrix = spatial.distance.cdist(centers[..., :2], centers[..., :2], 'euclidean')
        return np.max(distance_matrix)

    @property
    def max_distance_bounding_boxes(self):
        centers = np.array([d.bounding_box[:3] for d in self.detections])
        distance_matrix = spatial.distance.cdist(centers, centers, 'euclidean')
        return np.max(distance_matrix)

    def _append_detection(self, detection: Detection, frame_index: int):
        self.detections.append(detection)
        if self.first_frame is None:
            self.first_frame = frame_index
        self.last_frame = frame_index
        self.frame_indices.append(frame_index)
        self.covariances.append(self.kf.P)
        self.velocities.append(self.kf.x[2:4])

    def _init_kf(self, dt=0.1, x=[0., 0., 0., 0.]):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array(x)  # position (x, y), velocity (x, y)
        self.kf.F = np.array([[1., 0., dt, 0.],
                              [0., 1., 0., dt],
                              [0., 0., 1., 0.],
                              [0., 0., 0., 1.]])
        self.kf.Q = Q_discrete_white_noise(4, dt, 0.15)
        self.kf.H = np.array([[1., 0., 0, 0],
                              [0., 1., 0, 0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[2:, 2:] *= 50
        self.kf.P *= 10.

    def init(self, detection: Detection, frame_index: int):
        # init kalman filter
        self._init_kf(x=[*detection.cluster_mass_center[:2], 0., 0.])
        # append detection
        self._append_detection(detection, frame_index)
        # init current prediction
        self._current_prediction = self.current_state.copy()

    def predict(self):
        if len(self.detections) > 0:
            # # predict state
            self.kf.predict()
            # update current prediction
            self._current_prediction[:2] = self.kf.x[:2]
            # update z-value with last assigned detection
            self._current_prediction[2] = self.detections[-1].cluster_mass_center[2]

    def update(self, detection: Detection, frame_index: int):
        # update current state
        if detection is not None:
            # reset miss count
            self._miss_count = 0
            # update kalman filter
            self.kf.update(detection.cluster_mass_center[:2])
            # update position within detection
            detection.cluster_mass_center[:2] = self.kf.x[:2]
        else:
            # matched detection
            # increase miss counter
            self._miss_count += 1
            # clone last detection
            detection = deepcopy(self.detections[-1])
            detection.track_prediction = True
            # update position of cloned detection to predicted position
            detection.cluster_mass_center[:2] = self._current_prediction[:2]

        # add detection to list
        self._append_detection(detection, frame_index)

    def finalize(self, min_distance_dynamic=2.0, **kwargs):
        # disable track
        self.active = False
        count = 0
        for d in reversed(self.detections):
            if d.track_prediction:
                count += 1
            else:
                break

        if count > 0:
            self.detections = self.detections[:-count]
            self.frame_indices = self.frame_indices[:-count]