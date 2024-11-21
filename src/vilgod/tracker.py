import numpy as np

from src.dataclass.objects import Detection, Track
from src.utils import tracking_utils


class Tracker():
    def __init__(self, name, track_cfg) -> None:
        self.name = name
        self.cfg = track_cfg
        self.tracks: list[Track] = []
        self.mode = self.cfg.mode
        self.assignment_fn = getattr(tracking_utils, self.cfg.assignment.method)
        self.next_tid = 0        
        
    def __len__(self):
        return len(self.tracks)

    def next_track_id(self):
        tid_ = self.next_tid
        self.next_tid += 1
        return tid_
    
    @property
    def tracks_active(self):
        return [track for track in self.tracks if track.active]

    @property
    def tracks_valid(self):
        return [track for track in self.tracks if track.valid]
    
    def next(self, detection_list: 'list[Detection]', frame_index: int):      
        # predict track states
        for track in self.tracks_active:
            track.predict()
            
        tracks = np.array([track.current_prediction for track in self.tracks_active])
        if self.mode == 'bounding_box':
            detections = np.array([detection.bounding_box for detection in detection_list])
        elif self.mode == 'cluster_center':
            detections = np.array([detection.cluster_mass_center for detection in detection_list])
        
        weights = None
        matches_d_t, mask_d_t, distance = self.assignment_fn(detections, tracks, weights=weights, **self.cfg.assignment)
        matches_d_t_all = matches_d_t.copy()
        
        # update tracks
        if len(matches_d_t) > 0:
            det_idx_d_t = matches_d_t[:, 0]
            matches_d_t = matches_d_t[mask_d_t[det_idx_d_t]]
            
        for t_idx, track in enumerate(self.tracks_active):
            if len(matches_d_t) > 0 and t_idx in matches_d_t[:, 1]:
                track.update(detection_list[matches_d_t[matches_d_t[:, 1] == t_idx, 0][0]], frame_index)
            elif len(matches_d_t_all) > 0 and t_idx in matches_d_t_all[:, 1]:
                p1 = detection_list[matches_d_t_all[matches_d_t_all[:, 1] == t_idx, 0][0]].cluster_points
                p2 = track.detections[-1].cluster_points
                c1 = detection_list[matches_d_t_all[matches_d_t_all[:, 1] == t_idx, 0][0]].cluster_mass_center
                c2 = track.detections[-1].cluster_mass_center

                if (min(len(p1), len(p2)) / max(len(p1), len(p2))) > 0.7 and np.linalg.norm(c1 - c2) < 5:
                    track.update(detection_list[matches_d_t_all[matches_d_t_all[:, 1] == t_idx, 0][0]], frame_index)
                else:
                    track.update(None, frame_index)
            else:
                if track.n_missed >= self.cfg.max_missed:
                    track.finalize(self.cfg)
                else:
                    track.update(None, frame_index)

        # init new tracks for unmatched detections
        for d_idx, detection in enumerate(detection_list):
            if len(matches_d_t) == 0 or d_idx not in matches_d_t[:, 0]:
                track = Track(self.next_track_id(), self.mode)
                track.init(detection, frame_index)
                self.tracks.append(track)
    
    def finish(self):
        for track in self.tracks_active:
            track.finalize(self.cfg)