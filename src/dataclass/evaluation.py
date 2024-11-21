import numpy as np

from dataclasses import dataclass, field

@dataclass
class ClusterResult:
    point_recall: float = 0.0
    box_recall: float = 0.0
    box_precision: float = 0.0
    
@dataclass
class Accuracy:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    

@dataclass
class SequenceEvaluation:
    """Evaluation class for storing evaluation results."""
    cluster_results: 'list[ClusterResult]' = field(default_factory=list)
    cluster_filtered_results: 'list[ClusterResult]' = field(default_factory=list)
    cluster_filtered_tracked_results: 'list[ClusterResult]' = field(default_factory=list)
    cluster_moving_accuracy: 'list[Accuracy]' = field(default_factory=list)
    
    def cluster_results_mean(self):
        return ClusterResult(point_recall=np.mean([cr.point_recall for cr in self.cluster_results]),
                             box_recall=np.mean([cr.box_recall for cr in self.cluster_results]),
                             box_precision=np.mean([cr.box_precision for cr in self.cluster_results]))
        
    def cluster_filtered_results_mean(self):
        return ClusterResult(point_recall=np.mean([cr.point_recall for cr in self.cluster_filtered_results]),
                             box_recall=np.mean([cr.box_recall for cr in self.cluster_filtered_results]),
                             box_precision=np.mean([cr.box_precision for cr in self.cluster_filtered_results]))
    
    def cluster_filtered_tracked_results_mean(self):
        return ClusterResult(point_recall=np.mean([cr.point_recall for cr in self.cluster_filtered_tracked_results]),
                             box_recall=np.mean([cr.box_recall for cr in self.cluster_filtered_tracked_results]),
                             box_precision=np.mean([cr.box_precision for cr in self.cluster_filtered_tracked_results]))
    
    def cluster_moving_precision_mean(self):
        precision_list = [acc.precision for acc in self.cluster_moving_accuracy if acc.precision is not None]
        return np.mean(precision_list) if len(precision_list) > 0 else 0
    
    def cluster_moving_recall_mean(self):
        recall_list = [acc.recall for acc in self.cluster_moving_accuracy if acc.recall is not None]
        return np.mean(recall_list) if len(recall_list) > 0 else 0
    
    def cluster_moving_tp(self):
        return np.sum([acc.tp for acc in self.cluster_moving_accuracy if acc is not None])
    
    def cluster_moving_fp(self):
        return np.sum([acc.fp for acc in self.cluster_moving_accuracy if acc is not None])
    
    def cluster_moving_fn(self):
        return np.sum([acc.fn for acc in self.cluster_moving_accuracy if acc is not None])