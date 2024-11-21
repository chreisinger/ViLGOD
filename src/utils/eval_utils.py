import torch
import numpy as np
import pandas as pd

from scipy import stats

from src.dataclass.evaluation import ClusterResult, Accuracy
from pcdet.utils import common_utils
from src.utils import pointcloud_utils

from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils


EVAL_MAPPING = {
    'OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/AP':  'Vehicle AP  L1',
    'OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/APH': 'Vehicle APH L1',
    'OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/AP':  'Vehicle AP  L2',
    'OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/APH': 'Vehicle APH L2',
    
    'OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/AP':  'Pedestrian AP  L1',
    'OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/APH': 'Pedestrian APH L1',
    'OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/AP':  'Pedestrian AP  L2',
    'OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/APH': 'Pedestrian APH L2',
    
    'OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/AP':  'Cyclist AP  L1',
    'OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/APH': 'Cyclist APH L1',
    'OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/AP':  'Cyclist AP  L2',
    'OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/APH': 'Cyclist APH L2',
    
    'RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_1/AP':     'Vehicle AP  L1 [0, 30)   ',
    'RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_1/APH':    'Vehicle APH L1 [0, 30)   ',
    'RANGE_TYPE_VEHICLE_[30, 50)_LEVEL_1/AP':    'Vehicle AP  L1 [30, 50)  ',
    'RANGE_TYPE_VEHICLE_[30, 50)_LEVEL_1/APH':   'Vehicle APH L1 [30, 50)  ',
    'RANGE_TYPE_VEHICLE_[50, +inf)_LEVEL_1/AP':  'Vehicle AP  L1 [50, +inf)',
    'RANGE_TYPE_VEHICLE_[50, +inf)_LEVEL_1/APH': 'Vehicle APH L1 [50, +inf)',
    
    'RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_2/AP':     'Vehicle AP  L2 [0, 30)   ',
    'RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_2/APH':    'Vehicle APH L2 [0, 30)   ',
    'RANGE_TYPE_VEHICLE_[30, 50)_LEVEL_2/AP':    'Vehicle AP  L2 [30, 50)  ',
    'RANGE_TYPE_VEHICLE_[30, 50)_LEVEL_2/APH':   'Vehicle APH L2 [30, 50)  ',
    'RANGE_TYPE_VEHICLE_[50, +inf)_LEVEL_2/AP':  'Vehicle AP  L2 [50, +inf)',
    'RANGE_TYPE_VEHICLE_[50, +inf)_LEVEL_2/APH': 'Vehicle APH L2 [50, +inf)',
    
    'RANGE_TYPE_PEDESTRIAN_[0, 30)_LEVEL_1/AP':     'Pedestrian AP  L1 [0, 30)   ',
    'RANGE_TYPE_PEDESTRIAN_[0, 30)_LEVEL_1/APH':    'Pedestrian APH L1 [0, 30)   ',
    'RANGE_TYPE_PEDESTRIAN_[30, 50)_LEVEL_1/AP':    'Pedestrian AP  L1 [30, 50)  ',
    'RANGE_TYPE_PEDESTRIAN_[30, 50)_LEVEL_1/APH':   'Pedestrian APH L1 [30, 50)  ',
    'RANGE_TYPE_PEDESTRIAN_[50, +inf)_LEVEL_1/AP':  'Pedestrian AP  L1 [50, +inf)',
    'RANGE_TYPE_PEDESTRIAN_[50, +inf)_LEVEL_1/APH': 'Pedestrian APH L1 [50, +inf)',
    
    'RANGE_TYPE_PEDESTRIAN_[0, 30)_LEVEL_2/AP':     'Pedestrian AP  L2 [0, 30)   ',
    'RANGE_TYPE_PEDESTRIAN_[0, 30)_LEVEL_2/APH':    'Pedestrian APH L2 [0, 30)   ',
    'RANGE_TYPE_PEDESTRIAN_[30, 50)_LEVEL_2/AP':    'Pedestrian AP  L2 [30, 50)  ',
    'RANGE_TYPE_PEDESTRIAN_[30, 50)_LEVEL_2/APH':   'Pedestrian APH L2 [30, 50)  ',
    'RANGE_TYPE_PEDESTRIAN_[50, +inf)_LEVEL_2/AP':  'Pedestrian AP  L2 [50, +inf)',
    'RANGE_TYPE_PEDESTRIAN_[50, +inf)_LEVEL_2/APH': 'Pedestrian APH L2 [50, +inf)',
    
    'RANGE_TYPE_CYCLIST_[0, 30)_LEVEL_1/AP':     'Cyclist AP  L1 [0, 30)   ',
    'RANGE_TYPE_CYCLIST_[0, 30)_LEVEL_1/APH':    'Cyclist APH L1 [0, 30)   ',
    'RANGE_TYPE_CYCLIST_[30, 50)_LEVEL_1/AP':    'Cyclist AP  L1 [30, 50)  ',
    'RANGE_TYPE_CYCLIST_[30, 50)_LEVEL_1/APH':   'Cyclist APH L1 [30, 50)  ',
    'RANGE_TYPE_CYCLIST_[50, +inf)_LEVEL_1/AP':  'Cyclist AP  L1 [50, +inf)',
    'RANGE_TYPE_CYCLIST_[50, +inf)_LEVEL_1/APH': 'Cyclist APH L1 [50, +inf)',
    
    'RANGE_TYPE_CYCLIST_[0, 30)_LEVEL_2/AP':     'Cyclist AP  L2 [0, 30)   ',
    'RANGE_TYPE_CYCLIST_[0, 30)_LEVEL_2/APH':    'Cyclist APH L2 [0, 30)   ',
    'RANGE_TYPE_CYCLIST_[30, 50)_LEVEL_2/AP':    'Cyclist AP  L2 [30, 50)  ',
    'RANGE_TYPE_CYCLIST_[30, 50)_LEVEL_2/APH':   'Cyclist APH L2 [30, 50)  ',
    'RANGE_TYPE_CYCLIST_[50, +inf)_LEVEL_2/AP':  'Cyclist AP  L2 [50, +inf)',
    'RANGE_TYPE_CYCLIST_[50, +inf)_LEVEL_2/APH': 'Cyclist APH L2 [50, +inf)',
}

EVAL_ORDER: list = [
    'OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/AP',
    'OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/AP',
    'BREAK',
    'OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/APH',
    'OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/APH',
    'BREAK',
    'OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/AP',
    'OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/AP',
    'BREAK',
    'OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/APH',
    'OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/APH',
    'BREAK',
    'OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/AP',
    'OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/AP',
    'BREAK',
    'OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/APH',
    'OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/APH',
    'BREAK',
    'RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_1/AP',    
    'RANGE_TYPE_VEHICLE_[30, 50)_LEVEL_1/AP', 
    'RANGE_TYPE_VEHICLE_[50, +inf)_LEVEL_1/AP', 
    'RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_1/APH',   
    'RANGE_TYPE_VEHICLE_[30, 50)_LEVEL_1/APH',  
    'RANGE_TYPE_VEHICLE_[50, +inf)_LEVEL_1/APH',
    'BREAK',
    'RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_2/AP',  
    'RANGE_TYPE_VEHICLE_[30, 50)_LEVEL_2/AP',   
    'RANGE_TYPE_VEHICLE_[50, +inf)_LEVEL_2/AP',
    'RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_2/APH',    
    'RANGE_TYPE_VEHICLE_[30, 50)_LEVEL_2/APH',
    'RANGE_TYPE_VEHICLE_[50, +inf)_LEVEL_2/APH',
    'BREAK',
    'RANGE_TYPE_PEDESTRIAN_[0, 30)_LEVEL_1/AP',    
    'RANGE_TYPE_PEDESTRIAN_[30, 50)_LEVEL_1/AP', 
    'RANGE_TYPE_PEDESTRIAN_[50, +inf)_LEVEL_1/AP', 
    'RANGE_TYPE_PEDESTRIAN_[0, 30)_LEVEL_1/APH',   
    'RANGE_TYPE_PEDESTRIAN_[30, 50)_LEVEL_1/APH',  
    'RANGE_TYPE_PEDESTRIAN_[50, +inf)_LEVEL_1/APH',
    'BREAK',
    'RANGE_TYPE_PEDESTRIAN_[0, 30)_LEVEL_2/AP',  
    'RANGE_TYPE_PEDESTRIAN_[30, 50)_LEVEL_2/AP',   
    'RANGE_TYPE_PEDESTRIAN_[50, +inf)_LEVEL_2/AP',
    'RANGE_TYPE_PEDESTRIAN_[0, 30)_LEVEL_2/APH',    
    'RANGE_TYPE_PEDESTRIAN_[30, 50)_LEVEL_2/APH',
    'RANGE_TYPE_PEDESTRIAN_[50, +inf)_LEVEL_2/APH',
    'BREAK',
    'RANGE_TYPE_CYCLIST_[0, 30)_LEVEL_1/AP',    
    'RANGE_TYPE_CYCLIST_[30, 50)_LEVEL_1/AP', 
    'RANGE_TYPE_CYCLIST_[50, +inf)_LEVEL_1/AP', 
    'RANGE_TYPE_CYCLIST_[0, 30)_LEVEL_1/APH',   
    'RANGE_TYPE_CYCLIST_[30, 50)_LEVEL_1/APH',  
    'RANGE_TYPE_CYCLIST_[50, +inf)_LEVEL_1/APH',
    'BREAK',
    'RANGE_TYPE_CYCLIST_[0, 30)_LEVEL_2/AP',  
    'RANGE_TYPE_CYCLIST_[30, 50)_LEVEL_2/AP',   
    'RANGE_TYPE_CYCLIST_[50, +inf)_LEVEL_2/AP',
    'RANGE_TYPE_CYCLIST_[0, 30)_LEVEL_2/APH',    
    'RANGE_TYPE_CYCLIST_[30, 50)_LEVEL_2/APH',
    'RANGE_TYPE_CYCLIST_[50, +inf)_LEVEL_2/APH',
]


def print_eval_log(ap_dict, logger):
    for key in EVAL_ORDER:
        if key in ap_dict:
            logger.info(f"{EVAL_MAPPING[key]}: {ap_dict[key][0]*100:0.2f}")
        elif key == 'BREAK':
            logger.info('_' * 40)