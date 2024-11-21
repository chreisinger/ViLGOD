import os
import hydra
import gc
import torch
import pickle

import numpy as np

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pathlib import Path

from src.utils.clip_utils import ClipWrapper
from src.utils import common_utils, cluster_utils, eval_utils
from src.vilgod.zero_shot_detector import ZeroShotDetector


def resolve_tuple(*args):
    return tuple(args)

OmegaConf.register_new_resolver('as_tuple', resolve_tuple)
OmegaConf.register_new_resolver('join', lambda x : '_'.join(x))
OmegaConf.register_new_resolver('format_split_join', lambda x : '_'.join(x.format('').split(' ')[:-1]))

@hydra.main(version_base=None, config_path="configs", config_name="preprocessing")
def main(cfg : DictConfig) -> None:
    # init
    logger = common_utils.create_logger(__name__)
    logger.info("Working directory: {}".format(os.getcwd()))
    
    if cfg.get('random_seed', False):
        common_utils.set_random_seed(cfg.random_seed)
        
    # dataset
    dataset = instantiate(cfg.dataset_class, logger=logger, training=True,
                          start_sequence=cfg.start_sequence, end_sequence=cfg.end_sequence)
    # adjust dataset
    if cfg.split != 'train':
        dataset.set_split(cfg.split)
    dataset.training = False
    # clustering model
    cluster_model = None
    if 'spatial_clustering' in cfg.pipeline_active or 'spatio_temporal_clustering' in cfg.pipeline_active:
        cluster_model = cluster_utils.init(cfg.preprocessor.clustering.model)
    # clip model
    clip_model = None
    if 'classification' in cfg.pipeline_active:
        clip_model = ClipWrapper(cfg.preprocessor.clip, cfg.paths.clip_model, device='cuda')
    
    result_path = Path(cfg.paths.results) / cfg.results_folder / '_'.join(cfg.pipeline_active) 
    common_utils.check_and_create_dir(result_path)
    
    # log pipeline
    logger.info('_' * 40)
    logger.info('Pipeline:')
    t_idx = 1
    for task in cfg.pipeline:
        if task['name'] in cfg.pipeline_active:
            logger.info(f"[{t_idx}] {task['name']}")
            t_idx += 1
    logger.info('_' * 40)
            
    indices = []
    detection_results = []
    
    result_data = None
    if cfg.load_detection_results:
        if Path(cfg.result_path).exists():
            with Path(cfg.result_path).open('rb') as f:
                result_data = pickle.load(f)

    # main loop
    for sequence_name in dataset.next_sequence():
        if result_data is not None:
            continue
        
        result_file = result_path / f'{sequence_name}.pkl'
        indices_file = result_path / f'{sequence_name}_indices.pkl'
        
        if cfg.use_cached_results and 'evaluate_sequence' in cfg.pipeline_active:                  
            if result_file.exists():
                with result_file.open('rb') as f:
                    detection_results.extend(pickle.load(f))
                with indices_file.open('rb') as f:
                    indices.extend(pickle.load(f))
            
            if result_file.exists():
                continue

        zsd = ZeroShotDetector(dataset, sequence_name, 
                               cfg=cfg, logger=logger,
                               cluster_model=cluster_model,
                               clip_model=clip_model)
        zsd.process()
        detection_results.extend(zsd.detection_3d_result_list)
        indices.extend(zsd.dataset.sequence_indices)
        
        if 'evaluate_sequence' in cfg.pipeline_active:    
            # save detection results    
            with open(result_file, 'wb') as f:
                pickle.dump(zsd.detection_3d_result_list, f)
            with open(indices_file, 'wb') as f:
                pickle.dump(zsd.dataset.sequence_indices, f)
                
        del zsd
        gc.collect()
        torch.cuda.empty_cache()
    
    if result_data is not None:
        detection_results = result_data
    
    # eval all sequences
    if len(detection_results) > 0:
        det3d_args = [pp for pp in cfg.pipeline if pp.name=='evaluate_sequence'][0]['args']
        det3d_cfg = det3d_args['detection_3d']
        
        logger.info('_' * 100)
        logger.info('Evaluate all Sequences - Detection 3D')
        logger.info('_' * 100)
        ap_dict = dataset.evaluation(detection_results, class_names=dataset.class_names, 
                                     indices=indices, eval_cfg=cfg.eval_cfg, 
                                     class_agnostic=det3d_cfg['class_agnostic'], 
                                     eval_range=det3d_args['eval_range'], 
                                     bev=det3d_cfg['bev'], 
                                     moving=det3d_args['moving'], static=det3d_args['static'],
                                     score_thresh=det3d_cfg['score_thresh'],
                                     sampling_rate=det3d_cfg['sampling_rate'])
        # log results
        eval_utils.print_eval_log(ap_dict, logger)
        logger.info('_' * 100)

if __name__ == "__main__":
    main()
