import logging
import pickle
import copy
import random
import torch

import numpy as np
from PIL import Image, ImageFont, ImageDraw

def flatten(xss):
    return [x for xs in xss for x in xs]

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_logger(name):
    logger = logging.getLogger(name)
    return logger


def print_separator(logger, separator='_', length=80):
    logger.info(f"{separator * length}")
    logger.info("")


def build_number_file_path(dir_path, number, postfix='.pkl', n_zeros=4):
    file_path = dir_path / f"{number:0{n_zeros}d}{postfix}"
    return file_path


def check_and_create_dir(dir_path):
    requires_mk_dir = not dir_path.exists()
    if requires_mk_dir:
        dir_path.mkdir(parents=True, exist_ok=True)
    return requires_mk_dir

def dfs(matrix, visited, i, j, current_group):
    if i < 0 or i >= len(matrix) or j < 0 or j >= len(matrix[0]) or visited[i][j] or matrix[i][j] == 0:
        return

    visited[i][j] = True
    current_group.add((i, j))

    # Gehe zu allen verbundenen Nachbarn
    for x in range(len(matrix)):
        if matrix[x][j] > 0 and not visited[x][j]:
            dfs(matrix, visited, x, j, current_group)

    for y in range(len(matrix[0])):
        if matrix[i][y] > 0 and not visited[i][y]:
            dfs(matrix, visited, i, y, current_group)

def extract_groups(matrix):
    n, m = len(matrix), len(matrix[0])
    visited = [[False] * m for _ in range(n)]
    groups = []

    for i in range(n):
        for j in range(m):
            if matrix[i][j] > 0 and not visited[i][j]:
                current_group = set()
                dfs(matrix, visited, i, j, current_group)
                groups.append(current_group)

    return groups

def angle_between_vectors(v1, v2):
    cos = v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos = np.clip(cos, -0.9999, 0.9999)
    return np.rad2deg(np.arccos(cos))
            
def interpolate_boundin_boxes(boxes, indices, length):
    boxes_new = np.zeros((length, 7))
    for i in range(6):
        boxes_new[..., i] = np.interp(np.arange(length), indices, boxes[..., i])
    cos_x_new = np.interp(np.arange(length), indices, np.cos(boxes[..., 6]))
    sin_y_new = np.interp(np.arange(length), indices, np.sin(boxes[..., 6]))
    boxes_new[..., 6] = np.arctan2(sin_y_new, cos_x_new)
    return boxes_new