import numpy as np
import scipy
from scipy.spatial.distance import cdist
import lap
from utils.box_utils import bbox3d_overlaps_diou_dif
import torch

from cython_bbox import bbox_overlaps as bbox_ious
from utils import kalman_filter

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    cost_matrix = cost_matrix
    if isinstance(cost_matrix, torch.Tensor):
        cost_matrix = cost_matrix.numpy()
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b
            

def ious(abox, bbox):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(abox), len(bbox)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox3d_overlaps_diou_dif(
        np.ascontiguousarray(abox, dtype=np.float),
        np.ascontiguousarray(bbox, dtype=np.float)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.box() for track in atracks]
        btlbrs = [track.box() for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([np.array(track.curr_feat) for track in detections], dtype=np.float)
    track_features = np.asarray([np.array(track.smooth_feat) for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features)) # Nomalized features

    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98, mode = 'maha'):
    if cost_matrix.size == 0:
        return cost_matrix
    #gating_dim = 2 if only_position else 4
    #gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.box() for det in detections])
    for row, track in enumerate(tracks):
        if mode == 'maha':
            gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
            lambda_ = 0.98
        
        elif mode == 'euclidian':
            gating_distance = np.sqrt(((measurements - track.mean[:7])**2).sum(axis=1))

        elif mode == 'manhattan':
            gating_distance = np.abs(measurements - track.mean[:7]).sum(axis=1)

        elif mode == 'chebyshev':
            gating_distance = np.max(np.abs(measurements - track.mean[:7]), axis=1)

        elif mode == 'cosine':
            gating_distance = 1 - (measurements @ track.mean[:7]) / (np.linalg.norm(measurements, axis=1) * np.linalg.norm(track.mean[:7]))

        # cost_matrix[row] = lambda_ * cost_matrix[row] + (1-lambda_)* gating_distance
        cost_matrix[row] = gating_distance
    return cost_matrix

