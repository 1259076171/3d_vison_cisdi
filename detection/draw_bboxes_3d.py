import torch
import numpy as np

from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes

points = np.fromfile('./000008.bin', dtype=np.float32)
arr = np.load('./pose_det_results_list.npy',allow_pickle=True)
arr = torch.from_numpy(arr)
points = points.reshape(-1, 4)
visualizer = Det3DLocalVisualizer()
# set point cloud in visualizer
visualizer.set_points(points)
# print(len(arr))


bboxes_3d = LiDARInstance3DBoxes(arr)
# Draw 3D bboxes
visualizer.draw_bboxes_3d(bboxes_3d)
visualizer.show()