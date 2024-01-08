# 3D视觉检测和分割

## 可视化工具

### 检测

```python
import mmcv
import numpy as np
from mmengine import load

from mmdet3d.visualization import Det3DLocalVisualizer

info_file = load('./000008.pkl')
points = np.fromfile('./000008.bin', dtype=np.float32)
points = points.reshape(-1, 4)[:, :3]
lidar2img = np.array(info_file['data_list'][0]['images']['CAM2']['lidar2img'], dtype=np.float32)

visualizer = Det3DLocalVisualizer()
img = mmcv.imread('./000008.png')
img = mmcv.imconvert(img, 'bgr', 'rgb')
visualizer.set_image(img)
visualizer.draw_points_on_image(points, lidar2img)
visualizer.show()
```

### 分割

```python
import numpy as np

from mmdet3d.visualization import Det3DLocalVisualizer
label_color_map = {i: np.random.rand(3) for i in range(21)}

points = np.fromfile('./000008.bin', dtype=np.float32)
points = points.reshape(-1, 4)

#predict label
with open('000008.txt', 'r') as file:
    labels = [float(line.strip()) for line in file]

labels_array = np.array(labels)

print(points.shape)
visualizer = Det3DLocalVisualizer()
mask = np.array([label_color_map[label] for label in labels_array])
points_with_mask = np.concatenate((points, mask), axis=-1)

# Draw 3D points with mask
visualizer.set_points(points, pcd_mode=2, vis_mode='add')
visualizer.draw_seg_mask(points_with_mask)
visualizer.show()
```

