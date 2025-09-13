### Matterport3D Paper
- https://niessner.github.io/Matterport/
---

### Matterport3D Visualization
- https://aspis.cmpt.sfu.ca/scene-toolkit/scans/matterport3d/houses
---

### Evaluation Metrics
- SR（Success Rate，成功率）：如果代理最终停止的位置与目标点之间的 **geodesic 距离小于 3 米**，则认为该次导航任务成功。  
- OSR（Oracle Success Rate，神谕成功率）：如果代理在导航路径上的 **任意一个位置** 与目标点之间的 **geodesic 距离小于 3 米**，则判定为 Oracle 成功。  
- NE（Navigation Error，导航误差）：指代理最终停止的位置与目标点之间的 **geodesic 距离**，用于衡量导航终点与目标点的接近程度。
- TL（Trajectory Length，轨迹长度）：代理在整个导航过程中实际移动的路径总长度。
- SPL（Success weighted by Path Length，路径加权成功率）：s*(l/max(p,l))
    - \( S \)：任务是否成功（成功为 1，失败为 0）；
    - \( l \)：起点到目标的 **geodesic 最短距离**；
    - \( p \)：代理实际移动的路径长度。

---




### Record
main_v2.py
- 旧版本：1. 「想象 n 步走全程（例如想象 17 步，行动 24 步）」2. 很多任务走的长度不够
- 新版本：1. 「想象几步走几步」2. 微调进度审查的prompt(去除多选择1 少选0 的偏好) ，并且最大可行走步数变成 子任务数量+2（旧版本为1）



experiment 1 (2025-8-10): -> main_v1.py
- num_trajectory: 4      
- imagine_resolution: 320 
- imagine_step: 6 
- total length of trajectory -> 24 step

experiment 2 (2025-8-12):   -> main_v2.py
- num_trajectory: 4      
- imagine_resolution: 320 
- imagine_step: 9 
- Imagining step = action step

experiment 2 (2025-8-14):   -> main_v2.py
- num_trajectory: 4      
- imagine_resolution: 320 
- imagine_step: 12 
- Imagining step = action step