
## Requirements

* Python 3.5
* PyTorch 1.0
* torchvision 0.2.2.post3
* PIL
* scipy
* numpy
* pyyaml
* logging
* cffi
* matplotlib
* Cython
* CUDA 9.0/10.0

```bash
基于densefusion改写的点云特征提取器
tool.train_test3为主函数
lib.network_pointnet3为模型结构

除点云提取器外，该文件夹还包括使用实例模块和点云特征提取器进行测试的.py文件
```

## Train
1. To train
   ```bash
   ./experiments/scripts/train_ycb_test3.sh
   ./experiments/scripts/train_linemod_point3
   ```
   
2. To eval 测试内容加入ICP算法
   ```bash
   ./experiments/scripts/eval_ycb_icp.sh 
   ./experiments/scripts/eval_linemod_icp.sh
   ```