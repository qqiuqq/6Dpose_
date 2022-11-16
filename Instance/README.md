```
基于PNV3D代码修改的实例分割模块
Instance.instance_module为模块内容
```
# train：
# cd instance_module
# ./train_ycb.sh
# 训练好的.tar文件放在train_log/ycb/checkpoint内
# 训练好的.pth模型放在train_log/ycb内，训练好点云特征提取器后直接联合起来进行测试