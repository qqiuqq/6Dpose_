#----------------------------------------------------------------------------------#
基于densefusion改写的点云特征提取器

tool.train_test3为主函数
lib.network_pointnet3为模型结构

除点云提取器外，该文件夹还包括使用实例模块和点云特征提取器进行测试的.py文件

#----------------------------------------------------------------------------------#
# train：
# ./experiments/scripts/train_ycb_test3.sh
# ./experiments/scripts/train_linemod_point3.sh

# ycb训练好的.pth模型放在train_models/ycb/test3内
# linemod训练好的.pth模型放在train_models/linemod内
#----------------------------------------------------------------------------------#
# test（加入icp算法）：
#  ./experiments/scripts/eval_ycb_icp.sh
#  ./experiments/scripts/eval_linemod_icp.sh

#----------------------------------------------------------------------------------#
