# PointCloud_Classification_using_optimized_PointNet

1. Use SGD not mini-batch like PointNet

2. By using kdtree, I've solved the two weak points of the PointNet：
    1> All the point clouds used in PointNet must have the same point number. 
    2> All the point clouds used in PointNet must be uniformly sampled.
   These two requirements can not be satisfied in actual situation, which limit the application of the PointNet
