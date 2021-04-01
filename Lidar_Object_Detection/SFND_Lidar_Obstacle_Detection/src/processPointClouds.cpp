// PCL lib Functions for processing point clouds 

#include "processPointClouds.h"
#include "utils.cpp"
#include <iostream>
using namespace std;


//constructor:
template<typename PointT>
ProcessPointClouds<PointT>::ProcessPointClouds() {}


//de-constructor:
template<typename PointT>
ProcessPointClouds<PointT>::~ProcessPointClouds() {}


template<typename PointT>
void ProcessPointClouds<PointT>::numPoints(typename pcl::PointCloud<PointT>::Ptr cloud)
{
    std::cout << cloud->points.size() << std::endl;
}


template<typename PointT>
// typename here means pcl::PointCloud<PointT>::Ptr is a type, rather than a value
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::FilterCloud(typename pcl::PointCloud<PointT>::Ptr cloud, float filterRes, Eigen::Vector4f minPoint, Eigen::Vector4f maxPoint)
{

    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    // TODO:: Fill in the function to do voxel grid point reduction and region based filtering

    // pass through voxel grid 
    typename pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT>);
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize (filterRes, filterRes, filterRes);
    vg.filter (*cloud_filtered);

    // pass through min max point region
    typename pcl::PointCloud<PointT>::Ptr cloud_region (new pcl::PointCloud<PointT>);
    pcl::CropBox<PointT> region(true);
    region.setMin(minPoint);
    region.setMax(maxPoint);
    region.setInputCloud(cloud_filtered);
    region.filter(*cloud_region);

    // remove roof
    vector<int> indices;
    pcl::CropBox<PointT> roof(true);
    region.setMin(Eigen::Vector4f(-1.5, -1.7, -1, 1));
    region.setMax(Eigen::Vector4f(2.6, 1.7, -0.4, 1));
    region.setInputCloud(cloud_region);
    region.filter(indices);

    // get inliers
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    for (int pt : indices) {
        inliers->indices.push_back(pt);
    }

    // extract indices
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(cloud_region);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*cloud_region);


    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "filtering took " << elapsedTime.count() << " milliseconds" << std::endl;

    return cloud_region;

}


template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SeparateClouds(pcl::PointIndices::Ptr inliers, typename pcl::PointCloud<PointT>::Ptr cloud) 
{
    // TODO: Create two new point clouds, one cloud with obstacles and other with segmented plane
    typename pcl::PointCloud<PointT>::Ptr cloud_plane (new pcl::PointCloud<PointT> ());
    typename pcl::PointCloud<PointT>::Ptr cloud_obstacle (new pcl::PointCloud<PointT> ());

    // put in the inliers into the plane:
    for (int ind : inliers->indices) {
        cloud_plane->points.push_back(cloud->points[ind]);
    }

    // remove inliers from the cloud, left the obstacles
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud (cloud);
    extract.setIndices (inliers);
    extract.setNegative (true);
    extract.filter (*cloud_obstacle);

    // std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult(cloud, cloud);
    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult(cloud_obstacle, cloud_plane);
    return segResult;
}


template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SegmentPlane(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold)
{
    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    // modified...
    // Fill in this function to find inliers for the cloud.
    pcl::PointIndices::Ptr inliers {new pcl::PointIndices};
    pcl::ModelCoefficients::Ptr coefficients {new pcl::ModelCoefficients};


    // // TODO:: comment out the following block
    // pcl::SACSegmentation<PointT> seg;
    // seg.setOptimizeCoefficients (true);
    // seg.setModelType (pcl::SACMODEL_PLANE); // where the point is located
    // seg.setMethodType (pcl::SAC_RANSAC);// random sample consensus
    // seg.setDistanceThreshold (distanceThreshold);
    // seg.setMaxIterations(maxIterations);

    // // perform segmentation on the cloud structure
    // seg.setInputCloud (cloud);
    // // inliers points to the points in the plane found by Ransac
    // seg.segment (*inliers, *coefficients);
    
    // TODO: uncomment the following code to see my customized RANSAC performance.
    //       set the maxIteration to be at least 500 to see a good performance.
    //       performance time is long, consider evaluating other functions first.
    //       the last param of the seg.Ransac function is the number of points used to find plane.

    RANSAC<PointT> seg;
    std::unordered_set<int> inliers_index = seg.Ransac(cloud, maxIterations, distanceThreshold, 7); // use 300 points, 50 iterations and distanceTol of 0.2
    for (int index: inliers_index) {
        inliers->indices.push_back(index);
    }


    // edge case
    if (inliers->indices.size() == 0) {
        cout<<"could not perform segmentation on the plane."<<endl;
    } 

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult = SeparateClouds(inliers,cloud);
    return segResult;
}


template<typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::Clustering(typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance, int minSize, int maxSize)
{

    // Time clustering process
    auto startTime = std::chrono::steady_clock::now();

    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;

    // TODO:: comment out the following block

    // // Fill in the function to perform euclidean clustering to group detected obstacles
    // typename pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    // tree->setInputCloud (cloud);
    // std::vector<pcl::PointIndices> cluster_indices;
    // pcl::EuclideanClusterExtraction<PointT> ec;
    // ec.setClusterTolerance (clusterTolerance); // 2cm
    // ec.setMinClusterSize (minSize);
    // ec.setMaxClusterSize (maxSize);
    // ec.setSearchMethod (tree);
    // ec.setInputCloud (cloud);
    // ec.extract (cluster_indices);

    // int j = 0;
    // for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    // {
    //     typename pcl::PointCloud<PointT>::Ptr cloud_cluster (new pcl::PointCloud<PointT>);
    //     for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
    //         cloud_cluster->push_back ((*cloud)[*pit]); //*
    //     cloud_cluster->width = cloud_cluster->size ();
    //     cloud_cluster->height = 1;
    //     cloud_cluster->is_dense = true;
    //     clusters.push_back(cloud_cluster);
    // }


    // TODO: uncomment the following code to see my customized RANSAC performance. (uncomment the ) 
    //       the last param of the seg.Ransac function is the number of points used to find plane.

    KdTree* tree2 = new KdTree;
    vector<vector<float>> points;
    EUCLIDEAN clusterUnit;

    int ptind = 0;
    for (PointT &pt: cloud->points) {
        std::vector<float> point = {pt.x, pt.y, pt.z}; 
        points.push_back(point); 
        tree2->insert(point, ptind); 
        ++ptind;
    }
    std::vector<std::vector<int>> clusters_indices2 = clusterUnit.euclideanCluster(points, tree2, clusterTolerance, minSize, maxSize);
    for (vector<int> it : clusters_indices2) {
        typename pcl::PointCloud<PointT>::Ptr cloud_cluster (new pcl::PointCloud<PointT>);
        for (int pit : it) {
            // cloud_cluster->push_back((*cloud)[it]);
            cloud_cluster->push_back(cloud->points[pit]);
        }
        cloud_cluster->width = cloud_cluster->size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;
        clusters.push_back(cloud_cluster);
    }
    

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "clustering took " << elapsedTime.count() << " milliseconds and found " << clusters.size() << " clusters" << std::endl;

    return clusters;
}


template<typename PointT>
Box ProcessPointClouds<PointT>::BoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster)
{

    // Find bounding box for one of the clusters
    PointT minPoint, maxPoint;
    pcl::getMinMax3D(*cluster, minPoint, maxPoint);

    Box box;
    box.x_min = minPoint.x;
    box.y_min = minPoint.y;
    box.z_min = minPoint.z;
    box.x_max = maxPoint.x;
    box.y_max = maxPoint.y;
    box.z_max = maxPoint.z;

    return box;
}


// use PCA to calculate oriented bounding box
template<typename PointT>
BoxQ ProcessPointClouds<PointT>::BoundingBoxQ(typename pcl::PointCloud<PointT>::Ptr cluster)
{
    //http://codextechnicanum.blogspot.com/2015/04/find-minimum-oriented-bounding-box-of.html
    // Find bounding box for one of the clusters with z axis locked.

    Eigen::Vector4f OriginalpcaCentroid;
    pcl::compute3DCentroid(*cluster, OriginalpcaCentroid);

    // make a deepcopy
    typename pcl::PointCloud<PointT>::Ptr cluster2 (new pcl::PointCloud<PointT>);
    pcl::copyPointCloud(*cluster, *cluster2);

    for (PointT &p : cluster2->points) p.z = 0; // flatten the z axis
    // Compute principal directions
    Eigen::Vector4f pcaCentroid;
    pcl::compute3DCentroid(*cluster2, pcaCentroid);
    Eigen::Matrix3f covariance;
    computeCovarianceMatrixNormalized(*cluster2, pcaCentroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
    Eigen::Vector3f eigenValuesPCA = eigen_solver.eigenvalues();
    eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));  /// This line is necessary for proper orientation in some cases. The numbers come out the same without it, but
                                                                                    ///    the signs are different and the box doesn't get correctly oriented in some cases.
    // Transform the original cloud to the origin where the principal components correspond to the axes.
    Eigen::Matrix4f projectionTransform(Eigen::Matrix4f::Identity());
    projectionTransform.block<3,3>(0,0) = eigenVectorsPCA.transpose(); // rotation 
    projectionTransform.block<3,1>(0,3) = -1.f * (projectionTransform.block<3,3>(0,0) * OriginalpcaCentroid.head<3>()); // translation
    typename pcl::PointCloud<PointT>::Ptr cloudPointsProjected (new pcl::PointCloud<PointT>);
    pcl::transformPointCloud(*cluster, *cloudPointsProjected, projectionTransform);
    // Get the minimum and maximum points of the transformed cloud.
    PointT minPoint, maxPoint;
    pcl::getMinMax3D(*cloudPointsProjected, minPoint, maxPoint);
    const Eigen::Vector3f meanDiagonal = 0.5f*(maxPoint.getVector3fMap() + minPoint.getVector3fMap());

    BoxQ box;
    // dimensions
    box.cube_length = maxPoint.x - minPoint.x;
    box.cube_width = maxPoint.y - minPoint.y;
    box.cube_height = maxPoint.z - minPoint.z;
    // rotation
    box.bboxQuaternion = eigenVectorsPCA;
    // translation
    box.bboxTransform = eigenVectorsPCA * meanDiagonal + OriginalpcaCentroid.head<3>();
    return box;
}



template<typename PointT>
void ProcessPointClouds<PointT>::savePcd(typename pcl::PointCloud<PointT>::Ptr cloud, std::string file)
{
    pcl::io::savePCDFileASCII (file, *cloud);
    std::cerr << "Saved " << cloud->points.size () << " data points to "+file << std::endl;
}


template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::loadPcd(std::string file)
{

    typename pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);

    if (pcl::io::loadPCDFile<PointT> (file, *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file \n");
    }
    std::cerr << "Loaded " << cloud->points.size () << " data points from "+file << std::endl;

    return cloud;
}


template<typename PointT>
std::vector<boost::filesystem::path> ProcessPointClouds<PointT>::streamPcd(std::string dataPath)
{

    std::vector<boost::filesystem::path> paths(boost::filesystem::directory_iterator{dataPath}, boost::filesystem::directory_iterator{});

    // sort files in accending order so playback is chronological
    sort(paths.begin(), paths.end());

    return paths;

}