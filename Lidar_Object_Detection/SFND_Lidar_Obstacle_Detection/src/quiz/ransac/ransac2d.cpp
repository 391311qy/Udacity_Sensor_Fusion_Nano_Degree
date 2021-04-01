/* \author Aaron Brown */
// Quiz on implementing simple RANSAC line fitting

#include "../../render/render.h"
#include <unordered_set>
#include "../../processPointClouds.h"
// using templates for processPointClouds so also include .cpp to help linker
#include "../../processPointClouds.cpp"

pcl::PointCloud<pcl::PointXYZ>::Ptr CreateData()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
  	// Add inliers
  	float scatter = 0.6;
  	for(int i = -5; i < 5; i++)
  	{
  		double rx = 2*(((double) rand() / (RAND_MAX))-0.5);
  		double ry = 2*(((double) rand() / (RAND_MAX))-0.5);
  		pcl::PointXYZ point;
  		point.x = i+scatter*rx;
  		point.y = i+scatter*ry;
  		point.z = 0;

  		cloud->points.push_back(point);
  	}
  	// Add outliers
  	int numOutliers = 10;
  	while(numOutliers--)
  	{
  		double rx = 2*(((double) rand() / (RAND_MAX))-0.5);
  		double ry = 2*(((double) rand() / (RAND_MAX))-0.5);
  		pcl::PointXYZ point;
  		point.x = 5*rx;
  		point.y = 5*ry;
  		point.z = 0;

  		cloud->points.push_back(point);

  	}
  	cloud->width = cloud->points.size();
  	cloud->height = 1;

  	return cloud;

}

pcl::PointCloud<pcl::PointXYZ>::Ptr CreateData3D()
{
	ProcessPointClouds<pcl::PointXYZ> pointProcessor;
	return pointProcessor.loadPcd("../../../sensors/data/pcd/simpleHighway.pcd");
}


pcl::visualization::PCLVisualizer::Ptr initScene()
{
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("2D Viewer"));
	viewer->setBackgroundColor (0, 0, 0);
  	viewer->initCameraParameters();
  	viewer->setCameraPosition(0, 0, 15, 0, 1, 0);
  	viewer->addCoordinateSystem (1.0);
  	return viewer;
}

vector<float> planeGeneralForm(pcl::PointXYZ pt1, pcl::PointXYZ pt2, pcl::PointXYZ pt3) {
	// Ax + By + Cz + D = 0 (normal vector dot point on plane + 1 = 0)
	// take cross product of v1 and v2 to get the normal vector to the plane
	//v1 \times v2 = <(y2-y1)(z3-z1)-(z2-z1)(y3-y1),v1×v2=<(y2−y1)(z3−z1)−(z2−z1)(y3−y1),
	//	(z2-z1)(x3-x1)-(x2-x1)(z3-z1),(z2−z1)(x3−x1)−(x2−x1)(z3−z1),
	//	(x2-x1)(y3-y1)-(y2-y1)(x3-x1)>(x2−x1)(y3−y1)−(y2−y1)(x3−x1)>
	vector<float> cross = {
		(pt2.y - pt1.y)*(pt3.z - pt1.z) - (pt2.z - pt1.z)*(pt3.y - pt1.y), // A = i
		(pt2.z - pt1.z)*(pt3.x - pt1.x) - (pt2.x - pt1.x)*(pt3.z - pt1.z), // B = j
		(pt2.x - pt1.x)*(pt3.y - pt1.y) - (pt2.y - pt1.y)*(pt3.x - pt1.x) // C = k
	};
	cross.push_back(-(cross[0]*pt1.x + cross[1]*pt1.y + cross[2]*pt1.z)); // D = - (ix0 + jy0 + kz0)
	return cross;
}

vector<float> lineGeneralForm(pcl::PointXYZ pt1, pcl::PointXYZ pt2) {
	// Ax + By + C = 0
	// y = kx + b 斜截式
	// kx - y + b = 0 (k = A, B = -1, C = b) 	
	float A = (float)(pt2.y - pt1.y) / (float)(pt2.x - pt1.x);
	float C = -A * pt1.x + pt1. y; 
	float B = -1.0;
	vector<float> params = {A, B, C};
	return params;
}

float distance_point_to_line_2D(vector<float> param, pcl::PointXYZ pt3) {
	// calculate distance between a point and a line in 2D.
	// in general form of the line equation: Ax + By + C,
	// distance between a point and a line is |Ax0 + By0 + C| / sqrt(A**2 + B**2)
	float dist = abs(param[0] * pt3.x + param[1] * pt3.y + param[2]) / sqrt(param[0]*param[0] + param[1]*param[1]);
	return dist;
}

float distance_point_to_plane_3D(vector<float> param, pcl::PointXYZ pt3) {
	// distance between a point and a line is |Ax0 + By0 + Cz0 + D| / sqrt(A**2 + B**2 + C**2)
	float dist = abs(param[0] * pt3.x + param[1] * pt3.y + param[2] * pt3.z + param[3]) / sqrt(param[0]*param[0] + param[1]*param[1] + param[2]*param[2]);
	return dist;
}

std::unordered_set<int> Ransac(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int maxIterations, float distanceTol)
{
	// TODO: Fill in this function
	// Randomly sample subset and fit line
	// Measure distance between every point and fitted line
	// If distance is smaller than threshold count it as inlier
	// Return indicies of inliers from fitted line with most inliers

	std::unordered_set<int> inliersResult;
	std::unordered_set<int> last;
	srand(time(NULL));
	int numpts = cloud->size();
	// For max iterations 
	for (int i = 0; i<maxIterations; ++i){
		last.clear();
		int index1 = rand() % numpts;
		int index2 = rand() % numpts;
		int index3 = rand() % numpts;
		while (index1 == index2) index2 = rand() % numpts; // avoid duplicates
		while (index3 == index2 || index3 == index1) index3 = rand() % numpts;
		// line equation between two points
		vector<float> plane = planeGeneralForm(cloud->at(index1), cloud->at(index2), cloud->at(index3));
		// calculate distance between a point and a line specified with two points
		// calculate distance between a point and a plane specified with three points (plane)
		for (int j = 0; j < numpts; ++j) {
			if (j == index1 || j == index2 || j == index3) continue;
			// make the point inlier if the point has a dist smaller than the distance tolerence
			if (distance_point_to_plane_3D(plane, cloud->at(j)) <= distanceTol) last.insert(j);
		}
		if (last.size() > inliersResult.size()) inliersResult = last;
	}

	return inliersResult;

}

int main ()
{

	// Create viewer
	pcl::visualization::PCLVisualizer::Ptr viewer = initScene();

	// Create data
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = CreateData3D();
	

	// TODO: Change the max iteration and distance tolerance arguments for Ransac function
	std::unordered_set<int> inliers = Ransac(cloud, 500, 0.5);

	pcl::PointCloud<pcl::PointXYZ>::Ptr  cloudInliers(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOutliers(new pcl::PointCloud<pcl::PointXYZ>());

	for(int index = 0; index < cloud->points.size(); index++)
	{
		pcl::PointXYZ point = cloud->points[index];
		if(inliers.count(index))
			cloudInliers->points.push_back(point);
		else
			cloudOutliers->points.push_back(point);
	}


	// Render 2D point cloud with inliers and outliers
	if(inliers.size())
	{
		renderPointCloud(viewer,cloudInliers,"inliers",Color(0,1,0));
  		renderPointCloud(viewer,cloudOutliers,"outliers",Color(1,0,0));
	}
  	else
  	{
  		renderPointCloud(viewer,cloud,"data");
  	}
	
  	while (!viewer->wasStopped ())
  	{
  	  viewer->spinOnce ();
  	}
  	
}
