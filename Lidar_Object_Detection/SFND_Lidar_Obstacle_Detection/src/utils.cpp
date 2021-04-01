#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/transforms.h>
#include <iostream> 
#include <string>  
#include <vector>
#include <ctime>
#include <chrono>
#include <unordered_set>

#include <random>
#include <string>
#include <iterator>
#include <experimental/algorithm>

#include <pcl/cloud_iterator.h>

#include <pcl/pcl_base.h>
#include <pcl/PointIndices.h>
#include <pcl/ModelCoefficients.h>
#include <Eigen/Core>
using namespace std;


// util classes for segmentation and clustering (customized)
template <typename PointT>
class RANSAC : public pcl::PCLBase<PointT>
{ 
public: 
    int numSamples;
    vector<float> planeGeneralForm_3pt(PointT pt1, PointT pt2, PointT pt3) {
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

    vector<float> planeGeneralForm_npt(Eigen::MatrixXf A) {
        /* Ax + By + Cz + D = 0 (normal vector dot point on plane + 1 = 0)
        * solve system of equation
        * using svd
        * |  |  |  |
        * x, y, z, 1 * [A, B, C, D] = 0 
        * |  |  |  | 
        */
        // get a point
        Eigen::RowVector3f pt0 = A.row(0);

        // subtract the centroid
        float mean_x = A.col(0).mean();
        float mean_y = A.col(1).mean();
        float mean_z = A.col(2).mean();
        A.col(0) = A.col(0).array() - mean_x;
        A.col(1) = A.col(1).array() - mean_y;
        A.col(2) = A.col(2).array() - mean_z;

        //svd
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::MatrixXf VT = svd.matrixV();
        Eigen::MatrixXf x = VT.row(VT.rows() - 1);
        //Eigen::MatrixXf x = VT.row(1);

        // Ax + By + Cz = d, d = ax0 + by0 + cz0
        vector<float> res = {x(0,0), x(0,1), x(0,2)};
        res.push_back( -(res[0]*pt0(0) + res[1]*pt0(1) + res[2]*pt0(2)) );
        return res;
    }


    float distance_point_to_plane_3D(vector<float> param, float norm, PointT pt3) {
        // distance between a point and a line is |Ax0 + By0 + Cz0 + D| / sqrt(A**2 + B**2 + C**2)
        float dist = abs(param[0] * pt3.x + param[1] * pt3.y + param[2] * pt3.z + param[3]) / norm;
        return dist;
    }
    
    std::unordered_set<int> Ransac(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceTol, int num_samples)
    {
        // Randomly sample subset and fit line
        // Measure distance between every point and fitted line
        // If distance is smaller than threshold count it as inlier
        // Return indicies of inliers from fitted line with most inliers
        // use 20 points to fit a plane
        /*
        * Ax = 0 where x is <x,y,z> for normal vector
        * 
        */
        numSamples = num_samples;
        std::unordered_set<int> sample_inds;
        std::unordered_set<int> inliersResult;
        std::unordered_set<int> last;
        int numpts = cloud->size();
        int respts = last.size();

        Eigen::MatrixXf samples(numSamples, 3);
        
        // For max iterations 
        for (int i = 0; i<maxIterations; ++i){
            srand(i);
            last.clear(); 
            sample_inds.clear();
            samples.resize(0, 0);
            Eigen::MatrixXf samples(numSamples, 3);
            
            for (int k = 0 ; k < num_samples; ++k) {
                int s = rand() % numpts;
                sample_inds.insert(s);
                Eigen::RowVector3f v;
                v << cloud->at(s).x, cloud->at(s).y, cloud->at(s).z;
                samples.row(k) = v; 
            } 
            // plane equation from points
            vector<float> plane = planeGeneralForm_npt(samples);
            float plane_norm = sqrt(plane[0]*plane[0] + plane[1]*plane[1] + plane[2]*plane[2]);
            for (int j = 0; j < numpts; ++j) {
                if (sample_inds.count(j)) continue;
                // make the point inlier if the point has a dist smaller than the distance tolerence
                if (distance_point_to_plane_3D(plane, plane_norm, cloud->at(j)) <= distanceTol) last.insert(j);
            }
            if (last.size() > respts) {inliersResult = last; respts = last.size();}

            // If the fraction of the number of inliers over the total number points in the set
            // exceeds a predefined threshold τ , re-estimate the model parameters using all the
            // identified inliers and terminate
            // if (respts / numpts > 0.1) {
            //     cout<<"restimate"<<endl;
            //     Eigen::MatrixXf samples_inliers(respts, 3);
            //     Eigen::RowVector3f v2;
            //     int mm = 0;
            //     for (int s : last) {
            //         v2 << cloud->at(s).x, cloud->at(s).y, cloud->at(s).z;
            //         samples_inliers.row(mm) = v2; 
            //         ++mm;
            //     }
            //     plane = planeGeneralForm_npt(samples_inliers);
            //     last.clear();
            //     for (int j = 0; j < numpts; ++j) {
            //         if (sample_inds.count(j)) continue;
            //         if (distance_point_to_plane_3D(plane, cloud->at(j)) <= distanceTol) last.insert(j);
            //     }
            //     if (last.size() > respts) {inliersResult = last; respts = last.size();}
            // }
        }
    return inliersResult;
    }
};

// Structure to represent node of kd tree
struct Node
{
	std::vector<float> point;
	int id;
	Node* left;
	Node* right;

	Node(std::vector<float> arr, int setId)
	:	point(arr), id(setId), left(NULL), right(NULL)
	{}
};

struct KdTree
{
	Node* root;

	KdTree()
	: root(NULL)
	{}

	void insert(std::vector<float> point, int id)
	{
		// TODO: Fill in this function to insert a new point into the tree
		// the function should create a new node and place correctly with in the root 
		inserthelper(root, 0, point, id);
	}

	void inserthelper(Node* &node, int level, std::vector<float> point, int id) {
		// note: must pass tree by reference, kdtree evaluates insertion alternatively by level.
		if(node == NULL) node = new Node(point, id);
		else {
			uint cd = level % 3;
			if(point[cd] < node->point[cd]) inserthelper(node->left, level + 1, point, id);
			else inserthelper(node->right, level + 1, point, id);
		}
	}

	// return a list of point ids in the tree that are within distance of target
	std::vector<int> search(std::vector<float> target, float distanceTol)
	{
		std::vector<int> ids;
		searchhelper(target, root, 0, distanceTol, ids);
		return ids;
	}

	void searchhelper(std::vector<float> target, Node* node, int depth, float distanceTol, std::vector<int> &ids) {
		// note: must pass the ids by reference
		if (node != NULL) {
			// if node is within range of the taeget, calculate distance
			if ( (node->point[0] >= (target[0]-distanceTol) && node->point[0] <= (target[0] + distanceTol)) && 
			     (node->point[1] >= (target[1]-distanceTol) && node->point[1] <= (target[1] + distanceTol)) &&
                 (node->point[2] >= (target[2]-distanceTol) && node->point[2] <= (target[2] + distanceTol)) ) {
					float distance = sqrt((node->point[0] - target[0]) * (node->point[0] - target[0]) + 
										  (node->point[1] - target[1]) * (node->point[1] - target[1]) + 
                                          (node->point[2] - target[2]) * (node->point[2] - target[2]));
					if (distance <= distanceTol) ids.push_back(node->id);
					}
			// check across boundary, recurse on next nodes in the KdTree
			if (target[depth%3] - distanceTol < node->point[depth%3]) searchhelper(target, node->left, depth+1, distanceTol, ids);
			if (target[depth%3] + distanceTol > node->point[depth%3]) searchhelper(target, node->right, depth+1, distanceTol, ids);
		}
	}
};

class EUCLIDEAN {
public:
    void proximity(int targetId, const std::vector<std::vector<float>>& points, std::vector<int> &cluster, std::unordered_set<int> &visited, KdTree* tree, float distanceTol) {
        visited.insert(targetId);
        cluster.push_back(targetId); 
        std::vector<int> ids = tree->search(points[targetId], distanceTol);
        for (int id : ids) { if (!visited.count(id)) proximity(id, points, cluster, visited, tree, distanceTol);}
    }

    std::vector<std::vector<int>> euclideanCluster(const std::vector<std::vector<float>>& points, KdTree* tree, float distanceTol, int minsize, int maxsize){
        std::vector<std::vector<int>> clusters;
        std::unordered_set<int> visited;
        for (int id = 0; id < points.size(); ++id) {
            if (!visited.count(id)) {
                std::vector<int> cluster;
                proximity(id, points, cluster, visited, tree, distanceTol);
                if (cluster.size() > minsize && cluster.size() < maxsize) clusters.push_back(cluster);}}
        return clusters;
    }
};

