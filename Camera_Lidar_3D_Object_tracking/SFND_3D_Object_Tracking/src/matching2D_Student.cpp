
#include <numeric>
#include "matching2D.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // ... Student task for Flann matching method
        matcher = cv::FlannBasedMatcher::create();
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        // ... implement knn matching
        vector<vector<cv::DMatch>> knn_matches;
        int k = 2;
        matcher->knnMatch(descSource, descRef, knn_matches, k);

        // distance ratio
        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {

            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor BRIEF, ORB, FREAK, AKAZE, SIFT
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0) 
    {   
        int bytes = 32;
        bool use_orientation = false;
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, use_orientation);
    } 
    else if (descriptorType.compare("ORB") == 0)
    {
        // the program currently has an insufficient memory error when called upon ORB.
        // error: (-4:Insufficient memory) Failed to allocate 66866457856 bytes in function 'OutOfMemoryError'
        // you could try with a better machine
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        bool ori_normalized = true;
        bool scale_normalized = true;
        float float_pattern_scale = 0.22F;
        int nOctaves = 4;
        extractor = cv::xfeatures2d::FREAK::create(ori_normalized, scale_normalized, float_pattern_scale, nOctaves);
    }
    else if (descriptorType.compare("AKAZE") == 0) 
    {   
        // AKAZE must use it's own keypoints. 
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {   
        // when you encounter error: https://answers.opencv.org/question/10046/feature-2d-feature-matching-fails-with-assert-statcpp/
        // Since SIFT and SURF return a detectorType() of CV32F (=float) you cannot use any Hamming-distance as matcher. 
        // Hamming-distance works only for binary feature-types like ORB, FREAK, ... 
        // I guess it would be possible if you'd convert them to CV8U. 
        // Afaik for SIFT this should be fairly easy since they are actually already normalized for the range 0-255 (double check that before you do it). 
        // However, I doubt that this is useful at all, use L2 norm for those and you should be fine.
        int 	nfeatures = 500;
        int 	nOctaveLayers = 3;
        double 	contrastThreshold = 0.04;
        double 	edgeThreshold = 10;
        double 	sigma = 1.6 ;
        extractor = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
        img.convertTo(img, CV_8U);
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}


// Following codes are for feature detectors, including predified ShiTomasi, Harris, Fast, Brisk, Orb, Akaze, SIFT
// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis) {
    int blockSize = 10;
    int apertureSize = 3;
    double k = 0.12;
    cv::Mat dst = cv::Mat::zeros( img.size(), CV_32FC1 );

    
    // apply corner detection
    double t = (double)cv::getTickCount();
    cv::cornerHarris( img, dst, blockSize, apertureSize, k );

    // add corners to result vector
    int thresh = 200;
    cv::Mat dst_norm, dst_norm_scaled;
    cv::normalize( dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
    cv::convertScaleAbs( dst_norm, dst_norm_scaled );
    for( int i = 0; i < dst_norm.rows ; i++ )
    {
        for( int j = 0; j < dst_norm.cols; j++ )
        {
            if( (int) dst_norm.at<float>(i,j) > thresh )
            {
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(j, i);
                newKeyPoint.size = blockSize;
                keypoints.push_back(newKeyPoint);
                cv::circle( dst_norm_scaled, cv::Point(j,i), 5,  cv::Scalar(0), 2, 8, 0 );
            }
        }
    }
    
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;


    if (bVis) {
        const char* corners_window = "Harris Coener Detector Results";
        cv::namedWindow( corners_window );
        cv::imshow( corners_window, dst_norm_scaled );
    }


}

void detKeypointsFast(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis) {
    int thresh = 10;
    
    // apply corner detection
    double t = (double)cv::getTickCount();    
    cv::Ptr<cv::FeatureDetector> fast = cv::FastFeatureDetector::create(thresh); 
    fast->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "FAST detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis) {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "FAST Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsBrisk(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis) {
    int threshold = 30;        // FAST/AGAST detection threshold score.
    int octaves = 3;           // detection octaves (use 0 to do single scale)
    float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
    
    // apply corner detection
    double t = (double)cv::getTickCount();    
    cv::Ptr<cv::FeatureDetector> brisk = cv::BRISK::create(threshold, octaves, patternScale); 
    brisk->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "BRISK detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis) {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "BRISK Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsOrb(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis) {
    int 	nfeatures = 5000;
    float 	scaleFactor = 1.2f;
    int 	nlevels = 8;
    int 	edgeThreshold = 31;
    int 	firstLevel = 0;
    int 	WTA_K = 2;
    int 	scoreType = cv::ORB::HARRIS_SCORE;
    int 	patchSize = 31;
    int 	fastThreshold = 20 ;
    // apply corner detection
    double t = (double)cv::getTickCount();    
    cv::Ptr<cv::FeatureDetector> ORB = cv::ORB::create(nfeatures, scaleFactor); 
    ORB->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "ORB detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis) {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "ORB Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsAkaze(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis) {
    int 	descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
    int 	descriptor_size = 0;
    int 	descriptor_channels = 3;
    float 	threshold = 0.001f;
    int 	nOctaves = 4;
    int 	nOctaveLayers = 4;
    int 	diffusivity = cv::KAZE::DIFF_PM_G2;
    
    // apply corner detection
    double t = (double)cv::getTickCount();    
    cv::Ptr<cv::FeatureDetector> akaze = cv::AKAZE::create(); 
    akaze->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "akaze detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis) {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "akaze Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsSIFT(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis) {
    /* this part is not implemented because my opencv version contains xfeature2D, but in the hpp file
    it does not contain SIFT. Only SURF is given. It states that OPENCV_ENABLE_NONFREE needs to be configured when building opencv.
    Because going through the installation of OpenCV gets to consume a large amount of time, so this part is not done. IF
    you have a opencv that is older, you are welcome to uncomment the followin code and try.
    */
    int 	nfeatures = 5000;
    int 	nOctaveLayers = 3;
    double 	contrastThreshold = 0.04;
    double 	edgeThreshold = 10;
    double 	sigma = 1.6 ;
    // apply corner detection
    double t = (double)cv::getTickCount();    
    cv::Ptr<cv::Feature2D> sift = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma); 
    img.convertTo(img, CV_8U);
    sift->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "SIFT detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis) {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "SIFT Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}