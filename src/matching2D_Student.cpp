#include <numeric>
#include "matching2D.hpp"

void detKeypointsHarris (std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis) {
    // The image input is already grayscale
    int blockSize = 4; //  The neighborhood size of the Harris corner response
    int aperture = 3; // For the sobel operator
    double k = 0.04; // The free parameter in the formula R = eig1*eig2 - k * (eig1 + eig2)
    int minResponse = 100;
    cv::Mat dest = cv::Mat::zeros(img.size(), CV_32FC1); // The image type is decided according to the documentation
    cv::cornerHarris(img, dest, blockSize, aperture, k);
    
    // Normalize the image to increase contrast for better feature extract
    cv::Mat dst_norm, dst_norm_scaled;
    cv::normalize(dest, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    // Scales and calculates absolute value and converts the image to 8-bit (0-255)
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // Now what we do is calculate the local maxima in the harris response matrix and perform an NMS in a local neighborhood around each maxima
    // The keypoint vector has been given as argument
    double nmsOverlap;
    for (size_t i = 0; i < dst_norm_scaled.rows; i++) {
        for (size_t j = 0; j < dst_norm_scaled.cols; j++) {
            int response = (int)dst_norm_scaled.at<float>(i,j);
            if (response > minResponse) {

                cv::KeyPoint harrisKeypoint;
                harrisKeypoint.pt = cv::Point2f(j,i);
                harrisKeypoint.response = response; // Actual keypoint response value
                harrisKeypoint.size = 2*aperture; // Region around keypoint used to form keypoint descriptor
                // Create boolean overlap variable
                bool overlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it) {
                    // Compute the percentage overlap
                    double kptOverlap = cv::KeyPoint::overlap(harrisKeypoint, *it);
                    if (kptOverlap > nmsOverlap) {
                        overlap = true;
                        if (harrisKeypoint.response > (*it).response) {
                            *it = harrisKeypoint;
                            break;
                        }
                    }
                }

                if (!overlap) {
                    keypoints.push_back(harrisKeypoint);
                }
            }
        }
    }
    if (bVis) {
        std::string windowName = "Harris Keypoints";
        cv::namedWindow(windowName, 5);
        cv::Mat visImage = dst_norm_scaled.clone();
        cv::drawKeypoints(dst_norm_scaled, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsModern (std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis = false) {

}

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
        // ...
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        // ...
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, std::string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else
    {

        //...
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << std::endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / std::max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    std::vector<cv::Point2f> corners;
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
    std::cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << std::endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        std::string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}