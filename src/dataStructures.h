#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <opencv2/core.hpp>

// Represents the available sensor information at the same time instance
struct DataFrame {
    cv::Mat cameraImg; // The camera image
    std::vector<cv::KeyPoint> keypoints; // The Keypoints of the camera image
    cv::Mat descriptors; // keypoints descriptors
    std::vector<cv::DMatch> kptMatches; // Matches between the previous and current frame. Connects this data frame to the the next data frame in time
};

#endif /* dataStructures_h */