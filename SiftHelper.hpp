#ifndef SIFT_HELPER
#define SIFT_HELPER

#include "SIFT.hpp"

/**
 * [wrapper function for extractSiftFeatures]
 *
 * @param img        [source image, rgb only]
 * @param keypoints  [vector for store sift feature point]
 * @param descriptor [Mat for store sift descriptor, each row for a keypoint]
 */
void siftWrapper(const Mat &img, vector<KeyPoint> &keypoints, Mat &descriptor);

void kmeans(vector<Point2f> bad_feature, vector<Point2f> good_feature, int cluster_num, Mat &feature);
/**
 * [match 2 image]
 *
 * @param img1   [1st image's path]
 * @param img2   [2nd image's path]
 * @param output [match result]
 */
void match2img(const char *img1, const char *img2, Mat &output, Mat &feature);

#endif