#ifndef SIFT_HELPER
#define SIFT_HELPER

#include "SIFT.hpp"

void siftWrapper(const Mat &img, vector<KeyPoint> &keypoints, Mat &descriptor);

void kmeans(vector<Point2f> bad_feature, vector<Point2f> good_feature, int cluster_num, Mat &feature);

void match2img(const char *img1, const char *img2, Mat &output, Mat &feature);

#endif