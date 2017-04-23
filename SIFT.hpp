#ifndef SIFT
#define SIFT

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

void extractSiftFeatures(const Mat &img, vector<KeyPoint> &keypoints, Mat &descriptor, int intervals = 3,
                         double sigma = 1.6, double contrast_thres = 0.04,
                         int curvature_thres = 10, bool img_dbl = true,
                         int descr_width = 4, int descr_hist_bins = 8);

#endif