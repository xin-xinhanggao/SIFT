#ifndef __SIFT__
#define __SIFT__

#include <opencv2/opencv.hpp>
#include <vector>
#include <queue>

using namespace cv;
using namespace std;

#define SIFT_INIT_SIGMA 0.5
#define SIFT_IMG_BORDER 5
#define SIFT_MAX_INTERP_STEPS 5

#define SIFT_ORI_HIST_BINS 36
#define SIFT_ORI_SIG_FCTR 1.5
#define SIFT_ORI_RADIUS 3.0 * SIFT_ORI_SIG_FCTR
#define SIFT_ORI_SMOOTH_PASSES 2
#define SIFT_ORI_PEAK_RATIO 0.99

#define SIFT_DESCR_SCL_FCTR 3.0
#define SIFT_DESCR_MAG_THR 0.2
#define SIFT_INT_DESCR_FCTR 512.0

#define SIFT_KEYPOINT_DIAMETER 2.0

struct Feature {
	float __x;
	float __y;
	float __ori;
	float __contrast;
	float __scl;
	vector<double> __descriptor;

	int __r;
	int __c;
	int __interval;
	int __octave;
	int __idx;

	float __sub_interval;
	float __scl_octave;
};

static void __feats2KeyPoints(const vector<Feature> &feats, vector<KeyPoint> &keypoints);

static void __featsVec2Mat(const vector<Feature> &feats, Mat &mat);

static Mat __createInitImg(const Mat &img, bool img_dbl, double sigma);

static void __buildGaussPyramid(const Mat &base, vector<Mat> &gaussian_pyramid, int octaves, int intervals, double sigma);

static void __buildDogPyramid(const vector<Mat> &gaussian_pyramid, vector<Mat> &dog_pyramid, int octaves, int intervals);

static void __scaleSpaceExtrema(const vector<Mat> &dog_pyramid, vector<Feature> &feats, int octaves, int intervals,
                                double contrast_thres, int curvature_thres);

static bool __isExtremum(const vector<Mat> &dog_pyramid, int idx, int r, int c);

static bool __interpExtremum(const vector<Mat> &dog_pyramid, Feature &feat, int idx, int r, int c, int intervals, double contrast_thres);

static void __interpStep(const vector<Mat> &dog_pyramid, int idx, int r, int c, double &xi, double &xr, double &xc);

static Mat __derivative(const vector<Mat> &dog_pyramid, int idx, int r, int c);

static Mat __hessian(const vector<Mat> &dog_pyramid, int idx, int r, int c);

static double __interpContrast(const vector<Mat> &dog_pyramid, int idx, int r, int c, double xi, double xr, double xc);

static bool __isTooEdgeLike(const Mat &dog, int r, int c, int curvature_thres);

static void __calcFeatureScales(vector<Feature> &feats, double sigma, int intervals);

static void __adjustForImgDbl(vector<Feature> &feats);

static void __calcFeatureOris(vector<Feature> &feats, const vector<Mat> &gaussian_pyramid, int layer_per_octave);

static void __oriHist(const Mat &gaussian, vector<double> &hist, int r, int c, int rad, double sigma);

static bool __calcGradMagOri(const Mat &gaussian, int r, int c, double &mag, double &ori);

static void __smoothOriHist(vector<double> &hist);

static void __addGoodOriFeatures(queue<Feature> &feat_queue, const vector<double> &hist, double mag_thres, const Feature &feat);

static void __computeDescriptors(vector<Feature> &feats, const vector<Mat> &gaussian_pyramid, int layer_per_octave, int d, int n);

static void __descriptorHist(const Mat &gaussian, vector<double> &hist, int r, int c, double ori, double scl, int d, int n);

static void __interpHistEntry(vector<double> &hist, double rbin, double cbin, double obin, double mag, int d, int n);

static void __hist2Descriptor(const vector<double> &hist, Feature &feat, int d, int n);

static void __normalizeDescriptor(vector<double> &descriptor);


#endif