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
	float x;
	float y;
	float ori;
	float contrast;
	float scl;
	vector<double> descriptor;

	int r;
	int c;
	int interval;
	int octave;
	int idx;

	float sub_interval;
	float scl_octave;
};

static void feats2KeyPoints(const vector<Feature> &feats, vector<KeyPoint> &keypoints);

static void featsVec2Mat(const vector<Feature> &feats, Mat &mat);

static Mat createInitImg(const Mat &img, bool img_dbl, double sigma);

static void buildGaussPyramid(const Mat &base, vector<Mat> &gaussian_pyramid, int octaves, int intervals, double sigma);

static void buildDogPyramid(const vector<Mat> &gaussian_pyramid, vector<Mat> &dog_pyramid, int octaves, int intervals);

static void scaleSpaceExtrema(const vector<Mat> &dog_pyramid, vector<Feature> &feats, int octaves, int intervals,
                                double contrast_thres, int curvature_thres);

static bool isExtremum(const vector<Mat> &dog_pyramid, int idx, int r, int c);

static bool interpExtremum(const vector<Mat> &dog_pyramid, Feature &feat, int idx, int r, int c, int intervals, double contrast_thres);

static void interpStep(const vector<Mat> &dog_pyramid, int idx, int r, int c, double &xi, double &xr, double &xc);

static Mat derivative(const vector<Mat> &dog_pyramid, int idx, int r, int c);

static Mat hessian(const vector<Mat> &dog_pyramid, int idx, int r, int c);

static double interpContrast(const vector<Mat> &dog_pyramid, int idx, int r, int c, double xi, double xr, double xc);

static bool isTooEdgeLike(const Mat &dog, int r, int c, int curvature_thres);

static void calcFeatureScales(vector<Feature> &feats, double sigma, int intervals);

static void adjustForImgDbl(vector<Feature> &feats);

static void calcFeatureOris(vector<Feature> &feats, const vector<Mat> &gaussian_pyramid, int layer_per_octave);

static void oriHist(const Mat &gaussian, vector<double> &hist, int r, int c, int rad, double sigma);

static bool calcGradMagOri(const Mat &gaussian, int r, int c, double &mag, double &ori);

static void smoothOriHist(vector<double> &hist);

static void addGoodOriFeatures(queue<Feature> &feat_queue, const vector<double> &hist, double mag_thres, const Feature &feat);

static void computeDescriptors(vector<Feature> &feats, const vector<Mat> &gaussian_pyramid, int layer_per_octave, int d, int n);

static void descriptorHist(const Mat &gaussian, vector<double> &hist, int r, int c, double ori, double scl, int d, int n);

static void interpHistEntry(vector<double> &hist, double rbin, double cbin, double obin, double mag, int d, int n);

static void hist2Descriptor(const vector<double> &hist, Feature &feat, int d, int n);

static void normalizeDescriptor(vector<double> &descriptor);


#endif