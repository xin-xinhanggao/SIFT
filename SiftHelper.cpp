#include <opencv2/nonfree/features2d.hpp>

#include "SiftHelper.hpp"
#include <set>
#include <iostream>

void siftWrapper(const Mat &img, vector<KeyPoint> &keypoints, Mat &descriptor) {
	Mat gray_img;
	cvtColor(img, gray_img, CV_BGR2GRAY);
	gray_img.convertTo(gray_img, CV_32FC1, 1.0 / 255);

	extractSiftFeatures(gray_img, keypoints, descriptor);
}

void match2img(const char *img1_p, const char *img2_p, Mat &output, Mat &feature) {
	Mat img1 = imread(img1_p);
	vector<KeyPoint> keypoints1;
	Mat descriptor1;

	Mat img2 = imread(img2_p);
	feature = imread(img2_p);

	vector<KeyPoint> keypoints2;
	Mat descriptor2;

	siftWrapper(img1, keypoints1, descriptor1);
	siftWrapper(img2, keypoints2, descriptor2);

	BFMatcher matcher;
	vector<vector<DMatch> > matches;
	matcher.knnMatch(descriptor1, descriptor2, matches, 2);

	std::cout<<matches.size()<<std::endl;

	set<int> good_index;
	vector<DMatch> good_matches;
	for (size_t i = 0; i < matches.size(); i ++) {
		if (matches[i][0].distance < 0.8 * matches[i][1].distance) {
			good_matches.push_back(matches[i][0]);
			good_index.insert(matches[i][0].trainIdx);
		}
	}

	for(int i = 0; i < keypoints1.size(); i++)
	{
		if(good_index.count(i) == 0)
		{
			//bad match features here
			int row = floor(keypoints1[i].pt.y);
			int col = floor(keypoints1[i].pt.x);
			feature.at<Vec3b>(row, col) = Vec3b(255, 255, 255);
		}
	}
	
	std::cout<<good_matches.size()<<std::endl;

	drawMatches(img1, keypoints1, img2, keypoints2, good_matches, output, Scalar::all(-1), Scalar::all(-1),
	            vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
}