#include <opencv2/nonfree/features2d.hpp>

#include "SiftHelper.hpp"
#include <set>
#include <cstdlib>
#include <ctime>
#include <iostream>

void siftWrapper(const Mat &img, vector<KeyPoint> &keypoints, Mat &descriptor) {
	Mat gray_img;
	cvtColor(img, gray_img, CV_BGR2GRAY);
	gray_img.convertTo(gray_img, CV_32FC1, 1.0 / 255);

	extractSiftFeatures(gray_img, keypoints, descriptor);
}

void kmeans(vector<Point2f> bad_feature, int cluster_num, Mat &feature)
{
	vector<Point2f> *cluster = new vector<Point2f>[cluster_num]; // initial cluster class

	Point2f *center = new Point2f[cluster_num]; // init center for all points
	int sqrtSamps = (int)sqrtf(1.f * cluster_num);
    for (int i = 0; i < sqrtSamps; ++i)
        for (int j = 0; j < sqrtSamps; ++j) 
        {
            float x = (1.f * rand() / RAND_MAX + 1.f * i) / float(sqrtSamps);
            float y = (1.f * rand() / RAND_MAX + 1.f * j) / float(sqrtSamps);
            center[i * sqrtSamps + j].x = x;
            center[i * sqrtSamps + j].y = y;
        }

    Point2f *newcenter = new Point2f[cluster_num]; // the new center for iteration
    int *newsize = new int[cluster_num];
    for(int iter = 0; iter < 0; iter++)
    {
    	for(int i = 0; i < cluster_num; i++)
    	{
    		newsize[i] = 0;
    		newcenter[i].x = newcenter[i].y = 0;
    	}


    	for(vector<Point2f>::iterator it = bad_feature.begin(); it != bad_feature.end(); it++)
    	{
    		double distance = 0;
    		int cluster_index = 0;
    		for(int i = 0; i < cluster_num; i++)
    		{
    			double d = (center[i].x - it->x) * (center[i].x - it->x) + (center[i].y - it->y) * (center[i].y - it->y);
    			if(d > distance)
    			{
    				distance = d;
    				cluster_index = i;
    			}
    		}
    		newsize[cluster_index]++;
    		newcenter[cluster_index].x += it->x;
    		newcenter[cluster_index].y += it->y;
    	}

    	//modify the original center
    	for(int i = 0; i < cluster_num; i++)
    	{
    		center[i].x = newcenter[i].x * 1.0 / newsize[i];
    		center[i].y = newcenter[i].y * 1.0 / newsize[i];
    	}
    }

    //distribute the point to the cluster according to the final center
    for(vector<Point2f>::iterator it = bad_feature.begin(); it != bad_feature.end(); it++)
	{
		double distance = 0;
		int cluster_index = 0;
		for(int i = 0; i < cluster_num; i++)
		{
			double d = (center[i].x - it->x) * (center[i].x - it->x) + (center[i].y - it->y) * (center[i].y - it->y);
			if(d > distance)
			{
				distance = d;
				cluster_index = i;
			}
		}
		cluster[cluster_index].push_back(*it);
	}

	for(int i = 0; i < cluster_num; i++)
	{
		Vec3b color(245.0 * i / cluster_num, 245.0 * i / cluster_num, 245.0 * i / cluster_num);
		std::cout<<i<<" "<<cluster[i].size()<<std::endl;
		if(cluster[i].size() > 500)
		{
			double lx = 1.0, ly = 1.0;
			double mx = 0.0, my = 0.0;

			for(vector<Point2f>::iterator it = cluster[i].begin(); it != cluster[i].end(); it++)
			{
				if(it->x < lx)
					lx = it->x;

				if(it->y < ly)
					ly = it->y;

				if(it->x > mx)
					mx = it->x;

				if(it->y > my)
					my = it->y;

				int row = feature.rows * it->x;
				int col = feature.cols * it->y;
				feature.at<Vec3b>(row, col) = color;
			}

			int lxc = lx * feature.rows;
			int mxc = mx * feature.rows;
			int lyc = ly * feature.cols;
			int myc = my * feature.cols;

			for(int i = lxc; i < mxc; i++)
			{
				feature.at<Vec3b>(i, lyc) = Vec3b(0,0,255);
				feature.at<Vec3b>(i, myc) = Vec3b(0,0,255);
			}

			for(int i = lyc; i < myc; i++)
			{
				feature.at<Vec3b>(lxc, i) = Vec3b(0,0,255);
				feature.at<Vec3b>(mxc, i) = Vec3b(0,0,255);
			}
		}
	}
}

void match2img(const char *img1_p, const char *img2_p, Mat &output, Mat &feature) {
	srand((int)time(0));
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

	vector<Point2f> bad_feature;
	for(int i = 0; i < keypoints1.size(); i++)
	{
		if(good_index.count(i) == 0)
		{
			//bad match features here
			bad_feature.push_back(Point2f(keypoints1[i].pt.y / feature.rows, keypoints1[i].pt.x / feature.cols));
		}
	}

	std::cout<<good_matches.size()<<std::endl;

	int cluster_num = 16;
	kmeans(bad_feature, cluster_num, feature);

	drawMatches(img1, keypoints1, img2, keypoints2, good_matches, output, Scalar::all(-1), Scalar::all(-1),
	            vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
}