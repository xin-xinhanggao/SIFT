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
	set<int> centerinit; //center init
	while(centerinit.size() < cluster_num)
	{
		int point_index = rand() % bad_feature.size();
		centerinit.insert(point_index);
	}

	int centerindex = 0;
	for(set<int>::iterator it = centerinit.begin(); it != centerinit.end(); it++)
	{
		center[centerindex++] = bad_feature[*it];
	}


    Point2f *newcenter = new Point2f[cluster_num]; // the new center for iteration
    double *totalweight = new double[cluster_num];
    for(int iter = 0; iter < 1000; iter++)
    {
    	for(int i = 0; i < cluster_num; i++)
    	{
    		totalweight[i] = 0;
    		newcenter[i].x = newcenter[i].y = 0;
    	}


    	for(vector<Point2f>::iterator it = bad_feature.begin(); it != bad_feature.end(); it++)
    	{
    		double distance = (center[0].x - it->x) * (center[0].x - it->x) + (center[0].y - it->y) * (center[0].y - it->y);
    		int cluster_index = 0;
    		for(int i = 1; i < cluster_num; i++)
    		{
    			double d = (center[i].x - it->x) * (center[i].x - it->x) + (center[i].y - it->y) * (center[i].y - it->y);
    			if(d < distance)
    			{
    				distance = d;
    				cluster_index = i;
    			}
    		}
    		double weight = 1.0 / (1 + sqrtf(distance));
    		totalweight[cluster_index] += weight;
    		newcenter[cluster_index].x += it->x * weight;
    		newcenter[cluster_index].y += it->y * weight;
    	}

    	//modify the original center
    	for(int i = 0; i < cluster_num; i++)
    	{
    		center[i].x = newcenter[i].x * 1.0 / totalweight[i];
    		center[i].y = newcenter[i].y * 1.0 / totalweight[i];
    	}
    }

    //distribute the point to the cluster according to the final center
    for(vector<Point2f>::iterator it = bad_feature.begin(); it != bad_feature.end(); it++)
	{
		double distance = (center[0].x - it->x) * (center[0].x - it->x) + (center[0].y - it->y) * (center[0].y - it->y);;
		int cluster_index = 0;
		for(int i = 1; i < cluster_num; i++)
		{
			double d = (center[i].x - it->x) * (center[i].x - it->x) + (center[i].y - it->y) * (center[i].y - it->y);
			if(d < distance)
			{
				distance = d;
				cluster_index = i;
			}
		}
		cluster[cluster_index].push_back(*it);
	}

	set<int> cluster_size;
	for(int i = 0; i < cluster_num; i++)
		cluster_size.insert(cluster[i].size());

	int choose_num  = 4;

	int num = 0;
	int *max_size = new int[choose_num];
	for(set<int>::reverse_iterator it = cluster_size.rbegin(); it != cluster_size.rend(); it++)
	{
		if(num < choose_num)
		{
			max_size[num] = *it;
		}
		else 
			break;
		num++;
	}

	cluster_size.clear();
	for(int i = 0; i < choose_num; i++)
		cluster_size.insert(max_size[i]);

	for(int i = 0; i < cluster_num; i++)
	{
		Vec3b color(0,255,0);
		std::cout<<i<<" "<<cluster[i].size()<<std::endl;
		if(cluster[i].size() > 0)
		{
			double lx = 1000000.0, ly = 1000000.0;
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

				int row = it->x;
				int col = it->y;
				feature.at<Vec3b>(row, col) = color;
			}

			if(cluster_size.count(cluster[i].size()) == 1)
			{
				int lxc = lx;
				int mxc = mx;
				int lyc = ly;
				int myc = my;

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
			bad_feature.push_back(Point2f(keypoints1[i].pt.y, keypoints1[i].pt.x));
		}
	}

	std::cout<<good_matches.size()<<std::endl;

	int cluster_num = 16;
	kmeans(bad_feature, cluster_num, feature);

	drawMatches(img1, keypoints1, img2, keypoints2, good_matches, output, Scalar::all(-1), Scalar::all(-1),
	            vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
}