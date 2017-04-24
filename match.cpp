#include <opencv2/nonfree/features2d.hpp>

#include "match.hpp"
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

void depictfeature(vector<Point2f> bad_feature, vector<Point2f> good_feature, int cluster_num, Mat &feature)
{
	Vec3b color(0,255,0);
	for(vector<Point2f>::iterator it = bad_feature.begin(); it != bad_feature.end(); it++)
	{
		int row = it->x;
		int col = it->y;
		feature.at<Vec3b>(row, col) = color;
	}
}

void kmeans(vector<Point2f> bad_feature, vector<Point2f> good_feature, int cluster_num, Mat &feature)
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
		Vec3b color(0,255,0);
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
		int row = it->x;
		int col = it->y;
		feature.at<Vec3b>(row, col) = color;
		if(distance < 30000)
			cluster[cluster_index].push_back(*it);
	}

	int choose_num  = 3;

	int *max_index = new int[cluster_num];
	double *weight = new double[cluster_num];
	for(int i = 0; i < cluster_num; i++)
	{
		max_index[i] = i;
		weight[i] = cluster[i].size();

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
		}

		int good_count = 0;
		for(vector<Point2f>::iterator it = good_feature.begin(); it != good_feature.end(); it++)
		{
			if(lx < it->x && mx > it->x && ly < it->y && my > it->y)
				good_count++;
		}
		weight[i] = weight[i] - 5 * good_count;
	}

	for(int i = 0; i < cluster_num; i++)
		for(int j = i + 1; j < cluster_num; j++)
		{
			if(weight[j] > weight[i])
			{
				double w = weight[j];
				weight[j] = weight[i];
				weight[i] = w;

				int m_inedx = max_index[i];
				max_index[i] = max_index[j];
				max_index[j] = m_inedx;
			}
		}

	/*
	for(int i = 0; i < cluster_num; i++)
	{
		std::cout<<max_index[i]<<" "<<cluster[max_index[i]].size()<<" "<<weight[i]<<std::endl;
	}
	std::cout<<std::endl;
	*/
		
	for(int i = 0; i < choose_num; i++)
	{
		int now_index = max_index[i];
		if(cluster[now_index].size() > 0)
		{
			double lx = 1000000.0, ly = 1000000.0;
			double mx = 0.0, my = 0.0;

			for(vector<Point2f>::iterator it = cluster[now_index].begin(); it != cluster[now_index].end(); it++)
			{
				if(it->x < lx)
					lx = it->x;

				if(it->y < ly)
					ly = it->y;

				if(it->x > mx)
					mx = it->x;

				if(it->y > my)
					my = it->y;
			}

			int lxc = lx;
			int mxc = mx;
			int lyc = ly;
			int myc = my;

			if(lyc < 100)
				lyc = 100;
			if(myc < 100)
				myc = 100;

			for(int x = lxc; x < mxc; x++)
			{
				feature.at<Vec3b>(x, lyc) = Vec3b(0,0,255);
				feature.at<Vec3b>(x, myc) = Vec3b(0,0,255);
			}

			for(int y = lyc; y < myc; y++)
			{
				feature.at<Vec3b>(lxc, y) = Vec3b(0,0,255);
				feature.at<Vec3b>(mxc, y) = Vec3b(0,0,255);
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

	set<int> good_index;
	vector<Point2f> good_feature;

	vector<DMatch> good_matches;
	for (size_t i = 0; i < matches.size(); i ++) {
		if (matches[i][0].distance < 0.8 * matches[i][1].distance) {
			good_matches.push_back(matches[i][0]);
			good_index.insert(matches[i][0].trainIdx);
			good_feature.push_back(Point2f(keypoints2[matches[i][0].trainIdx].pt.y, keypoints2[matches[i][0].trainIdx].pt.x));
		}
	}

	vector<Point2f> bad_feature;
	for(int i = 0; i < keypoints2.size(); i++)
	{
		if(good_index.count(i) == 0)
		{
			//bad match features here
			bad_feature.push_back(Point2f(keypoints2[i].pt.y, keypoints2[i].pt.x));
		}
	}

	int cluster_num = 9;
	kmeans(bad_feature, good_feature, cluster_num, feature);

	drawMatches(img1, keypoints1, img2, keypoints2, good_matches, output, Scalar::all(-1), Scalar::all(-1),
	            vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
}