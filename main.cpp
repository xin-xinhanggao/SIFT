#include <opencv2/nonfree/features2d.hpp>

#include "SiftHelper.hpp"

int main() {
	Mat match_out;
	Mat feature_out;
	match2img("left.jpeg", "right.jpeg", match_out, feature_out);
	imwrite("MATCH.jpg", match_out);
	imwrite("Feature.jpg", feature_out);
	return 0;
}