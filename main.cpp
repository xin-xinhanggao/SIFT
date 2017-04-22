#include <opencv2/nonfree/features2d.hpp>

#include "SiftHelper.hpp"

int main() {
	Mat match_out;
	match2img("left.jpeg", "right.jpeg", match_out);
	imshow("MATCH", match_out);
	imwrite("MATCH.jpg", match_out);

	waitKey();
	return 0;
}