#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include <vector>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
	String path = "ai.png";
	Mat img = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
	Mat imgout;

	vector<KeyPoint> keypoints1(10);

	Ptr<ORB> fd = ORB::create(10);

	fd->detect(img, keypoints1);
	drawKeypoints(img, keypoints1, imgout);

	imshow("ai", imgout);
	waitKey(0);
	return 0;
}
