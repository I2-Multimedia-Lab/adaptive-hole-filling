#include <opencv2/opencv.hpp> 
#include "opencv\highgui.h"
using namespace cv;
using namespace std;

void Canny(Mat *input, Mat *output);
void adpCanny(Mat *input, Mat *output, double edgeThresh);
