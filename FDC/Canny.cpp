#include "Canny.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
using namespace cv;



void Canny(Mat *input, Mat *output)
{
	Mat srcGray;	
	cvtColor(*input, srcGray, CV_BGR2GRAY);
	//¸ßË¹ÂË²¨	
	GaussianBlur(srcGray, srcGray, Size(3, 3),0, 0, BORDER_DEFAULT);	
	//Canny¼ì²â	
	int edgeThresh =100;	
	Mat Canny_result;	
	Canny(*input, Canny_result, edgeThresh, edgeThresh * 3, 3);
	//imshow("Canny_result", Canny_result);
	*output = Canny_result;
}

