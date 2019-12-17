#include "Canny.h"

void Canny(Mat * input, Mat * output)
{
	//¸ßË¹ÂË²¨	
	GaussianBlur(*input, *input, Size(3, 3), 0, 0, BORDER_DEFAULT);
	//Canny¼ì²â	
	int edgeThresh = 100;
	Mat Canny_result;
	Canny(*input, Canny_result, edgeThresh, edgeThresh * 1.3, 3);
	//imshow("Canny_result", Canny_result);
	*output = Canny_result;
}
void adpCanny(Mat * input, Mat * output, double edgeThresh)
{
	Mat Canny_result;

	Canny(*input, Canny_result, edgeThresh*0.5, edgeThresh, 3);
	//imshow("Canny_result", Canny_result);
	*output = Canny_result;
}
