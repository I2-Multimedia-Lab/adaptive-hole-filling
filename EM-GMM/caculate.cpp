#include "caculate.h"
#include<stdio.h>
#include <opencv2/highgui/highgui.hpp >
#define PI 3.141592
using namespace cv;
float pro(float x, float u, float s)
{
	float result;
	result = 1 / (sqrt(2 * PI)*s);
	result=result*exp(-(pow(x - u, 2) / (2 * pow(s, 2))));
	return result;
}
int cmain() {
	printf("%f", pro(0, 0, 1));
	waitKey(0);
	system("pause");
	return 0;
}