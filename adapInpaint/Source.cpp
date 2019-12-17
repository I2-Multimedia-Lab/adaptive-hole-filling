#include "inpainting.h"
#include <opencv2/opencv.hpp> 
#include "opencv\highgui.h"
using namespace cv;
using namespace std;

int main() {
	for (int f = 0; f < 1; f++) {
		String virstring = "C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\3Dwarp\\Breakdancers\\02-03\\vimg"+format("%.2d",f)+".jpg";
		String depstring = "C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\3Dwarp\\Breakdancers\\02-03\\vdep" + format("%.2d", f) + ".png";
		//char* dep = "C:\\Data\\testsource\\3DVideos-distrib\\MSR3DVideo-Ballet\\cam1\\depth-cam1-f000.png";
		char* fdc = "C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\3Dwarp\\Breakdancers\\02-03\\fdc\\vimg.bmp";
		//char* fdc = "C:\\Data\\testsource\\experiencePictrue\\0-1\\fdc\\fdcvimg00.jpg";
		char* gmm = "C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\3Dwarp\\Breakdancers\\02-03\\em\\vimg.bmp";
		char* vir = (char*)virstring.c_str();
		char* dep = (char*)depstring.c_str();
		//char* gmmdep = "C:\\Data\\testsource\\experiencePictrue\\0-1\\fdc\\virtruel_Depth_image01.jpg";
		inpainting test(vir, dep, fdc, gmm,f);
		test.process();
	}
	

	return 0;
}
