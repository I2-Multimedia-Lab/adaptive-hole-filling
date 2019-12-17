#include"Binary.h"
#include<opencv2\opencv.hpp>
#include<highgui.h>
#include"Canny.h"

int tmain(int argc, char* argv[])
{
	//对边缘进行膨胀操作
	Mat img,edge,dilaedge;
	for (int frame_num = 0; frame_num < 100; frame_num++) {
		String img_frame_path;
		img_frame_path = "C:\\Data\\testsource\\3DVideos-distrib\\MSR3DVideo-Breakdancers\\cam2\\depth-cam2-f0" + cv::format("%.2d", frame_num) + ".png";
		img = imread(img_frame_path, 1);
		Canny(&img, &edge);
		//edge = imread("edge\\edge" + cv::format("%.2d", frame_num) + ".png", 0);
		morphologyEx(edge, dilaedge, MORPH_DILATE, Mat(3, 3, CV_8U), Point(-1, -1), 2);
		//imshow("edge", edge);
		//imshow("dila", dilaedge);
		//waitKey();
		imwrite("morphological dilation edge\\Breakdancers\\cam2\\edge" + cv::format("%.2d", frame_num) + ".png", dilaedge);
	}
return 0;
}


	//提取边缘
	//Mat img;
	//Mat edg(768, 1024, CV_8UC1, Scalar(0));
	////for (int frame_num = 1; frame_num < 100; frame_num++) {
	//	String img_frame_path;
	//	/*img_frame_path = "C:\\Data\\testsource\\3DVideos-distrib\\MSR3DVideo-ballet\\cam0\\depth-cam0-f0" + cv::format("%.2d", frame_num) + ".png";*/
	//	img_frame_path = "C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\3Dwarp\\ballet\\0-1\\vdep00.bmp";
	//	img = imread(img_frame_path, 1);
	//	Canny(&img, &edg);
	//	imshow("edg", edg);
	//	imwrite("C:\\Users\\a\\Documents\\Visual Studio 2015\\Projects\\adapInpaint\\adapInpaint\\depedg.png", edg);
	//	waitKey();
	//	//imwrite("C:\\Users\\a\\Documents\\Visual Studio 2015\\Projects\\adapInpaint\\fill\\b+gmmedg.png", edg);
	//	//imwrite("edge\\edge" + cv::format("%.2d", frame_num) + ".png", edg);
	////}



