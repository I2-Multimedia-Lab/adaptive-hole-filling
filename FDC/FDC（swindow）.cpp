#include"Binary.h"
#include<opencv2\opencv.hpp>
#include<highgui.h>
#include <windows.h>

using namespace cv;

void depfunc(Mat& img, Mat& FDC)
{
	for (int x = 0; x < img.rows; x++) {
		for (int y = 0; y < img.cols; y++) {

			if (img.at<uchar>(x, y) >= 250) {
				//printf("(%d,%d)", x, y);
				/*test.at<Vec3b>(x, y)[0]=0;
				test.at<Vec3b>(x, y)[1] = 0;
				test.at<Vec3b>(x, y)[2] = 0;*/
				img.at<uchar>(x, y) = FDC.at<uchar>(x, y);
			}
		}
	}
}
void func(Mat& img, Mat& FDC)
{
	Scalar t = img.at<Vec3b>(0, 0);
	for (int x = 0; x < img.rows; x++) {
		for (int y = 0; y < img.cols; y++) {
			Scalar s = img.at<Vec3b>(x, y);
			//if (img.at<Vec3b>(x, y)[1] - img.at<Vec3b>(x, y)[0] >= 60 || img.at<Vec3b>(x, y)[1] - img.at<Vec3b>(x, y)[2] >= 60)
			if (img.at<Vec3b>(x, y)[0] <= 5 || img.at<Vec3b>(x, y)[1] <= 5 || img.at<Vec3b>(x, y)[2] <= 5) {
				//printf("(%d,%d)", x, y);
				/*test.at<Vec3b>(x, y)[0]=0;
				test.at<Vec3b>(x, y)[1] = 0;
				test.at<Vec3b>(x, y)[2] = 0;*/
				img.at<Vec3b>(x, y) = FDC.at<Vec3b>(x, y);
			}
		}
	}
}
int fsmain(int argc, char* argv[])
{
	Mat img, depth1,edge,nextedg;
	Mat empty(768, 1024,CV_8UC1,Scalar(0));
	static int lback[768][1024] = { 0 };
	Mat findep(768, 1024, CV_8UC1, Scalar(0));
	Mat finback(768, 1024, CV_8UC3, Scalar(0));
	
	for (int frame_num =00; frame_num < 34; frame_num++) {
		String img_frame_path, dep_frame_path, dep_frame_path2;
		img_frame_path = "C:\\Data\\testsource\\3DVideos-distrib\\MSR3DVideo-Breakdancers\\cam2\\color-cam2-f0" + cv::format("%.2d", frame_num) + ".jpg";
		dep_frame_path2 = "C:\\Data\\testsource\\3DVideos-distrib\\MSR3DVideo-Breakdancers\\cam2\\depth-cam2-f0" + cv::format("%.2d", frame_num) + ".png";
		dep_frame_path = "C:\\test\\Breakdancers\\histo\\cam2\\eauframe" + cv::format("%.2d", frame_num) + ".png";//经过直方图处理的直方图 
	
		img = imread(img_frame_path, IMREAD_UNCHANGED);
		depth1 = imread(dep_frame_path, 0);
		edge = imread("morphological dilation edge\\Breakdancers\\cam2\\edge" + cv::format("%.2d", frame_num) + ".png", 0);//经过canny和膨胀操作得到的边缘 

		Mat depth(depth1.rows, depth1.cols, CV_8UC3);
		Mat depth2= imread(dep_frame_path2, 0);
		vector<Mat> channels;
		for (int i = 0; i<3; i++)
		{
			channels.push_back(depth1);
		}
		merge(channels, depth);
		Mat depkmeans(depth.rows, depth.cols, CV_8UC1);
		Binary(&depth, &depkmeans);
		uchar *p, *pdepth;
		int i, j = 0;
		for (i = 0; i < img.rows; i++)
		{
			p = img.ptr<uchar>(i);//行指针
			pdepth = depkmeans.ptr<uchar>(i);
			for (j = 0; j < img.cols; j++)
			{
				if (abs(pdepth[j] - depkmeans.at<uchar>(0, 0)) <= 5 && lback[i][j] == 0) {//由深度判断前景背景，将第一帧的背景复制到初始背景中
					if (edge.at<uchar>(i, j) == 255) {
						finback.at<Vec3b>(i, j)[0] = 255;
						finback.at<Vec3b>(i, j)[1] = 255;
						finback.at<Vec3b>(i, j)[2] = 255;
						findep.at<uchar>(i, j) = 255;
					}
					else {
						finback.at<Vec3b>(i, j) = img.at<Vec3b>(i, j);
						findep.at<uchar>(i, j) = depth2.at<uchar>(i,j);
						lback[i][j] = 1;
					}
					//findep.at<uchar>(i, j) = depth2.at<uchar>(i, j);
					
				}
				
			}
		}
		imwrite("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\edge-result(FDC)\\Breakdancers\\cam2\\win1\\findep" + cv::format("%.2d", frame_num) + ".jpg", findep);
		imwrite("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\edge-result(FDC)\\Breakdancers\\cam2\\win1\\kmeans" + cv::format("%.2d", frame_num) + ".jpg", depkmeans);
		imwrite("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\edge-result(FDC)\\Breakdancers\\cam2\\win1\\dilafinback" + cv::format("%.2d", frame_num) + ".jpg", finback);
	}

	//每一个区间最后结果中没有补上的背景用区间第一帧来填 
	//Mat fback=imread("C:\\Data\\testsource\\3DVideos-distrib\\MSR3DVideo-Breakdancers\\cam2\\color-cam2-f000.jpg");
	//Mat fbackdep = imread("C:\\Data\\testsource\\3DVideos-distrib\\MSR3DVideo-Breakdancers\\cam2\\depth-cam2-f000.png",0);
	//Mat back = imread("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\edge-result(FDC)\\Breakdancers\\cam2\\win1\\dilafinback33.jpg");
	//Mat backdep = imread("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\edge-result(FDC)\\Breakdancers\\cam2\\win1\\findep33.jpg",0);
	//for (int i = 0; i < back.rows; i++)
	//	for (int j = 0; j < back.cols; j++)
	//	{
	//		if (backdep.at<uchar>(i, j) >= 250) {
	//			back.at<Vec3b>(i, j) = fback.at<Vec3b>(i, j);
	//			backdep.at<uchar>(i, j) = fbackdep.at<uchar>(i, j);
	//		}
	//	}
	//imwrite("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\edge-result(FDC)\\Breakdancers\\cam2\\win1\\winback1.jpg",back);
	//imwrite("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\edge-result(FDC)\\Breakdancers\\cam2\\win1\\windep1.png", backdep);


	//对像素值不为0的点，比较第一帧深度值，取深度值小的点，若效果不佳，利用FDC生成深度图，取3个窗口最终深度图深度值低的点
	//Mat img1 = imread("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\edge-result(FDC)\\Breakdancers\\cam2\\win1\\winback1.jpg");
	//Mat img2 = imread("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\edge-result(FDC)\\Breakdancers\\cam2\\win2\\winback2.jpg");
	//Mat img3 = imread("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\edge-result(FDC)\\Breakdancers\\cam2\\win3\\winback3.jpg");
	//Mat depth1 = imread("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\edge-result(FDC)\\Breakdancers\\cam2\\win1\\windep1.png",0);//加0，否则默认按rgb方式读取
	//Mat depth2 = imread("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\edge-result(FDC)\\Breakdancers\\cam2\\win2\\windep2.png",0);
	//Mat depth3 = imread("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\edge-result(FDC)\\Breakdancers\\cam2\\win3\\windep3.png",0);
	//int channels = depth1.channels();
	//Mat gray1, gray2, gray3;
	//imshow("depth1", depth1);
	//imshow("depth2", depth2);
	//imshow("depth3", depth3);
	//Mat wdep(img1.rows, img1.cols, CV_8UC1, Scalar(255, 255, 255));
	//Mat wback(img1.rows,img1.cols,CV_8UC3,Scalar(0,0,0));
	//img1.copyTo(wback);
	//depth1.copyTo(wdep);
	//imshow("wdep", wdep);
	//imshow("wback", wback);
	//cvtColor(img1, gray1, CV_RGB2GRAY);
	//cvtColor(img2, gray2, CV_RGB2GRAY);
	//cvtColor(img3, gray3, CV_RGB2GRAY);
	//for (int i = 0; i < img1.rows; i++) {
	//	for (int j = 0; j < img1.cols; j++) {
	//		int i1 = gray1.at<uchar>(i, j);
	//		int i2 = gray2.at<uchar>(i, j);
	//		int i3 = gray3.at<uchar>(i, j);
	//		int mindep=255,label=0;
	//		/*if(i1&&i2&&i3){
	//			
	//		}*/
	//		int depth;
	//		bool flag = false;
	//		for (int k = 1; k <= 3; k++) {
	//			if (k == 1) depth = depth1.at<uchar>(i, j);
	//			if (k == 2) depth = depth2.at<uchar>(i, j);
	//			if (k == 3) depth = depth3.at<uchar>(i, j);
	//			if (depth <=5) {
	//				printf("(%d,%d)at%d", i, j, k);
	//				depth = 256;
	//			}

	//			if (depth <=mindep) {
	//				mindep = depth ;
	//				label = k;
	//			}
	//			
	//		}
	//		if (label == 1) {
	//			wdep.at<uchar>(i, j) = depth1.at<uchar>(i, j);
	//			wback.at<Vec3b>(i, j) = img1.at<Vec3b>(i, j);
	//		}
	//		
	//		if (label == 2&& abs(depth1.at<uchar>(i, j)- depth2.at<uchar>(i, j))>=10) {
	//			wdep.at<uchar>(i, j) = depth2.at<uchar>(i, j);
	//			wback.at<Vec3b>(i, j) = img2.at<Vec3b>(i, j);
	//		}
	//		if (label == 3&& abs(depth1.at<uchar>(i, j) - depth2.at<uchar>(i, j)) >= 10) {
	//			wdep.at<uchar>(i, j) = depth3.at<uchar>(i, j);
	//			wback.at<Vec3b>(i, j) = img3.at<Vec3b>(i, j);
	//		}
	//	}
	//}
	//namedWindow("result", 1);
	//imwrite("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\edge-result(FDC)\\Breakdancers\\cam2\\wdep.jpg", wdep);
	//imwrite("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\edge-result(FDC)\\Breakdancers\\cam2\\wback.jpg", wback);
	//imshow("result", wback);
	//imshow("result1", wdep);
	
	return 0;
}
