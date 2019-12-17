#include<opencv2/highgui.hpp>
#include<opencv2/core.hpp>
#include<opencv2/opencv.hpp>
#include "ClassGMM.h"
using namespace std;
using namespace cv;
const int width = 1024;
const int height = 768;
const int T = 0.7;

int emmain() {



	//Mat mergeImage(768,1024, CV_8UC3);
	//std::vector<Mat> channels;
	//Mat bback=imread("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\EM\\BookArrival\\cam08\\Bback01.jpg",0);//蓝通道	
	//Mat gback = imread("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\EM\\BookArrival\\cam08\\Gback01.jpg",0);//绿通道	
	//Mat rback = imread("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\EM\\BookArrival\\cam08\\Rback01.jpg", 0);//红通道	
	//channels.push_back(bback);

	//channels.push_back(gback);

	//channels.push_back(rback);
	//merge(channels, mergeImage);//合并	
	//imwrite("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\EM\\BookArrival\\cam08\\rgbback.jpg", mergeImage);

	
	int* bg_bw = (int*)malloc(sizeof(int)*width*height);
	double* rank = (double*)malloc(sizeof(double) * 1 * 3);
	Mat back(height, width, CV_8UC1);


	ClassGMM* GMMO = (ClassGMM*)malloc(sizeof(ClassGMM)*width *height);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			GMMO[i*width + j] = ClassGMM();
			GMMO[i*width+j].clear();
		}
	}
	
	
	//样本赋值
	int numPos = 99;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			GMMO[i * width + j].numSample = numPos;
			GMMO[i * width + j].pSampleNode = new struct_sampleNode[numPos];
		}
	}
	for (int f = 1; f < 100; f++) {
		String img_frame_path = "C:\\Data\\testsource\\3DVideos-distrib\\MSR3DVideo-Breakdancers\\cam2\\color-cam2-f0" + cv::format("%.2d", f) + ".jpg";
		//Mat mframe = imread(img_frame_path, 0);
		
		Mat mframe = imread(img_frame_path, IMREAD_UNCHANGED);
		vector<Mat> channels;
		Mat imageBlueChannel;
		Mat imageGreenChannel;
		Mat imageRedChannel;
		split(mframe, channels);
		imageBlueChannel = channels.at(0);
		imageGreenChannel = channels.at(1);
		imageRedChannel = channels.at(2);
		imshow("imageBlueChannel", imageBlueChannel);
		imshow("imageGreenChannel", imageGreenChannel);
		imshow("imageRedChannel", imageRedChannel);
		/*Mat RGB(channels[0].size(), CV_8UC3);
		merge(channels, RGB);
		imshow("merge", RGB);
		waitKey();*/
		Mat current(mframe.rows, mframe.cols, CV_8UC1);
		imageRedChannel.copyTo(current);
		//cvtColor(mframe, current, CV_BGR2GRAY);
		//imwrite("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\EM\\Ballet\\cam0\\current" + cv::format("%.2d", f) + ".jpg", current);
		//填入样本数据
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				GMMO[i * width + j].pSampleNode[f].x = current.at<uchar>(i, j);
			}
		}
		//GMMO[0].pSampleNode[f-1].x = current.at<uchar>(284,358);
	}
	//高斯分布初始化
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			GMMO[i * width + j].numClass = 2;
			GMMO[i * width + j].pGM = new struct_GM[GMMO[i * width + j].numClass];
			for (int k = 0; k<GMMO[0].numClass; k++)
			{
				if (k == 0) {
					GMMO[i * width + j].pGM[k].u = GMMO[i * width + j].pSampleNode[k].x;
					GMMO[i * width + j].pGM[k].piK = 0.998;
				}
				else {
					GMMO[i * width + j].pGM[k].u = 0;
					GMMO[i * width + j].pGM[k].piK = 0.001;
				}
				GMMO[i * width + j].pGM[k].sigma = 30;
				GMMO[i * width + j].pGM[k].Nk = GMMO[i * width + j].pGM[k].piK;
			}
		}
	}

	//EM迭代
	double temp;
	for (int f = 1; f < 100; f++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				GMMO[i * width + j].EM(1);
				//排序
				struct_GM tempGMMO;
				for (int k = 1; k<GMMO[i * width + j].numClass; k++) {
					for (int m = 0; m < k; m++)
					{
						if (GMMO[i * width + j].pGM[k].piK > GMMO[i * width + j].pGM[m].piK)
						{
							//swap max values 
							tempGMMO = GMMO[i * width + j].pGM[m];
							GMMO[i * width + j].pGM[m] = GMMO[i * width + j].pGM[k];
							GMMO[i * width + j].pGM[k] = tempGMMO;
						}
					}
				}
				temp = 0;
				bg_bw[i*width + j] = 0;
				for (int t = 0; t < GMMO[i * width + j].numClass; t++)//如果前几个单高斯模式的重要性之和大于T，则将这前几个单高斯模式认为为背景模型
				{
					temp += GMMO[i * width + j].pGM[t].piK;
					bg_bw[i*width + j] += GMMO[i * width + j].pGM[t].u *  GMMO[i * width + j].pGM[t].piK;
					if (temp >= T)
					{
						//M = k;
						break;
					}
				}

				back.at<uchar>(i, j) = (uchar)bg_bw[i*width + j];
			}
		}
		printf("第%d次迭代:\n", f);
		imwrite("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\EM\\Breakdancers\\cam2\\Rback" + cv::format("%.2d", f) + ".jpg", back);
	}
	
	return 0;
}



