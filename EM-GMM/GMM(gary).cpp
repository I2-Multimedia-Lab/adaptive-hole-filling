#include<opencv2/highgui.hpp>
#include<opencv2/core.hpp>
#include<opencv2/opencv.hpp>
#include "caculate.h"

using namespace std;
using namespace cv;
const int width = 1024;
const int height = 768;
const int framesize = width * height * 3 / 2;
int ggmain()
{
	Mat mframe = imread("C:\\Data\\testsource\\3DVideos-distrib\\MSR3DVideo-Ballet\\cam4\\color-cam4-f000.jpg", IMREAD_UNCHANGED);
	int height = mframe.rows;
	int width = mframe.cols;
	int C = 4;//number of gaussian components
	int M = 4;//number of background components
	int std_init = 30;//initial standard deviation初始标准差
	double D = 1;//若图像中相应位置的像素值与对应像素中像素的均值的距离小于标准差的D倍，则该点为背景
	double T = 0.7;//模型权重大于T则为背景模型
	double alpha = 0.05;//权值增量有关的参数
	double p = alpha;// / (1 / C);//更新率
	int min_index = 0;
	int*rank_ind = 0;
	int i, j, k, m;
	int rand_temp = 0;
	int rank_ind_temp = 0;
	Mat current(mframe.rows, mframe.cols, CV_8UC1);
	Mat back(mframe.rows, mframe.cols, CV_8UC1);
	Mat frg(mframe.rows, mframe.cols, CV_8UC1);

																		   //建立每个像素的在每个模型下的均值，标准差，权重，与均值差值，背景像素，重要性值数组
	double* mean = (double*)malloc(sizeof(double)*width*height*C);//pixelmeans（每个像素的均值）
	double* std = (double*)malloc(sizeof(double)*width*height*C);//pixel standard deviations
	double* w = (double*)malloc(sizeof(double)*width*height*C);//权值
	double* u_diff = (double*)malloc(sizeof(double)*width*height*C);//存放像素值与每一个单高斯模式的均值的差值
	int* bg_bw = (int*)malloc(sizeof(int)*width*height);
	double* rank = (double*)malloc(sizeof(double) * 1 * C);
	//printf("%d", numFrames);
	vector<Mat> channels;
	Mat imageBlueChannel;
	Mat imageGreenChannel;
	Mat imageRedChannel;
	split(mframe, channels);
	imageBlueChannel = channels.at(0);
	imageGreenChannel = channels.at(1);
	imageRedChannel = channels.at(2);
	imageBlueChannel.copyTo(current);
	//cvtColor(mframe, current, CV_BGR2GRAY);
	//初始化
	for (i = 0; i < height; i++)//对于每一个像素
	{
		for (j = 0; j < width; j++)
		{
			for (k = 0; k < C; k++)//对于每一个单高斯模型，初始化它的均值，标准差，权值
			{
				if (k == 0) {
					mean[i*width*C + j * C + k] = current.at<uchar>(i,j);
					w[i*width*C + j * C + k] = 1;
				}
				else {
					mean[i*width*C + j * C + k] = 0;
					w[i*width*C + j * C + k] = 0;
				}
				std[i*width*C + j * C + k] = std_init;
			}
		}
	}
	for (int f = 1; f<100; f++)
	{
		rank_ind = (int*)malloc(sizeof(int)*C);//模型重要性值排序对应的模型数组
		String img_frame_path;
		//img_frame_path = "C:\\Data\\testsource\\3DVideos-distrib\\MSR3DVideo-Breakdancers\\cam0\\color-cam0-f0" + cv::format("%.2d", f) + ".jpg";
		img_frame_path = "C:\\Data\\testsource\\3DVideos-distrib\\MSR3DVideo-Ballet\\cam2\\color-cam2-f0" + cv::format("%.2d", f) + ".jpg";
		Mat mframe = imread(img_frame_path, IMREAD_UNCHANGED);
		split(mframe, channels);
		imageBlueChannel = channels.at(0);
		imageGreenChannel = channels.at(1);
		imageRedChannel = channels.at(2);
		imageBlueChannel.copyTo(current);
		//cvtColor(mframe, current, CV_RGB2GRAY);//灰度化 current为mframe灰度化后的图像

												 //对于每一个像素，分别计算它和每一个单高斯模型的均值的差值
		for (i = 0; i < height; i++)//对于每一个像素
		{
			for (j = 0; j < width; j++)
			{
				for (k = 0; k < C; k++)
				{
					u_diff[i*width*C + j * C + k] = abs((uchar)current.at<uchar>(i, j) - mean[i*width*C + j * C + k]);

				}
			}
		}

		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				int match = 0;
				double temp = 0;
				double single_temp = 0;
				//遍历所有的单高斯模式，如果此像素满足任一单高斯模式，则匹配；如果此像素不满足任何的单高斯模式，则不匹配
				for (k = 0; k < C; k++)
				{
					if (abs(u_diff[i*width*C + j * C + k]) < D*std[i*width*C + j * C + k])//如果像素匹配某单个高斯模式，则对其权值、均值和标准差进行更新
					{
						match = 1;
						p = pro(current.at<uchar>(i, j), mean[i*width*C + j * C + k], std[i*width*C + j * C + k]);
						w[i*width*C + j * C + k] += p * (1 - w[i*width*C + j * C + k]);//更新权值
						mean[i*width*C + j * C + k] = (1 - p)*mean[i*width*C + j * C + k] + p * (uchar)current.at<uchar>(i, j);//更新均值
						std[i*width*C + j * C + k] = sqrt((1 - p)*(std[i*width*C + j * C + k] * std[i*width*C + j * C + k]) + p * (pow((uchar)current.at<uchar>(i, j) - mean[i*width*C + j * C + k], 2)));//更新标准差

					}
					else
					{
						w[i*width*C + j * C + k] = (1 - alpha)*w[i*width*C + j * C + k];//如果像素不符合某单个高斯模型，则将此单高斯模型的权值降低
					}
				}

				if (match == 1)//如果和任一单高斯模式匹配，则将权值归一化
				{
					for (k = 0; k < C; k++)
					{
						temp += w[i*width*C + j * C + k];//计算四个单高斯模式权值的和
					}
					for (k = 0; k < C; k++)
					{
						w[i*width*C + j * C + k] = w[i*width*C + j * C + k] / temp;//权值归一化，使得所有权值和为1
					}
				}
				else//如果和所有单高斯模式都不匹配，则寻找权值最小的高斯模式并删除，然后增加一个新的高斯模式
				{
					single_temp = w[i*width*C + j * C];
					for (k = 0; k < C; k++)
					{
						if (w[i*width*C + j * C + k] < single_temp)
						{
							min_index = k;//寻找权值最小的高斯模式
							single_temp = w[i*width*C + j * C + k];
						}

					}
					mean[i*width*C + j * C + min_index] = current.at<uchar>(i, j);//建立一个新的高斯模式，均值为当前像素值
					std[i*width*C + j * C + min_index] = std_init;//标准差为初始值
					w[i*width*C + j * C + min_index] = 0.01;
					for (k = 0; k < C; k++)
					{
						temp += w[i*width*C + j * C + k];//计算四个单高斯模式权值的和
					}
					for (k = 0; k < C; k++)
					{
						w[i*width*C + j * C + k] = w[i*width*C + j * C + k] / temp;//权值归一化，使得所有权值和为1
					}

				}

				for (k = 0; k < C; k++)//计算每个单高斯模式的重要性
				{
					rank[k] = w[i*width*C + j * C + k] / std[i*width*C + j * C + k];
					rank_ind[k] = k;
				}

				for (k = 1; k < C; k++)//对重要性排序
				{
					for (m = 0; m < k; m++)
					{
						if (rank[k] > rank[m])
						{
							//swap max values  
							rand_temp = rank[m];
							rank[m] = rank[k];
							rank[k] = rand_temp;
							//swap max index values  
							rank_ind_temp = rank_ind[m];
							rank_ind[m] = rank_ind[k];
							rank_ind[k] = rank_ind_temp;
						}
					}
				}

				bg_bw[i*width + j] = 0;
				for (k = 0; k < C; k++)//如果前几个单高斯模式的重要性之和大于T，则将这前几个单高斯模式认为为背景模型
				{
					temp += w[i*width*C + j * C + rank_ind[k]];
					bg_bw[i*width + j] += mean[i*width*C + j * C + rank_ind[k]] * w[i*width*C + j * C + rank_ind[k]];
					if (temp >= T)
					{
						M = k;
						break;
					}
				}

				back.at<uchar>(i, j) = (uchar)bg_bw[i*width + j];//背景图像 test存储背景图像信息

				match = 0; k = 0;
				while ((match == 0) && (k <= M))//如果某像素不符合背景模型中任一单高斯模型，则此像素为前景像素
				{
					if (abs(u_diff[i*width*C + j * C + rank_ind[k]]) <= D * std[i*width*C + j * C + rank_ind[k]])
					{
						frg.at<uchar>(i, j) = 0;
						match = 1;
					}
					else
						frg.at<uchar>(i, j) = current.at<uchar>(i, j);

					k += 1;

				}


			}
		}
		/*mframe = cvQueryFrame(capture);
		if (mframe == NULL)
		return -1;*/

		imshow("frg", frg);
		imshow("back", back);
		//存储前景背景
		//if (f == 99) {
		//	const char* path1;
		//	const char* path2;
		//	path1 = "C:\\Users\\a\\Documents\\Visual Studio 2015\\Projects\\newProject\\balletsatuback.jpg";
		//	//path2 = "C:\\Users\\a\\Documents\\Visual Studio 2015\\Projects\\newProject\\frg.bmp";
		//	cvSaveImage(path1, test);
		//	//cvSaveImage(path2, frg);
		//}
		imwrite("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\Ballet\\cam2\\GMM\\Blue\\bback" + cv::format("%.2d", f) + ".jpg", back);

	}
	//char key = waitKey(100);

	return 0;
}