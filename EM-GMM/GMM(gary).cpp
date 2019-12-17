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
	int std_init = 30;//initial standard deviation��ʼ��׼��
	double D = 1;//��ͼ������Ӧλ�õ�����ֵ���Ӧ���������صľ�ֵ�ľ���С�ڱ�׼���D������õ�Ϊ����
	double T = 0.7;//ģ��Ȩ�ش���T��Ϊ����ģ��
	double alpha = 0.05;//Ȩֵ�����йصĲ���
	double p = alpha;// / (1 / C);//������
	int min_index = 0;
	int*rank_ind = 0;
	int i, j, k, m;
	int rand_temp = 0;
	int rank_ind_temp = 0;
	Mat current(mframe.rows, mframe.cols, CV_8UC1);
	Mat back(mframe.rows, mframe.cols, CV_8UC1);
	Mat frg(mframe.rows, mframe.cols, CV_8UC1);

																		   //����ÿ�����ص���ÿ��ģ���µľ�ֵ����׼�Ȩ�أ����ֵ��ֵ���������أ���Ҫ��ֵ����
	double* mean = (double*)malloc(sizeof(double)*width*height*C);//pixelmeans��ÿ�����صľ�ֵ��
	double* std = (double*)malloc(sizeof(double)*width*height*C);//pixel standard deviations
	double* w = (double*)malloc(sizeof(double)*width*height*C);//Ȩֵ
	double* u_diff = (double*)malloc(sizeof(double)*width*height*C);//�������ֵ��ÿһ������˹ģʽ�ľ�ֵ�Ĳ�ֵ
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
	//��ʼ��
	for (i = 0; i < height; i++)//����ÿһ������
	{
		for (j = 0; j < width; j++)
		{
			for (k = 0; k < C; k++)//����ÿһ������˹ģ�ͣ���ʼ�����ľ�ֵ����׼�Ȩֵ
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
		rank_ind = (int*)malloc(sizeof(int)*C);//ģ����Ҫ��ֵ�����Ӧ��ģ������
		String img_frame_path;
		//img_frame_path = "C:\\Data\\testsource\\3DVideos-distrib\\MSR3DVideo-Breakdancers\\cam0\\color-cam0-f0" + cv::format("%.2d", f) + ".jpg";
		img_frame_path = "C:\\Data\\testsource\\3DVideos-distrib\\MSR3DVideo-Ballet\\cam2\\color-cam2-f0" + cv::format("%.2d", f) + ".jpg";
		Mat mframe = imread(img_frame_path, IMREAD_UNCHANGED);
		split(mframe, channels);
		imageBlueChannel = channels.at(0);
		imageGreenChannel = channels.at(1);
		imageRedChannel = channels.at(2);
		imageBlueChannel.copyTo(current);
		//cvtColor(mframe, current, CV_RGB2GRAY);//�ҶȻ� currentΪmframe�ҶȻ����ͼ��

												 //����ÿһ�����أ��ֱ��������ÿһ������˹ģ�͵ľ�ֵ�Ĳ�ֵ
		for (i = 0; i < height; i++)//����ÿһ������
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
				//�������еĵ���˹ģʽ�����������������һ����˹ģʽ����ƥ�䣻��������ز������κεĵ���˹ģʽ����ƥ��
				for (k = 0; k < C; k++)
				{
					if (abs(u_diff[i*width*C + j * C + k]) < D*std[i*width*C + j * C + k])//�������ƥ��ĳ������˹ģʽ�������Ȩֵ����ֵ�ͱ�׼����и���
					{
						match = 1;
						p = pro(current.at<uchar>(i, j), mean[i*width*C + j * C + k], std[i*width*C + j * C + k]);
						w[i*width*C + j * C + k] += p * (1 - w[i*width*C + j * C + k]);//����Ȩֵ
						mean[i*width*C + j * C + k] = (1 - p)*mean[i*width*C + j * C + k] + p * (uchar)current.at<uchar>(i, j);//���¾�ֵ
						std[i*width*C + j * C + k] = sqrt((1 - p)*(std[i*width*C + j * C + k] * std[i*width*C + j * C + k]) + p * (pow((uchar)current.at<uchar>(i, j) - mean[i*width*C + j * C + k], 2)));//���±�׼��

					}
					else
					{
						w[i*width*C + j * C + k] = (1 - alpha)*w[i*width*C + j * C + k];//������ز�����ĳ������˹ģ�ͣ��򽫴˵���˹ģ�͵�Ȩֵ����
					}
				}

				if (match == 1)//�������һ����˹ģʽƥ�䣬��Ȩֵ��һ��
				{
					for (k = 0; k < C; k++)
					{
						temp += w[i*width*C + j * C + k];//�����ĸ�����˹ģʽȨֵ�ĺ�
					}
					for (k = 0; k < C; k++)
					{
						w[i*width*C + j * C + k] = w[i*width*C + j * C + k] / temp;//Ȩֵ��һ����ʹ������Ȩֵ��Ϊ1
					}
				}
				else//��������е���˹ģʽ����ƥ�䣬��Ѱ��Ȩֵ��С�ĸ�˹ģʽ��ɾ����Ȼ������һ���µĸ�˹ģʽ
				{
					single_temp = w[i*width*C + j * C];
					for (k = 0; k < C; k++)
					{
						if (w[i*width*C + j * C + k] < single_temp)
						{
							min_index = k;//Ѱ��Ȩֵ��С�ĸ�˹ģʽ
							single_temp = w[i*width*C + j * C + k];
						}

					}
					mean[i*width*C + j * C + min_index] = current.at<uchar>(i, j);//����һ���µĸ�˹ģʽ����ֵΪ��ǰ����ֵ
					std[i*width*C + j * C + min_index] = std_init;//��׼��Ϊ��ʼֵ
					w[i*width*C + j * C + min_index] = 0.01;
					for (k = 0; k < C; k++)
					{
						temp += w[i*width*C + j * C + k];//�����ĸ�����˹ģʽȨֵ�ĺ�
					}
					for (k = 0; k < C; k++)
					{
						w[i*width*C + j * C + k] = w[i*width*C + j * C + k] / temp;//Ȩֵ��һ����ʹ������Ȩֵ��Ϊ1
					}

				}

				for (k = 0; k < C; k++)//����ÿ������˹ģʽ����Ҫ��
				{
					rank[k] = w[i*width*C + j * C + k] / std[i*width*C + j * C + k];
					rank_ind[k] = k;
				}

				for (k = 1; k < C; k++)//����Ҫ������
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
				for (k = 0; k < C; k++)//���ǰ��������˹ģʽ����Ҫ��֮�ʹ���T������ǰ��������˹ģʽ��ΪΪ����ģ��
				{
					temp += w[i*width*C + j * C + rank_ind[k]];
					bg_bw[i*width + j] += mean[i*width*C + j * C + rank_ind[k]] * w[i*width*C + j * C + rank_ind[k]];
					if (temp >= T)
					{
						M = k;
						break;
					}
				}

				back.at<uchar>(i, j) = (uchar)bg_bw[i*width + j];//����ͼ�� test�洢����ͼ����Ϣ

				match = 0; k = 0;
				while ((match == 0) && (k <= M))//���ĳ���ز����ϱ���ģ������һ����˹ģ�ͣ��������Ϊǰ������
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
		//�洢ǰ������
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