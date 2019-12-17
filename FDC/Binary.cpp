#include "Binary.h"
#include "opencv\highgui.h"

void Binary(Mat *input, Mat *output)
{
	int channels = input->channels();
	Mat samples(input->cols*input->rows, 1, CV_32FC3);//CV_[λ��][���������][����ǰ׺]C[ͨ����]
													  //��Ǿ���32λ���� 
	Mat labels(input->cols*input->rows, 1, CV_32SC1);
	uchar* p;
	int i, j, k = 0;
	for (i = 0; i < input->rows; i++)
	{
		p = input->ptr<uchar>(i);//��ָ��
		for (j = 0; j< input->cols; j++)
		{
			samples.at<Vec3f>(k, 0)[0] = float(p[j]);
			samples.at<Vec3f>(k, 0)[1] = float(p[j * 3 + 1]);
			samples.at<Vec3f>(k, 0)[2] = float(p[j * 3 + 2]);
			k++;
		}
	}


	int clusterCount = 2;
	Mat centers(clusterCount, 1, samples.type());//���ڴ洢���������ĵ�
	kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);//labels��ʾÿһ����������ı�ǩ����һ����������0��ʼ����������,�Ǵ���.
																																	//������֪��3�����࣬�ò�ͬ�ĻҶȲ��ʾ�� 
																																	//Mat output(input->rows, input->cols, CV_8UC1);
	float step = 255 / (clusterCount - 1);
	k = 0;
	for (i = 0; i < output->rows; i++)
	{
		p = output->ptr<uchar>(i);//��ָ��
		for (j = 0; j< output->cols; j++)
		{
			int tt = labels.at<int>(k, 0);
			k++;
			p[j] = 255 - tt*step;
		}
	}
}


