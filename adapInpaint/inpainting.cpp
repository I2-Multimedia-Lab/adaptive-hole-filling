#include ".\inpainting.h"
#include <memory.h>
#include <math.h>
#include<stdio.h>
#include <opencv2/opencv.hpp> 
#include "opencv\highgui.h"
using namespace cv;
using namespace std;

#include"Canny.h"
#define MAX(a, b)  (((a) > (b)) ? (a) : (b)) 
#define MIN(a, b)  (((a) < (b)) ? (a) : (b)) 
#define pi 3.14159
#define win 2
#define CannyAccThresh  128
//inpainting::inpainting(char * imgname, char * FDCname, char * GMMname, char * depFDC, char * depGMM);
//{
//	
//}
struct point {
	int x;
	int y;
	double pri;
};
inpainting::inpainting(char * imgname, char * depname, char * FDCname, char * GMMname,int f)
{
	Frame = f;
	img = imread(imgname);
	Mat dep = imread(depname, 1);
	Mat FDCimg = imread(FDCname);
	Mat GMMimg = imread(GMMname);
	
	Mat depte(dep.rows, dep.cols, CV_8UC3);
	depte.copyTo(deptest);
	img.copyTo(m_pImage);
	FDCimg.copyTo(m_FDC);
	GMMimg.copyTo(m_GMM);
	
	if (!img.empty()) {
		m_width = m_pImage.cols;
		m_height = m_pImage.rows;

		m_mark = new int[m_width*m_height];
		m_confid = new double[m_width*m_height];
		memset(m_confid, 0, m_width*m_height * sizeof(double));
		m_pri = new double[m_width*m_height];
		m_gray = new double[m_width*m_height];
		m_source = new bool[m_width*m_height];

		m_r = new double[m_width*m_height];
		m_g = new double[m_width*m_height];
		m_b = new double[m_width*m_height];

		mfdc_r = new double[m_width*m_height];
		mfdc_g = new double[m_width*m_height];
		mfdc_b = new double[m_width*m_height];
		mdfdc = new double[m_width*m_height];

		mgmm_r = new double[m_width*m_height];
		mgmm_g = new double[m_width*m_height];
		mgmm_b = new double[m_width*m_height];
		mdgmm = new double[m_width*m_height];
		m_dep = new int[m_width*m_height];
		depmask = new int[m_width*m_height];
	}

	else printf("one or more file is not opened!\n");
	//ballet
	//m_top = 170;  // initialize the rectangle area
	//m_bottom = 628;
	//m_left = 239;
	//m_right = 492;
	//breakdancers
	//m_top = 100;  // initialize the rectangle area
	//m_bottom = 700;
	//m_left = 300;
	//m_right = 1000;
	//BookArrival
	m_top = 300;  // initialize the rectangle area
	m_bottom = 730;
	m_left = 360;
	m_right = 930;
	
	Canny(&dep, &m_DepBound);
	morphologyEx(m_DepBound, m_DepBound, MORPH_DILATE, Mat(3, 3, CV_8U), Point(-1, -1), 1);
	//imshow("m_DepBound", m_DepBound);
	Mat deprgb;
	deprgb = imread(depname, 1);
	dep_inpaint(deprgb, &m_DepInpaint);
	//imshow("deptest", deptest);
	/*imshow("dep", dep);
	imshow("depbound", m_DepBound);
	imshow("m_DepInpaint", m_DepInpaint);
	waitKey();*/
	for (int y = 0; y<m_height; y++)
		for (int x = 0; x<m_width; x++)
		{
			
			m_r[y*m_width + x] = m_pImage.at<Vec3b>(y, x)[2];
			m_g[y*m_width + x] = m_pImage.at<Vec3b>(y, x)[1];
			m_b[y*m_width + x] = m_pImage.at<Vec3b>(y, x)[0];

			mfdc_r[y*m_width + x] = m_FDC.at<Vec3b>(y, x)[2];
			mfdc_g[y*m_width + x] = m_FDC.at<Vec3b>(y, x)[1];
			mfdc_b[y*m_width + x] = m_FDC.at<Vec3b>(y, x)[0];

			mgmm_r[y*m_width + x] = m_GMM.at<Vec3b>(y, x)[2];
			mgmm_g[y*m_width + x] = m_GMM.at<Vec3b>(y, x)[1];
			mgmm_b[y*m_width + x] = m_GMM.at<Vec3b>(y, x)[0];
			m_dep[y*m_width + x] = m_DepInpaint.at<Vec3b>(y, x)[0];
		}
	
	for (int i = m_left; i <= m_right; i++)
	{
		for (int j = m_top; j <= m_bottom; j++)
		{
				for (int y = MAX(j - win, 0); y <= MIN(j + win, m_height - 1); y++)
					for (int x = MAX(i - win, 0); x <= MIN(i + win, m_width - 1); x++) {

						if (depmask[y*m_width + x] == LEFTBOUN|| depmask[y*m_width + x] == RIGHTBOUN) {
							//m_dep[j*m_width + i] = 255;
							break;
						}					
			}
		}
	}
}



inpainting::~inpainting(void)
{
	if (m_mark)delete m_mark;
	if (m_source)delete m_source;
	if (m_r)delete m_r;
	if (m_g)delete m_g;
	if (m_b)delete m_b;
}

bool inpainting::process(void)
{
	char path1[200];
	char path[200];
	char temp[30];
	windowsize = 3;
	Convert2Gray();  // convert it to gray image
	DrawBoundary();  // first time draw boundary
	memset(m_pri, 0, m_width*m_height * sizeof(double));
	
	int count = 0;
	for (int j = m_top; j <= m_bottom; j++)
		for (int i = m_left; i <= m_right; i++)
			if (m_mark[j*m_width + i] == BOUNDARY)
				m_pri[j*m_width + i] = priority(i, j);//if it is boundary, calculate the priority
	
	while (TargetExist())
	{
		count++;
		double max_pri = 0;
		int pri_x = 0;
		int pri_y = 0;
		int flag = 0;
		windowsize = 4;
		for (int i = m_left; i <= m_right; i++)
		{
			for (int j = m_top; j <= m_bottom; j++)
			{

				if (m_mark[j*m_width + i] == BOUNDARY&&m_pri[j*m_width + i] > max_pri)// find the boundary pixel with highest priority
				{
					pri_x = i;
					pri_y = j;
					max_pri = m_pri[j*m_width + i];
				}
			}
		}
		printf("pri_x is %d, pri_y is %d, amount is %lf\n", pri_x, pri_y, max_pri);
		int patch;
		patch = CannyPatch(pri_x, pri_y);
		update(pri_x, pri_y, patch, ComputeConfidence(pri_x, pri_y));// inpaint this area and update confidence
		UpdateBoundary(pri_x, pri_y); // update boundary near the changed area
		UpdatePri(pri_x, pri_y);  //  update priority near the changed area

		strcpy(path, save_path);
	}
	strcat(path, "_book");
	strcat(path, format("%.2d", Frame).c_str());
	strcat(path, ".bmp");
	imwrite(path, m_pImage);

	return true;
}

void inpainting::DrawBoundary(void)
{
	for (int y = 0; y<m_height; y++)
		for (int x = 0; x<m_width; x++)
		{
			// if the pixel is specified as boundary
			if(m_pImage.at<Vec3b>(y, x)[1]- m_pImage.at<Vec3b>(y, x)[0] >=color_threshold||m_pImage.at<Vec3b>(y, x)[1]- m_pImage.at<Vec3b>(y, x)[2] >=color_threshold)
			{
				m_mark[y*m_width + x] = TARGET;
				m_confid[y*m_width + x] = 0;
			}
			else {
				m_mark[y*m_width + x] = SOURCE;
				m_confid[y*m_width + x] = 1;
			}
		}

	for (int j = 0; j< m_height; j++)
		for (int i = 0; i< m_width; i++)
		{
			if (m_mark[j*m_width + i] == TARGET)
			{
				if (j == m_height - 1 || j == 0 || i == 0 || i == m_width - 1 || m_mark[(j - 1)*m_width + i] == SOURCE || m_mark[j*m_width + i - 1] == SOURCE
					|| m_mark[j*m_width + i + 1] == SOURCE || m_mark[(j + 1)*m_width + i] == SOURCE)m_mark[j*m_width + i] = BOUNDARY;
			}
		}
}

void inpainting::dep_inpaint(Mat depori, Mat * depinpaint)
{
	int *m_depmark= new int[m_width*m_height];
	Mat deptrans;
	depori.copyTo(deptrans);
	//划分target和source区域 m_depmask depmask
	for (int y = 0; y< m_height; y++)
		for (int x = 0; x< m_width; x++)
		{
			if (depori.at<Vec3b>(y, x)[0]<= 15)
			{
				m_depmark[y*m_width + x] = TARGET;
				depmask[y*m_width + x] = TARGET;
				deptest.at<Vec3b>(y, x)[0] = 255;
				deptest.at<Vec3b>(y, x)[1] = 255;
				deptest.at<Vec3b>(y, x)[2] = 255;
			}
			else {
				m_depmark[y*m_width + x] = SOURCE;
				depmask[y*m_width + x] = SOURCE;
				deptest.at<Vec3b>(y, x)[0] = 0;
				deptest.at<Vec3b>(y, x)[1] = 0;
				deptest.at<Vec3b>(y, x)[2] = 0;
			}
		}
	for (int j = m_top; j< m_bottom; j++)
		for (int i = m_left; i< m_right; i++)
		{
			if (m_depmark[j*m_width + i] == TARGET)
			{
		
				if (j == m_height - 1 || j == 0 || i == 0 || i == m_width - 1 || m_depmark[(j - 1)*m_width + i] == SOURCE || m_depmark[j*m_width + i - 1] == SOURCE
					|| m_depmark[j*m_width + i + 1] == SOURCE || m_depmark[(j + 1)*m_width + i] == SOURCE)m_depmark[j*m_width + i] = BOUNDARY;
			}
			if (depmask[j*m_width + i] == TARGET)
			{
				if (j == m_height - 1 || j == 0 || i == 0 || i == m_width - 1 || depmask[(j - 1)*m_width + i] == SOURCE || depmask[j*m_width + i - 1] == SOURCE
					|| depmask[j*m_width + i + 1] == SOURCE || depmask[(j + 1)*m_width + i] == SOURCE) {
					if ((depmask[(j - 1)*m_width + i] == SOURCE&&depmask[j*m_width + i+1] == TARGET) || (depmask[j*m_width + i - 1] == SOURCE&&depmask[j*m_width + i + 1] == TARGET))
					{
						depmask[j*m_width + i] = LEFTBOUN;
						deptest.at<Vec3b>(j, i)[0] = 0;
						deptest.at<Vec3b>(j, i)[1] = 255;
						deptest.at<Vec3b>(j, i)[2] = 0;
					}
					else {
						depmask[j*m_width + i] = RIGHTBOUN;
						deptest.at<Vec3b>(j, i)[0] = 255;
						deptest.at<Vec3b>(j, i)[1] = 0;
						deptest.at<Vec3b>(j, i)[2] = 0;
					}
				}
					
			}
		}

	int dep_x, dep_y;
	for (int j = m_top; j <= m_bottom; j++) {
		for (int i = m_left; i <= m_right; i++)
		{
			if (m_depmark[j*m_width + i] == BOUNDARY)// find the boundary pixel with highest priority
			{
				dep_x = i;
				dep_y = j;
				int px, py,dcount=0;
				double pixel_num=0;
				for (int iter_y = (-1)*depwindowsize; iter_y <= depwindowsize; iter_y++)
					for (int iter_x = (-1)*depwindowsize; iter_x <= depwindowsize; iter_x++)
					{
						px = dep_x + iter_x;
						py = dep_y + iter_y;
						if (m_depmark[py*m_width + px] == SOURCE)
						{
							pixel_num += deptrans.at<Vec3b>(py, px)[0];
							dcount++;

						}
					}
				pixel_num = pixel_num / dcount;
				deptrans.at<Vec3b>(dep_y, dep_x)[0] = pixel_num;
				deptrans.at<Vec3b>(dep_y, dep_x)[1] = pixel_num;
				deptrans.at<Vec3b>(dep_y, dep_x)[2] = pixel_num;
				m_depmark[dep_y*m_width + dep_x] = SOURCE;
				for (int y = MAX(j - depwindowsize - 2, 0); y <= MIN(j + depwindowsize + 2, m_height - 1); y++)
					for (int x = MAX(i - depwindowsize - 2, 0); x <= MIN(i + depwindowsize + 2, m_width - 1); x++)
					{
						if (m_depmark[y*m_width + x]<0)
							m_depmark[y*m_width + x] = TARGET;
						else m_depmark[y*m_width + x] = SOURCE;
					}

				for (int y = MAX(j - depwindowsize - 2, 0); y <= MIN(j + depwindowsize + 2, m_height - 1); y++)
					for (int x = MAX(i - depwindowsize - 2, 0); x <= MIN(i + depwindowsize + 2, m_width - 1); x++)
					{
						if (m_depmark[y*m_width + x] == TARGET)
						{
							if (y == m_height - 1 || y == 0 || x == 0 || x == m_width - 1 || m_depmark[(y - 1)*m_width + x] == SOURCE || m_depmark[y*m_width + x - 1] == SOURCE
								|| m_depmark[y*m_width + x + 1] == SOURCE || m_depmark[(y + 1)*m_width + x] == SOURCE)m_depmark[y*m_width + x] = BOUNDARY;
						}
					}
			}
		}
	}
	deptrans.copyTo(*depinpaint);
}



double inpainting::ComputeConfidence(int i, int j)
{
	double confidence = 0;
	for (int y = MAX(j - windowsize, 0); y <= MIN(j + windowsize, m_height - 1); y++)
		for (int x = MAX(i - windowsize, 0); x <= MIN(i + windowsize, m_width - 1); x++)
			confidence += m_confid[y*m_width + x];
	confidence /= (windowsize * 2 + 1)*(windowsize * 2 + 1);
	return confidence;
}

double inpainting::priority(int x, int y)
{
	double confidence, data,edepth;
	confidence = ComputeConfidence(x, y); // confidence term
	edepth = ComputeDepth(x, y);
	return confidence*100/ edepth;
}

double inpainting::ComputeData(int i, int j)
{
	gradient grad, temp, grad_T;
	grad.grad_x = 0;
	grad.grad_y = 0;
	double result;
	double magnitude;
	double max = 0;
	int x, y;
	for (y = MAX(j - windowsize, 0); y <= MIN(j + windowsize, m_height - 1); y++)
		for (x = MAX(i - windowsize, 0); x <= MIN(i + windowsize, m_width - 1); x++)
		{

			if (y < 1) y = 1;
			// find the greatest gradient in this patch, this will be the gradient of this pixel(according to "detail paper")
			if (m_mark[y*m_width + x] >= 0) // source pixel
			{
				//since I use four neighbors to calculate the gradient, make sure this four neighbors do not touch target region(big jump in gradient)
				if (m_mark[y*m_width + x + 1]<0 || m_mark[y*m_width + x - 1]<0 || m_mark[(y + 1)*m_width + x]<0 || m_mark[(y - 1)*m_width + x]<0)continue;
				temp = GetGradient(x, y);
				magnitude = temp.grad_x*temp.grad_x + temp.grad_y*temp.grad_y;
				if (magnitude>max)
				{
					grad.grad_x = temp.grad_x;
					grad.grad_y = temp.grad_y;
					max = magnitude;
				}
			}
		}
	grad_T.grad_x = grad.grad_y;// perpendicular to the gradient: (x,y)->(y, -x)
	grad_T.grad_y = -grad.grad_x;

	mynorm nn = GetNorm(i, j);
	result = nn.norm_x*grad_T.grad_x + nn.norm_y*grad_T.grad_y; // dot product
	result /= 255; //"alpha" in the paper: normalization factor
	result = fabs(result);
	return result;
}
double inpainting::ComputeDepth(int i, int j)
{
	double edepth = 0;
	int count = 0;
	for (int y = MAX(j - windowsize, 0); y <= MIN(j + windowsize, m_height - 1); y++)
		for (int x = MAX(i - windowsize, 0); x <= MIN(i + windowsize, m_width - 1); x++)
			if (m_confid[y*m_width + x] == 1) {
				edepth += m_dep[y*m_width + x];
				count++;
			}
	if (count == 0) edepth = 0;
	else edepth /= count;
	return edepth;
}
mynorm inpainting::GetNorm(int i, int j)
{
	mynorm result;
	int num = 0;
	int neighbor_x[9];
	int neighbor_y[9];
	int record[9];
	int count = 0;
	for (int y = MAX(j - 1, 0); y <= MIN(j + 1, m_height - 1); y++)
		for (int x = MAX(i - 1, 0); x <= MIN(i + 1, m_width - 1); x++)
		{
			count++;
			if (x == i&&y == j)continue;
			if (m_mark[y*m_width + x] == BOUNDARY)
			{
				num++;
				neighbor_x[num] = x;
				neighbor_y[num] = y;
				record[num] = count;
			}
		}
	if (num == 0 || num == 1) // if it doesn't have two neighbors, give it a random number to proceed
	{
		result.norm_x = 0.6;
		result.norm_y = 0.8;
		return result;
	}
	// draw a line between the two neighbors of the boundary pixel, then the norm is the perpendicular to the line
	int n_x = neighbor_x[2] - neighbor_x[1];
	int n_y = neighbor_y[2] - neighbor_y[1];
	int temp = n_x;
	n_x = n_y;
	n_y = temp;
	double square = pow(double(n_x*n_x + n_y*n_y), 0.5);

	result.norm_x = n_x / square;
	result.norm_y = n_y / square;
	return result;
}

void inpainting::Convert2Gray(void)
{
	double r, g, b;
	for (int y = 0; y<m_height; y++)
		for (int x = 0; x<m_width; x++)
		{
			//m_pImage->GetPixel(x, y, cc);
			r = m_pImage.at<Vec3b>(y, x)[0];
			g = m_pImage.at<Vec3b>(y, x)[1];
			b = m_pImage.at<Vec3b>(y, x)[2];
			m_gray[y*m_width + x] = (double)((r * 3735 + g * 19267 + b * 9765) / 32767);
		}
}

gradient inpainting::GetGradient(int i, int j)
{
	gradient result;
	if (j >= 513)j = 512;
	result.grad_x = (m_gray[j*m_width + i + 1] - m_gray[j*m_width + i - 1]) / 2.0;
	result.grad_y = (m_gray[(j + 1)*m_width + i] - m_gray[(j - 1)*m_width + i]) / 2.0;

	if (i == 0)result.grad_x = m_gray[j*m_width + i + 1] - m_gray[j*m_width + i];
	if (i == m_width - 1)result.grad_x = m_gray[j*m_width + i] - m_gray[j*m_width + i - 1];
	if (j == 0)result.grad_y = m_gray[(j + 1)*m_width + i] - m_gray[j*m_width + i];
	if (j == m_height - 1)result.grad_y = m_gray[j*m_width + i] - m_gray[(j - 1)*m_width + i];
	return result;
}

bool inpainting::draw_source(void)
{
	// draw a window around the pixel, if all of the points within the window are source pixels, then this patch can be used as a source patch
	bool flag;
	for (int j = 0; j<m_height; j++)
		for (int i = 0; i<m_width; i++)
		{
			flag = 1;
			if (i<windowsize || j<windowsize || i >= m_width - windowsize || j >= m_height - windowsize)m_source[j*m_width + i] = false;//cannot form a complete window
			else
			{
				for (int y = j - windowsize; y <= j + windowsize; y++)
				{
					for (int x = i - windowsize; x <= i + windowsize; x++)
					{
						if (m_mark[y*m_width + x] != SOURCE)
						{
							m_source[j*m_width + i] = false;
							flag = false;
							break;
						}
					}
					if (flag == false)break;
				}
				if (flag != false)m_source[j*m_width + i] = true;
			}
		}
	return true;
}

int inpainting::PatchTexture(int x, int y)
{
	double ftemp_r;
	double ftemp_g;
	double ftemp_b;

	double gtemp_r;
	double gtemp_g;
	double gtemp_b;
	// find the most similar patch, according to SSD
	//COLORREF color_target, color_source, color_diff;
	double r0, g0, b0;
	double r1, g1, b1;

	long min = 99999999;
	double FDCsum,GMMsum;
	//int source_x, source_y;
	int target_x, target_y;
	int flag = 0;
	FDCsum = 0; GMMsum = 0;
	if (x==212 && y==211) 
flag = 1;
	//bool thresholdflag = true;
	//while (thresholdflag) {
		for (int iter_y = (-1)*windowsize; iter_y <= windowsize; iter_y++)
			for (int iter_x = (-1)*windowsize; iter_x <= windowsize; iter_x++)
			{
				/*source_x = i + iter_x;
				source_y = j + iter_y;*/

				target_x = x + iter_x;
				target_y = y + iter_y;

				if (target_x<0 || target_x >= m_width || target_y<0 || target_y >= m_height)continue;

				if (m_mark[target_y*m_width + target_x] >= 0)
				{
					ftemp_r = img.at<Vec3b>(target_y, target_x)[2] - m_FDC.at<Vec3b>(target_y, target_x)[2];
					ftemp_g = img.at<Vec3b>(target_y, target_x)[1] - m_FDC.at<Vec3b>(target_y, target_x)[1];
					ftemp_b = img.at<Vec3b>(target_y, target_x)[0] - m_FDC.at<Vec3b>(target_y, target_x)[0];
					FDCsum += sqrt(ftemp_r*ftemp_r + ftemp_g*ftemp_g + ftemp_b*ftemp_b) ;
					
					gtemp_r = img.at<Vec3b>(target_y, target_x)[2] - m_GMM.at<Vec3b>(target_y, target_x)[2];
					gtemp_g = img.at<Vec3b>(target_y, target_x)[1] - m_GMM.at<Vec3b>(target_y, target_x)[1];
					gtemp_b = img.at<Vec3b>(target_y, target_x)[0] - m_GMM.at<Vec3b>(target_y, target_x)[0];
					GMMsum += sqrt(gtemp_r*gtemp_r + gtemp_g*gtemp_g + gtemp_b*gtemp_b) ;
					//printf("(%d,%d) img:%d %d %d,fdc: %d %d %d,gmm:%d %d %d\n", target_y, target_x, img.at<Vec3b>(target_y, target_x)[2], img.at<Vec3b>(target_y, target_x)[1], img.at<Vec3b>(target_y, target_x)[0], m_FDC.at<Vec3b>(target_y, target_x)[2], m_FDC.at<Vec3b>(target_y, target_x)[1], m_FDC.at<Vec3b>(target_y, target_x)[0
					//], m_GMM.at<Vec3b>(target_y, target_x)[2], m_GMM.at<Vec3b>(target_y, target_x)[1], m_GMM.at<Vec3b>(target_y, target_x)[2]);
					//printf("img-fdc:%d %d %d,img-gmm:%d %d %d\n", img.at<Vec3b>(target_y, target_x)[2] - m_FDC.at<Vec3b>(target_y, target_x)[2], img.at<Vec3b>(target_y, target_x)[1] - m_FDC.at<Vec3b>(target_y, target_x)[1], img.at<Vec3b>(target_y, target_x)[0] - m_FDC.at<Vec3b>(target_y, target_x)[0], img.at<Vec3b>(target_y, target_x)[2] - m_GMM.at<Vec3b>(target_y, target_x)[2], img.at<Vec3b>(target_y, target_x)[1] - m_GMM.at<Vec3b>(target_y, target_x)[1], img.at<Vec3b>(target_y, target_x)[0] - m_GMM.at<Vec3b>(target_y, target_x)[0]);
				}
			}

	
	if (FDCsum<=GMMsum)
	{
		return FDC;
	}
	else return GMM;

}

int inpainting::CannyPatch(int x, int y)
{
	bool simflag = false;
	//bool cannyflag = false;
	double thod = 0;
	int dx, dy;
	for (int iter_y = (-1)*2; iter_y <= 2; iter_y++)
		for (int iter_x = (-1)*2; iter_x <= 2; iter_x++)
		{
			dx = x + iter_x;
			dy = y + iter_y;
			if (dx >= m_width) dx = m_width - 1;
			if (dy >= m_height) dy = m_height - 1;
			if (dx <0) dx = 0;
			if (dy<0) dy = 0;
			//|| depmask[dy*m_width + dx] == LEFTBOUN
			if (depmask[dy*m_width + dx] == RIGHTBOUN) {
				simflag = true;
				break;
			}
			/*else {
				cannyflag = true;
				break;
			}*/
		}

	double fdcsim = 0, gmmsim = 0,fdccount=0,gmmcount=0;
	Mat decSim(2 * (windowsize + 2) + 1, 2 * (windowsize + 2) + 1, CV_8UC3);
	int simx, simy;
	//构造小块图像
	Mat decMask(2 * (windowsize + 2) + 1, 2 * (windowsize + 2) + 1, CV_8UC1);
	for (int j = 0; j < decSim.rows; j++) {
		for (int i = 0; i < decSim.cols; i++) {
			simx = x - (windowsize + 2) + i;
			simy = y - (windowsize + 2) + j;
			decSim.at<Vec3b>(j, i) = m_pImage.at<Vec3b>(simy, simx);
		}
	}
	for (int j = 0; j < decSim.rows; j++) {
		for (int i = 0; i < decSim.cols; i++) {
			simx = x - (windowsize + 2) + i;
			simy = y - (windowsize + 2) + j;
			decMask.at<uchar>(j, i) = m_mark[simy*m_width + simx];
		}
	}
	bool thresholdflag = true;
	jishu++;
	while (thresholdflag) {
		for (int j = 2; j < decSim.rows - 2; j++) {
			for (int i = 2; i < decSim.cols - 2; i++) {
				if (decMask.at<uchar>(j, i) == 0) {
					simx = x - (windowsize + 2) + i;
					simy = y - (windowsize + 2) + j;
					fdcsim += pow(decSim.at<Vec3b>(j, i)[0] - m_FDC.at<Vec3b>(simy, simx)[0], 2);
					fdcsim += pow(decSim.at<Vec3b>(j, i)[1] - m_FDC.at<Vec3b>(simy, simx)[1], 2);
					fdcsim += pow(decSim.at<Vec3b>(j, i)[2] - m_FDC.at<Vec3b>(simy, simx)[2], 2);
					gmmsim += pow(decSim.at<Vec3b>(j, i)[0] - m_GMM.at<Vec3b>(simy, simx)[0], 2);
					gmmsim += pow(decSim.at<Vec3b>(j, i)[1] - m_GMM.at<Vec3b>(simy, simx)[1], 2);
					gmmsim += pow(decSim.at<Vec3b>(j, i)[2] - m_GMM.at<Vec3b>(simy, simx)[2], 2);
				}

			}
		}
		fdcsim = sqrt(fdcsim);
		gmmsim = sqrt(gmmsim);
		if (simflag) {
			/*thod = 15;
			fdcsim = 1;
			gmmsim = 1;*/
			thod = 250;
			if (fdcsim == 0)
				fdcsim = 1;
			if (gmmsim == 0)
				gmmsim = 1;
		}
		else {
				thod = 15;
				fdcsim = 1;
				gmmsim = 1;
		}
		

	double fdcgrad, gmmgrad;
	Mat fdccanny(2 * (windowsize + 2) + 1, 2 * (windowsize + 2) + 1, CV_8UC1);
	Mat gmmcanny(2 * (windowsize + 2) + 1, 2 * (windowsize + 2) + 1, CV_8UC1);
	Mat decEdge(2 * (windowsize + 2) + 1, 2 * (windowsize + 2) + 1, CV_8UC1);
	Mat fdcdecEdge(2 * (windowsize + 2) + 1, 2 * (windowsize + 2) + 1, CV_8UC3);
	Mat gmmdecEdge(2 * (windowsize + 2) + 1, 2 * (windowsize + 2) + 1, CV_8UC3);
	//添加fdcori,gmmori用于边界点的边缘计算
	Mat fdcori(2 * windowsize + 1, 2 * windowsize + 1, CV_8UC3);
	Mat gmmori(2 * windowsize + 1, 2 * windowsize + 1, CV_8UC3);

	int grayx, grayy;
	for (int j = 0; j < decEdge.rows; j++) {
		for (int i = 0; i < decEdge.cols; i++) {
			grayx = x - (windowsize + 2) + i;
			grayy = y - (windowsize + 2) + j;
			decEdge.at<uchar>(j, i) = m_gray[grayy*m_width + grayx];
			fdcdecEdge.at<Vec3b>(j, i) = m_pImage.at<Vec3b>(grayy, grayx);
			gmmdecEdge.at<Vec3b>(j, i) = m_pImage.at<Vec3b>(grayy, grayx);
		}
	}
	for (int j = 2; j < decEdge.rows - 2; j++) {
		for (int i = 2; i < decEdge.cols - 2; i++) {
			grayx = x - (windowsize + 2) + i;
			grayy = y - (windowsize + 2) + j;
			fdcori.at<Vec3b>(j - 2, i - 2)= m_FDC.at<Vec3b>(grayy, grayx);
			gmmori.at<Vec3b>(j - 2, i - 2) = m_GMM.at<Vec3b>(grayy, grayx);
			fdcdecEdge.at<Vec3b>(j, i) = m_FDC.at<Vec3b>(grayy, grayx);
			gmmdecEdge.at<Vec3b>(j, i) = m_GMM.at<Vec3b>(grayy, grayx);
			decEdge.at<uchar>(j, i) = (int)((mfdc_r[grayy*m_width + grayx] * 3735 + mfdc_g[grayy*m_width + grayx] * 19267 + mfdc_b[grayy*m_width + grayx] * 9765) / 32767);
		}
	}
	
	adpCanny(&fdcdecEdge, &fdccanny, CannyAccThresh);
	adpCanny(&gmmdecEdge, &gmmcanny, CannyAccThresh);
	String fdcedge = "fdc\\fdcedge" + format("%.4d", jishu) + ".jpg";
	String gmmedge = "gmm\\gmmedge" + format("%.4d", jishu) + ".jpg";
	imwrite(fdcedge, fdcdecEdge);
	imwrite(gmmedge, fdcdecEdge);
	
	String fdcnum = "fdc\\fdc" + format("%.4d",jishu)+ ".jpg";
	String gmmnum = "gmm\\gmm" + format("%.4d", jishu) + ".jpg";
	imwrite(fdcnum, fdccanny);
	imwrite(gmmnum, gmmcanny);
	
	//更改边缘点的计算方式为 fdcinpanum-fdcorinum
	Mat fdcoriEdge, gmmoriEdge;
	adpCanny(&fdcori, &fdcoriEdge, CannyAccThresh);
	adpCanny(&gmmori, &gmmoriEdge, CannyAccThresh);
	String orifdcnum = "fdc\\orifdcedge" + format("%.4d", jishu) + ".jpg";
	String origmmnum = "gmm\\origmmedge" + format("%.4d", jishu) + ".jpg";
	imwrite(orifdcnum, fdcoriEdge);
	imwrite(origmmnum, gmmoriEdge);
	String orifdcimg = "fdc\\orifdc" + format("%.4d", jishu) + ".jpg";
	String origmmimg = "gmm\\origmme" + format("%.4d", jishu) + ".jpg";
	imwrite(orifdcimg, fdcori);
	imwrite(origmmimg, gmmori);
	int fdcorinum = countedge(fdcoriEdge);
	int gmmorinum = countedge(gmmoriEdge);
	int fdcinpanum = countedge(fdccanny);
	int gmminpanum = countedge(gmmcanny);
	int cfdccount = fdcinpanum - fdcorinum;
	int cgmmcount = gmminpanum - gmmorinum;


	//if (MIN(cfdccount*fdcsim,cgmmcount*gmmsim) < thod || windowsize == 2)
		thresholdflag = false;
	//else windowsize = windowsize - 1;
	fdccount = cfdccount;
	gmmcount = cgmmcount;
	}
	int ran = rand();
	if (fdccount*fdcsim < gmmcount*gmmsim) return FDC;
	else if(fdccount*fdcsim > gmmcount*gmmsim)return GMM;
	else {
		if (fdcsim<=gmmsim) return FDC;
		else return GMM;
	}
}

int inpainting::SimPatch(int x, int y)
{
	
	double fdcsim=0, gmmsim=0;
	Mat decSim(2 * (windowsize + 2) + 1, 2 * (windowsize + 2) + 1, CV_8UC3);
	int simx, simy;
	Mat decMask(2 * (windowsize + 2) + 1, 2 * (windowsize + 2) + 1, CV_8UC1);
	for (int j = 0; j < decSim.rows; j++) {
		for (int i = 0; i < decSim.cols; i++) {
			simx = x - (windowsize + 2) + i;
			simy = y - (windowsize + 2) + j;
			decSim.at<Vec3b>(j, i) = m_pImage.at<Vec3b>(simy, simx);
		}
	}
	for (int j = 0; j < decSim.rows; j++) {
		for (int i = 0; i < decSim.cols; i++) {
			simx = x - (windowsize + 2) + i;
			simy = y - (windowsize + 2) + j;
			decMask.at<uchar>(j, i) = m_mark[simy*m_width + simx];
		}
	}
	bool thresholdflag = true;
	while (thresholdflag) {
		for (int j = 2; j < decSim.rows - 2; j++) {
			for (int i = 2; i < decSim.cols - 2; i++) {
				if (decMask.at<uchar>(j, i) == 0) {
					simx = x - (windowsize + 2) + i;
					simy = y - (windowsize + 2) + j;
					fdcsim += pow(decSim.at<Vec3b>(j, i)[0] - m_FDC.at<Vec3b>(simy, simx)[0], 2);
					fdcsim += pow(decSim.at<Vec3b>(j, i)[1] - m_FDC.at<Vec3b>(simy, simx)[1], 2);
					fdcsim += pow(decSim.at<Vec3b>(j, i)[2] - m_FDC.at<Vec3b>(simy, simx)[2], 2);
					gmmsim += pow(decSim.at<Vec3b>(j, i)[0] - m_GMM.at<Vec3b>(simy, simx)[0], 2);
					gmmsim += pow(decSim.at<Vec3b>(j, i)[1] - m_GMM.at<Vec3b>(simy, simx)[1], 2);
					gmmsim += pow(decSim.at<Vec3b>(j, i)[2] - m_GMM.at<Vec3b>(simy, simx)[2], 2);
				}

			}
		}
		fdcsim = sqrt(fdcsim);
		gmmsim = sqrt(gmmsim);
		//if (MIN(fdcsim, gmmsim) <= thold|| windowsize ==1)thresholdflag = false;
		//else windowsize = windowsize- 1;
		printf("%d\n", windowsize);
	}
	
	jishu++;
	printf("fdcsim:%lf,gmmsim:%lf jishu:%d\n", fdcsim, gmmsim, jishu);
	if (fdcsim <= gmmsim) return FDC;
	else return GMM;
}

double inpainting::countedge(Mat img)
{
	double countedg=0;
	for (int j = 1; j < img.rows-1 ; j++) {
		for (int i = 1; i < img.cols-1; i++) {
			if((j==2&&i<= img.cols - 2&& i>= 2)||(j==img.rows-2&&i <= img.cols - 2 && i >= 2)||(i==2 && j <= img.rows - 2 && j >= 2)||(i==img.cols - 2&&j <= img.rows - 2 && j >= 2)|| (j == 1 && i <= img.cols - 1 && i >= 1) || (j == img.rows-1&&i <= img.cols - 1 && i >= 1) || (i == 1 && j <= img.rows - 1 && j >= 1) || (i == img.cols - 1 && j <= img.rows - 1 && j >= 1)|| (j == 3 && i <= img.cols - 3 && i >= 3) || (j == img.rows - 3 && i <= img.cols - 3 && i >= 3) || (i == 3 && j <= img.rows - 3 && j >= 3) || (i == img.cols - 3 && j <= img.rows - 3 && j >= 3)){
			//if (mask.at<uchar>(j, i) == 0) {
				// 卷积区域 3*3
				if (img.at<uchar>(j, i) > 250)countedg++;

			}
		}
	}
	return countedg;
}

bool inpainting::update(int target_x, int target_y, int sourcepatch, double confid)
{
	//COLORREF color;
	double r, g, b;
	bool inpflag=false;
	int x0, y0;
	for (int iter_y = (-1)*windowsize; iter_y <= windowsize; iter_y++)
		for (int iter_x = (-1)*windowsize; iter_x <= windowsize; iter_x++)
		{
			x0 = target_x + iter_x;
			y0 = target_y + iter_y;
			if (x0 >= m_width) x0 = m_width - 1;
			if (y0 >= m_height) y0 = m_height - 1;
			if (x0 <0) x0 = 0;
			if (y0<0) y0 = 0;
			if (depmask[target_y*m_width + target_x] == LEFTBOUN || depmask[target_y*m_width + target_x] == RIGHTBOUN)
			/*if (depmask[x0*m_width + y0]==LEFTBOUN)*/ {
				inpflag = true; 
				break;
			}
		}
	//if (!inpflag) {
	//	printf("(%d,%d) bool:%d", target_x, target_y, inpflag);
	//	//system("pause");
	//}
	
	int x1, y1;
	if (inpflag) {
		double r=0, g=0, b=0;
		int pcount=0;
		for (int iter_y = (-1)*windowsize; iter_y <= windowsize; iter_y++)
			for (int iter_x = (-1)*windowsize; iter_x <= windowsize; iter_x++)
			{
				x1 = target_x + iter_x;
				y1 = target_y + iter_y;
				if (x1 >= m_width) x1 = m_width - 1;
				if (y1 >= m_height) y1 = m_height - 1;
				if (x1 <0) x1 = 0;
				if (y1<0) y1 = 0;
				if (m_mark[y1*m_width + x1]>=0)
				{
					r += m_r[y1*m_width + x1];
					g += m_g[y1*m_width + x1];
					b += m_b[y1*m_width + x1];
					pcount++;
				}
			}
		r = r / pcount;
		g = g / pcount;
		b = b / pcount;
		m_pImage.at<Vec3b>(target_y, target_x)[0] =b;// inpaint the color
		m_pImage.at<Vec3b>(target_y, target_x)[1] = g;
		m_pImage.at<Vec3b>(target_y, target_x)[2] = r;
		m_r[target_y*m_width + target_x] = r;
		m_g[target_y*m_width + target_x] = g;
		m_b[target_y*m_width + target_x] =b;
		m_mark[target_y*m_width + target_x] = SOURCE;
	}
	else {
	//if(inpflag)
	//printf("(%d,%d) flag:%d ", target_x, target_y, inpflag);
		if (sourcepatch == FDC) {
			for (int iter_y = (-1)*windowsize; iter_y <= windowsize; iter_y++)
				for (int iter_x = (-1)*windowsize; iter_x <= windowsize; iter_x++)
				{
					x1 = target_x + iter_x;
					y1 = target_y + iter_y;
					if (x1 >= m_width) x1 = m_width - 1;
					if (y1 >= m_height) y1 = m_height - 1;
					if (x1 <0) x1 = 0;
					if (y1<0) y1 = 0;
					//if (m_mark[y1*m_width + x1]<0)
					if (m_mark[y1*m_width + x1]<0 || inpflag)
					{
						m_pImage.at<Vec3b>(y1, x1) = m_FDC.at<Vec3b>(y1, x1);// inpaint the color
						m_r[y1*m_width + x1] = mfdc_r[y1*m_width + x1];
						m_g[y1*m_width + x1] = mfdc_g[y1*m_width + x1];
						m_b[y1*m_width + x1] = mfdc_b[y1*m_width + x1];
						m_gray[y1*m_width + x1] = (double)((m_r[y1*m_width + x1] * 3735 + m_g[y1*m_width + x1] * 19267 + m_b[y1*m_width + x1] * 9765) / 32767); // update gray image
						m_confid[y1*m_width + x1] = confid; // update the confidence
						m_mark[y1*m_width + x1] = SOURCE;
					}
				}
		}
		else if (sourcepatch == GMM) {
			for (int iter_y = (-1)*windowsize; iter_y <= windowsize; iter_y++)
				for (int iter_x = (-1)*windowsize; iter_x <= windowsize; iter_x++)
				{
					x1 = target_x + iter_x;
					y1 = target_y + iter_y;
					/*if (x1 >= 1024) x1 = 1023;
					if (y1 >= 768) x1 = 767;*/
					if (x1 >= m_width) x1 = m_width - 1;
					if (y1 >= m_height) y1 = m_height - 1;
					if (x1 <0) x1 = 0;
					if (y1<0) y1 = 0;
					//if (m_mark[y1*m_width + x1]<0)
					if (m_mark[y1*m_width + x1]<0 || inpflag)
					{
						m_pImage.at<Vec3b>(y1, x1) = m_GMM.at<Vec3b>(y1, x1);// inpaint the color
						m_r[y1*m_width + x1] = mgmm_r[y1*m_width + x1];
						m_g[y1*m_width + x1] = mgmm_g[y1*m_width + x1];
						m_b[y1*m_width + x1] = mgmm_b[y1*m_width + x1];
						m_gray[y1*m_width + x1] = (double)((m_r[y1*m_width + x1] * 3735 + m_g[y1*m_width + x1] * 19267 + m_b[y1*m_width + x1] * 9765) / 32767); // update gray image
						m_confid[y1*m_width + x1] = confid; // update the confidence
						m_mark[y1*m_width + x1] = SOURCE;
					}
				}
		}
		else {
			for (int iter_y = (-1)*windowsize; iter_y <= windowsize; iter_y++)
				for (int iter_x = (-1)*windowsize; iter_x <= windowsize; iter_x++)
				{
					x1 = target_x + iter_x;
					y1 = target_y + iter_y;
					/*if (x1 >= 1024) x1 = 1023;
					if (y1 >= 768) x1 = 767;*/
					if (x1 >= m_width) x1 = m_width - 1;
					if (y1 >= m_height) y1 = m_height - 1;
					if (x1 <0) x1 = 0;
					if (y1<0) y1 = 0;
					//if (m_mark[y1*m_width + x1]<0)
					if (m_mark[y1*m_width + x1]<0 || inpflag)
					{
						m_pImage.at<Vec3b>(y1, x1) = (m_GMM.at<Vec3b>(y1, x1)+m_FDC.at<Vec3b>(y1,x1))/2;// inpaint the color
						m_r[y1*m_width + x1] = (mgmm_r[y1*m_width + x1]+mfdc_r[y1*m_width + x1])/2;
						m_g[y1*m_width + x1] = (mgmm_g[y1*m_width + x1] + mfdc_g[y1*m_width + x1]) / 2;
						m_b[y1*m_width + x1] = (mgmm_b[y1*m_width + x1] + mfdc_b[y1*m_width + x1]) / 2;
						m_gray[y1*m_width + x1] = (double)((m_r[y1*m_width + x1] * 3735 + m_g[y1*m_width + x1] * 19267 + m_b[y1*m_width + x1] * 9765) / 32767); // update gray image
						m_confid[y1*m_width + x1] = confid; // update the confidence
						m_mark[y1*m_width + x1] = SOURCE;
					}
				}
		}
	}
	
	return true;
}

bool inpainting::TargetExist(void)
{
	for (int j = m_top; j <= m_bottom; j++)
		for (int i = m_left; i <= m_right; i++)if (m_mark[j*m_width + i]<0)return true;
	return false;
}

void inpainting::UpdateBoundary(int i, int j)
{
	int x, y;
	//COLORREF color;

	for (y = MAX(j - windowsize - 2, 0); y <= MIN(j + windowsize + 2, m_height - 1); y++)
		for (x = MAX(i - windowsize - 2, 0); x <= MIN(i + windowsize + 2, m_width - 1); x++)
		{  // if the pixel is specified as boundary
			//if (m_pImage.at<Vec3b>(y, x)[1] - m_pImage.at<Vec3b>(y, x)[0] >= color_threshold && m_pImage.at<Vec3b>(y, x)[1] - m_pImage.at<Vec3b>(y, x)[2] >= color_threshold)
			if(m_mark[y*m_width + x]<0)
				m_mark[y*m_width + x] = TARGET;
			else m_mark[y*m_width + x] = SOURCE;
		}

	for (y = MAX(j - windowsize - 2, 0); y <= MIN(j + windowsize + 2, m_height - 1); y++)
		for (x = MAX(i - windowsize - 2, 0); x <= MIN(i + windowsize + 2, m_width - 1); x++)
		{
			if (m_mark[y*m_width + x] == TARGET)
			{
				if (y == m_height - 1 || y == 0 || x == 0 || x == m_width - 1 || m_mark[(y - 1)*m_width + x] == SOURCE || m_mark[y*m_width + x - 1] == SOURCE
					|| m_mark[y*m_width + x + 1] == SOURCE || m_mark[(y + 1)*m_width + x] == SOURCE)m_mark[y*m_width + x] = BOUNDARY;
			}
		}
}

void inpainting::UpdatePri(int i, int j)
{
	int x, y;
	for (y = MAX(j - windowsize - 2, 0); y <= MIN(j + windowsize + 2, m_height - 1); y++)
		for (x = MAX(i - windowsize - 2, 0); x <= MIN(i + windowsize + 2, m_width - 1); x++)if (m_mark[y*m_width + x] == BOUNDARY)m_pri[y*m_width + x] = priority(x, y);
}
