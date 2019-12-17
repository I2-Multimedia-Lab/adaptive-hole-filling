#pragma once

/*Author: Qiushuang Zhang */
/*E-mail: qszhang@cc.gatech.edu */
/*Nov.29, 2005 */

#ifndef INPAINTING_H
#define INPAINTING_H

#define save_path ".\\result"
#define save_path1 ".\\rresult"

#define PAINT_COLOR  Vec3b color=(0,255,0);  // the color used to specify the target area
#define SOURCE 0
#define TARGET -1
#define BOUNDARY -2
#define BACKBOUN -3
#define FRONTBOUN -4
#define LEFTBOUN -5
#define RIGHTBOUN -6
#define depwindowsize 3  // the window size
#define boud 50
#define FDC 13
#define GMM 17
#define BOTH 15
#define thold 250
#define color_threshold 50
#include <opencv/cv.h>  
#include <opencv2/highgui/highgui.hpp> 
using namespace cv;

typedef struct
{
	double grad_x;
	double grad_y;
}gradient; //the structure that record the gradient

typedef struct
{
	double norm_x;
	double norm_y;
}mynorm;  // the structure that record the norm

class inpainting
{
public:

	Mat img;
	Mat m_pImage; // the image to be inpainted
	Mat m_FDC;
	Mat m_GMM;
	Mat m_FDCdep;
	Mat m_GMMdep;
	Mat m_DepBound;
	Mat m_DepInpaint;
	Mat deptest;
	int Frame;

	int windowsize;
	bool m_bOpen; // whether it is successfully opened
	int m_width; // image width
	int m_height; // image height
	int jishu;
	//COLORREF * m_color;
	double * m_r;
	double * m_g;
	double * m_b;
	double * mfdc_r;
	double * mfdc_g;
	double * mfdc_b;
	double * mdfdc;
	double * mgmm_r;
	double * mgmm_g;
	double * mgmm_b;
	double * mdgmm;
	int * depmask;

	int m_top, m_bottom, m_left, m_right; // the rectangle of inpaint area

	int * m_mark;// mark it as source(0) or to-be-inpainted area(-1) or bondary(-2).
	double * m_confid;// record the confidence for every pixel
	double * m_pri; // record the priority for pixels. only boudary pixels will be used
	double * m_gray; // the gray image
	bool * m_source; // whether this pixel can be used as an example texture center
	int *m_dep;

	inpainting(char * imgname, char * depname, char * FDCname, char * GMMname,int f);
	~inpainting(void);
	bool process(void);  // the main function to process the whole image
	void DrawBoundary(void);  // the first time to draw boundary on the image.
	void dep_inpaint(Mat dep, Mat *depinpaint);
	double ComputeConfidence(int i, int j); // the function to compute confidence
	double priority(int x, int y); // the function to compute priority
	double ComputeData(int i, int j);//the function to compute data item
	double ComputeDepth(int i, int j);
	void Convert2Gray(void);  // convert the input image to gray image
	gradient GetGradient(int i, int j); // calculate the gradient at one pixel
	mynorm GetNorm(int i, int j);  // calculate the norm at one pixel ·¶Êý
	bool draw_source(void);  // find out all the pixels that can be used as an example texture center
	int PatchTexture(int x, int y);// find the most similar patch from sources.
	int CannyPatch(int x, int y);
	int SimPatch(int x, int y);
	double countedge(Mat img);
	bool update(int target_x, int target_y, int sourcepatch, double confid);// inpaint this patch and update pixels' confidence within this area
	bool TargetExist(void);// test whether this is still some area to be inpainted.
	void UpdateBoundary(int i, int j);// update boundary
	void UpdatePri(int i, int j); //update priority for boundary pixels.
};


#endif
