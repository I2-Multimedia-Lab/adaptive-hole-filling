
#include <cmath>
#include "ClassGMM.h"

#define PI acos(-1.0)


ClassGMM::ClassGMM(void)
{
	numSample = 0;
	numClass = 0;
	pGM = NULL;
	pSampleNode = NULL;
}


ClassGMM::~ClassGMM(void)
{
	if (pGM != NULL)
	{
		delete[]pGM;
		pGM = NULL;
	}
	if (pSampleNode != NULL)
	{
		delete[]pSampleNode;
		pSampleNode = NULL;
	}
}


void ClassGMM::clear(void)
{
	numSample = 0;
	numClass = 0;
	if (pGM != NULL)
	{
		delete[]pGM;
		pGM = NULL;
	}
	if (pSampleNode != NULL)
	{
		delete[]pSampleNode;
		pSampleNode = NULL;
	}
}

//EM迭代 nTimes
double ClassGMM::EM(int nTimes)
{
	double em=0;
	int i, j, k;
	//开辟空间
	double **ppR;
	ppR = new double*[numSample];
	for (i = 0; i<numSample; i++)
	{
		ppR[i] = new double[numClass];
	}


	while (--nTimes >= 0)
	{
		//计算r(i,k)，用ppR[i,k]表示
		for (i = 0; i<numSample; i++)
		{
			double *piKN;
			piKN = new double[numClass];//
			double sum_piKN = 0;
			for (k = 0; k<numClass; k++)
			{
				piKN[k] = pGM[k].piK*getGaussRatio(pGM[k].u, pGM[k].sigma, pSampleNode[i].x);//权重*概率密度
				sum_piKN += piKN[k];
			}
			for (k = 0; k<numClass; k++)
			{
				ppR[i][k] = piKN[k] / sum_piKN;//计算rik的值
				em += ppR[i][k] * (log(pGM[k].piK) + log(getGaussRatio(pGM[k].u, pGM[k].sigma, pSampleNode[i].x)));
			}
			delete[]piKN;
		}
		//利用r(i,k)求高斯参数，Nk，piK，u，sigma，
		//Nk
		for (k = 0; k<numClass; k++)
		{
			pGM[k].Nk = 0;
			for (i = 0; i<numSample; i++)
			{
				pGM[k].Nk += ppR[i][k];//rik的对i的累加
			}
		}
		//piK
		double sumNK = 0;
		for (k = 0; k<numClass; k++)
		{
			sumNK += pGM[k].Nk;
		}
		for (k = 0; k<numClass; k++)
		{
			pGM[k].piK = pGM[k].Nk / sumNK;
		}
		//u
		for (k = 0; k<numClass; k++)
		{
			pGM[k].u = 0;
			for (i = 0; i<numSample; i++)
			{
				pGM[k].u += pSampleNode[i].x*ppR[i][k];
			}
			pGM[k].u /= pGM[k].Nk;
		}
		//sigma
		for (k = 0; k<numClass; k++)
		{
			double sum_temp = 0;
			for (i = 0; i<numSample; i++)
			{
				sum_temp += ppR[i][k] * (pSampleNode[i].x - pGM[k].u)*(pSampleNode[i].x - pGM[k].u);
			}
			sum_temp /= pGM[k].Nk;
			pGM[k].sigma = sqrt(sum_temp);
		}
	}
	//销毁空间
	for (i = 0; i<numSample; i++)
	{
		delete[]ppR[i];
	}
	delete[]ppR;

	return em;
}

//求高斯概率密度
double ClassGMM::getGaussRatio(double u, double sigma, double x)
{
	double fenMu = sqrt(2 * PI)*sigma;
	double zhiShu = -(x - u)*(x - u) / (2 * sigma*sigma);
	double fenZi = exp(zhiShu);
	return fenZi / fenMu;
}
