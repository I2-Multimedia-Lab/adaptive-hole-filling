
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

//EM���� nTimes
double ClassGMM::EM(int nTimes)
{
	double em=0;
	int i, j, k;
	//���ٿռ�
	double **ppR;
	ppR = new double*[numSample];
	for (i = 0; i<numSample; i++)
	{
		ppR[i] = new double[numClass];
	}


	while (--nTimes >= 0)
	{
		//����r(i,k)����ppR[i,k]��ʾ
		for (i = 0; i<numSample; i++)
		{
			double *piKN;
			piKN = new double[numClass];//
			double sum_piKN = 0;
			for (k = 0; k<numClass; k++)
			{
				piKN[k] = pGM[k].piK*getGaussRatio(pGM[k].u, pGM[k].sigma, pSampleNode[i].x);//Ȩ��*�����ܶ�
				sum_piKN += piKN[k];
			}
			for (k = 0; k<numClass; k++)
			{
				ppR[i][k] = piKN[k] / sum_piKN;//����rik��ֵ
				em += ppR[i][k] * (log(pGM[k].piK) + log(getGaussRatio(pGM[k].u, pGM[k].sigma, pSampleNode[i].x)));
			}
			delete[]piKN;
		}
		//����r(i,k)���˹������Nk��piK��u��sigma��
		//Nk
		for (k = 0; k<numClass; k++)
		{
			pGM[k].Nk = 0;
			for (i = 0; i<numSample; i++)
			{
				pGM[k].Nk += ppR[i][k];//rik�Ķ�i���ۼ�
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
	//���ٿռ�
	for (i = 0; i<numSample; i++)
	{
		delete[]ppR[i];
	}
	delete[]ppR;

	return em;
}

//���˹�����ܶ�
double ClassGMM::getGaussRatio(double u, double sigma, double x)
{
	double fenMu = sqrt(2 * PI)*sigma;
	double zhiShu = -(x - u)*(x - u) / (2 * sigma*sigma);
	double fenZi = exp(zhiShu);
	return fenZi / fenMu;
}
