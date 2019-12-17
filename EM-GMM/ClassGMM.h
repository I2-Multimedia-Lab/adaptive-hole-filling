#pragma once
struct struct_sampleNode
{
	double x;
};
struct struct_GM
{
	double Nk;
	double piK;
	double u;
	double sigma;
};
class ClassGMM
{
public:
	struct_GM* pGM;
	struct_sampleNode* pSampleNode;
	int numSample;
	int numClass;
	ClassGMM(void);
	~ClassGMM(void);
	void clear(void);
	double EM(int nTimes);
	double getGaussRatio(double u, double sigma, double x);
};

