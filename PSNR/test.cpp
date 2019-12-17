#include <iostream>
#include <vector>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\core\core.hpp>
using namespace std;
using namespace cv;

double PSNR(const Mat& I1, const Mat& I2) {
	Mat s1;
	absdiff(I1, I2, s1);
	s1.convertTo(s1, CV_32F);//转换为32位的float类型，8位不能计算平方
	s1 = s1.mul(s1);
	Scalar s = sum(s1);//计算每个通道的和
	double sse = s.val[0] + s.val[1] + s.val[2];
	if (sse <= 1e-10)// for small values return zero
		return 0;
	else {
		double mse = sse / (double)(I1.channels()*I1.total());//  sse/(w*h*3)
		double psnr = 10.0*log10(255 * 255 / mse);
		return psnr;
	}

}
Scalar MSSIM(const Mat& i1, const Mat& i2) {
	const double C1 = 6.5025, C2 = 58.5225;
	int d = CV_32F;

	Mat I1, I2;
	i1.convertTo(I1, d);
	i2.convertTo(I2, d);
	Mat I2_2 = I2.mul(I2);     // I2^2
	Mat I1_2 = I1.mul(I1);      //I1^2
	Mat I1_I2 = I1.mul(I2);     // I1*I2

	Mat mu1, mu2;
	GaussianBlur(I1, mu1, Size(11, 11), 1.5);
	GaussianBlur(I2, mu2, Size(11, 11), 1.5);

	Mat mu1_2 = mu1.mul(mu1);
	Mat mu2_2 = mu2.mul(mu2);
	Mat mu1_mu2 = mu1.mul(mu2);

	Mat sigma1_2, sigma2_2, sigma12;

	GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;

	GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
	sigma2_2 -= mu2_2;

	GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
	sigma12 -= mu1_mu2;

	Mat t1, t2, t3;
	t1 = 2 * mu1_mu2 + C1;
	t2 = 2 * sigma12 + C2;
	t3 = t1.mul(t2);

	t1 = mu1_2 + mu2_2 + C1;
	t2 = sigma1_2 + sigma2_2 + C2;
	t1 = t1.mul(t2);

	Mat ssim_map;
	divide(t3, t1, ssim_map);
	Scalar mssim = mean(ssim_map);
	return mssim;
}
int main() {
	int m_top,m_bottom,m_left,m_right;
	m_top =100;  // initialize the rectangle area
	m_bottom = 500;
	m_left = 200;
	m_right = 700;
	int height = m_bottom - m_top + 1;
	int width = m_right - m_left + 1;
	//"C:\\Data\\testsource\\3DVideos-distrib\\MSR3DVideo-Breakdancers\\cam1\\color-cam1-f000.jpg"
	for (int f = 0; f < 50; f++) {
		Mat ori = imread("C:\\Data\\testsource\\3DVideos-distrib\\BookArrival\\cam10\\color"+ format("%.2d", f) +".jpg");
		Mat img = imread("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\3Dwarp\\BookArrival\\06-10\\vimg" + format("%.2d", f) + ".jpg");
		Mat fdc = imread("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\3Dwarp\\BookArrival\\06-10\\(original)fdc+gmm\\vimg.bmp");
		Mat gmm = imread("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\3Dwarp\\BookArrival\\06-10\\gmm\\vimg.bmp");
		Mat fdcimg, gmmimg;
		Mat adapimg = imread("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\3Dwarp\\BookArrival\\06-10\\adaptive\\result_book" + format("%.2d", f) + ".bmp");
		Mat gmm_num = imread("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\BookArrival\\cam06_10\\GMM_Fill\\rgbvimg" + format("%.2d", f) + ".jpg");
		img.copyTo(fdcimg);
		img.copyTo(gmmimg);
		for (int x = 0; x < img.rows; x++)
			for (int y = 0; y < img.cols; y++) {
				if (img.at<Vec3b>(x, y)[1] - img.at<Vec3b>(x, y)[0] >= 50 || img.at<Vec3b>(x, y)[1] - img.at<Vec3b>(x, y)[2] >= 50) {
					fdcimg.at<Vec3b>(x, y) = fdc.at<Vec3b>(x, y);
					gmmimg.at<Vec3b>(x, y) = gmm.at<Vec3b>(x, y);
				}
			}
		Mat pori(height, width, CV_8UC3);
		Mat pfdc(height, width, CV_8UC3);
		Mat pgmm(height, width, CV_8UC3);
		Mat padap(height, width, CV_8UC3);
		Mat pgnum(height, width, CV_8UC3);
		for (int j = m_left; j <= m_right; j++)
		{
			for (int i = m_top; i <= m_bottom; i++)
			{
				int x = i - m_top;
				int y = j - m_left;
				pori.at<Vec3b>(x, y) = ori.at<Vec3b>(i, j);
				pfdc.at<Vec3b>(x, y) = fdcimg.at<Vec3b>(i, j);
				pgmm.at<Vec3b>(x, y) = gmmimg.at<Vec3b>(i, j);
				padap.at<Vec3b>(x, y) = adapimg.at<Vec3b>(i, j);
				pgnum.at<Vec3b>(x, y) = gmm_num.at<Vec3b>(i, j);
			}
		}
		imwrite("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\3Dwarp\\BookArrival\\test\\06-10\\pori" + format("%.2d", f) + ".jpg", pori);
		imwrite("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\3Dwarp\\BookArrival\\test\\06-10\\pfdc" + format("%.2d", f) + ".jpg", pfdc);
		imwrite("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\3Dwarp\\BookArrival\\test\\06-10\\pgmm" + format("%.2d", f) + ".jpg", pgmm);
		imwrite("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\3Dwarp\\BookArrival\\test\\06-10\\padap" + format("%.2d", f) + ".jpg", padap);
		imwrite("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\3Dwarp\\BookArrival\\test\\06-10\\pgnum" + format("%.2d", f) + ".jpg", pgnum);

		//imwrite("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\3Dwarp\\Breakdancers\\fdc\\fdcimg.jpg", fdcimg);
		//imwrite("C:\\Users\\a\\Documents\\Visual Studio 2015\\Outcome\\3Dwarp\\Breakdancers\\gmm\\gmmimg.jpg", gmmimg);
		//printf("chanel:%d,%d;size: %d,%d", ori.channels(), fdcimg.channels(),ori.size, fdcimg.size);
		printf("%d ", f);
		printf("%lf %lf %lf %lf ", PSNR(pori, pfdc), PSNR(pori, pgmm), PSNR(pori, padap), PSNR(pori, pgnum));
		Scalar result1 = MSSIM(pori, pfdc);
		Scalar result2 = MSSIM(pori, pgmm);
		Scalar result3 = MSSIM(pori, padap);
		Scalar result4 = MSSIM(pori, pgnum);
		printf("%lf %lf %lf %lf", (result1.val[0] + result1.val[1] + result1.val[2]) / 3, (result2.val[0] + result2.val[1] + result2.val[2]) / 3, (result3.val[0] + result3.val[1] + result3.val[2]) / 3, (result4.val[0] + result4.val[1] + result4.val[2]) / 3);
		/*if (fdcimg.channels() == 3)
			cout << " " << (result1.val[0] + result1.val[1] + result1.val[2]) / 3 << endl;
		else cout << " " << result1.val[0] << endl;
		if (gmmimg.channels() == 3)
			cout << " " << (result2.val[0] + result2.val[1] + result2.val[2]) / 3 << endl;
		else cout << "" << result2.val[0] << endl;
		if (adapimg.channels() == 3)
			cout << " " << (result3.val[0] + result3.val[1] + result3.val[2]) / 3 << endl;
		else cout << " " << result3.val[0] << endl;*/
		printf("\n");
	}
	
	system("pause");
}
