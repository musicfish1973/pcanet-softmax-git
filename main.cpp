#include <opencv2\opencv.hpp>
#include "utils.h"
#include <iostream>
#include <fstream>
#include <omp.h>
#include <string>
using namespace std;
using namespace cv;

#define elif else if
#define AT at<double>
#define MAX_ITER 100000

double cost = 0.0;
Mat grad;
double lrate = 0.1;
double lambda = 0.01;
int nclasses = 40;

Mat vec2mat(vector<vector<double> >&vec){
    int cols = vec.size();
    int rows = vec[0].size();
    Mat result(rows, cols, CV_64FC1);
    double *pData; 
    for(int i = 0; i<rows; i++){
        pData = result.ptr<double>(i);
        for(int j=0; j<cols; j++){
            pData[j] = vec[j][i];        
        }
    }
    return result;
}

Mat vec2colvec(vector<double>& vec){
    int length = vec.size();
    Mat A(length, 1, CV_64FC1);
    for(int i=0; i<length; i++){
        A.AT(i, 0) = vec[i];
    }
    return A;
}

Mat vec2rowvec(vector<double>& vec){
    Mat A = vec2colvec(vec);
    return A.t();
}

void update_CostFunction_and_Gradient(Mat x, Mat y, Mat weightsMatrix, double lambda){

    int nsamples = x.cols;
    int nfeatures = x.rows;
	//cout << nsamples << endl;
	//cout << nfeatures << endl;
    //calculate cost function
    Mat theta(weightsMatrix);
    Mat M = theta * x;
    Mat temp, temp2;
    temp = Mat::ones(1, M.cols, CV_64FC1);
    reduce(M, temp, 0, CV_REDUCE_SUM);
    temp2 = repeat(temp, nclasses, 1);
    M -= temp2;
    exp(M, M);
    temp = Mat::ones(1, M.cols, CV_64FC1);
    reduce(M, temp, 0, CV_REDUCE_SUM);
    temp2 = repeat(temp, nclasses, 1);
    divide(M, temp2, M); 
	//cout << "before groundTruth.mul(logM)..." << endl;
    Mat groundTruth = Mat::zeros(nclasses, nsamples, CV_64FC1);
    for(int i=0; i<nsamples; i++){
        groundTruth.AT(y.AT(0, i), i) = 1.0;
    }
    Mat logM;
    log(M, logM);
    temp = groundTruth.mul(logM);
	//cout << "after groundTruth.mul(logM)..." << endl;
    cost = - sum(temp)[0] / nsamples;
    Mat theta2;
    pow(theta, 2.0, theta2);
    cost += sum(theta2)[0] * lambda / 2;
    //calculate gradient
    temp = groundTruth - M;   
    temp = temp * x.t();
    grad = - temp / nsamples;
    grad += lambda * theta;

}

Mat calculateY(Mat x, Mat weightsMatrix){

    int nsamples = x.cols;
    int nfeatures = x.rows;
    //calculate cost function
    Mat theta(weightsMatrix);
    Mat M = theta * x;
    Mat temp, temp2;
    temp = Mat::ones(1, M.cols, CV_64FC1);
    reduce(M, temp, 0, CV_REDUCE_SUM);
    temp2 = repeat(temp, nclasses, 1);
    M -= temp2;
    exp(M, M);
    temp = Mat::ones(1, M.cols, CV_64FC1);
    reduce(M, temp, 0, CV_REDUCE_SUM);
    temp2 = repeat(temp, nclasses, 1);
    divide(M, temp2, M); 
    //log(M, M);//transform [0,1] to [-Inf,0]

    Mat result = Mat::ones(1, M.cols, CV_64FC1);
    for(int i=0; i<M.cols; i++){
        double maxele = M.AT(0, i);
        int which = 0;
        for(int j=1; j<M.rows; j++){
            if(M.AT(j, i) > maxele){
                maxele = M.AT(j, i);
                which = j;
            }
        }
    	cout << maxele << endl;//add by yhp for learning
        result.AT(0, i) = which;
    }
    return result;
}

void softmax(Mat x, vector<double> &vecY, Mat xT, vector<double>& testY){

    //int nsamples = vecX.size();
    //int nfeatures = vecX[0].size();
    int nsamples = x.cols;
    int nfeatures = x.rows;
	cout << nsamples << endl;
	cout << nfeatures << endl;
    //change vecX and vecY into matrix or vector.
    Mat y = vec2rowvec(vecY);
    //Mat x = vec2mat(vecX);
	//cout << y.rows << "," << y.cols << endl; 
	//cout << y << endl;
	//cout << x << endl;

    double init_epsilon = 0.12;
    Mat weightsMatrix = Mat::ones(nclasses, nfeatures, CV_64FC1);
    double *pData; 
    for(int i = 0; i<nclasses; i++){
        pData = weightsMatrix.ptr<double>(i);
        for(int j=0; j<nfeatures; j++){
            pData[j] = randu<double>();        
        }
    }
    weightsMatrix = weightsMatrix * (2 * init_epsilon) - init_epsilon;

    grad = Mat::zeros(nclasses, nfeatures, CV_64FC1);

/*
    //Gradient Checking (remember to disable this part after you're sure the 
    //cost function and dJ function are correct)
    update_CostFunction_and_Gradient(x, y, weightsMatrix, lambda);
    Mat dJ(grad);
//    grad.copyTo(dJ);
    cout<<"test!!!!"<<endl;
    double epsilon = 1e-4;
    for(int i=0; i<weightsMatrix.rows; i++){
        for(int j=0; j<weightsMatrix.cols; j++){
            double memo = weightsMatrix.AT(i, j);
            weightsMatrix.AT(i, j) = memo + epsilon;
            update_CostFunction_and_Gradient(x, y, weightsMatrix, lambda);
            double value1 = cost;
            weightsMatrix.AT(i, j) = memo - epsilon;
            update_CostFunction_and_Gradient(x, y, weightsMatrix, lambda);
            double value2 = cost;
            double tp = (value1 - value2) / (2 * epsilon);
            cout<<i<<", "<<j<<", "<<tp<<", "<<dJ.AT(i, j)<<", "<<dJ.AT(i, j) / tp<<endl;
            weightsMatrix.AT(i, j) = memo;
        }
    }
*/

    int converge = 0;
    double lastcost = 0.0;
    while(converge < MAX_ITER){
        update_CostFunction_and_Gradient(x, y, weightsMatrix, lambda);
		cout << "after update_CostFunction_and_Gradient()" << endl;
        weightsMatrix -= lrate * grad;

        cout<<"learning step: "<<converge<<", Cost function value = "<<cost<<endl;
        if(fabs((cost - lastcost) ) <= 5e-6 && converge > 0) break;
        lastcost = cost;
        ++ converge;
    }
    cout<<"############result#############"<<endl;

    Mat yT = vec2rowvec(testY);
    //Mat xT = vec2mat(testX);
    Mat result = calculateY(xT, weightsMatrix);
    Mat err(yT);
    err -= result;
    int correct = err.cols;
    for(int i=0; i<err.cols; i++){
        if(err.AT(0, i) != 0) --correct;
    }
    cout<<"correct: "<<correct<<", total: "<<err.cols<<", accuracy: "<<double(correct) / (double)(err.cols)<<endl;
}

int main(int argc, char** argv){
	const int DIR_LENGTH = 256;
	const int DIR_NUM = 40;
	//input image size height: 60, width: 48
	// 路径根据自己的情况来修改即可
	const char *dir[DIR_NUM] = {
		"D:\\pcanet\\data-model\\att\\datas\\train\\s1\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s2\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s3\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s4\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s5\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s6\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s7\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s8\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s9\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s10\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s11\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s12\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s13\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s14\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s15\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s16\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s17\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s18\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s19\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s20\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s21\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s22\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s23\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s24\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s25\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s26\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s27\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s28\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s29\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s30\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s31\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s32\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s33\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s34\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s35\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s36\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s37\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s38\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s39\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s40\\"
	};

	const char *test_dir[DIR_NUM] = {
		"D:\\pcanet\\data-model\\att\\datas\\train\\s1\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s2\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s3\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s4\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s5\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s6\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s7\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s8\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s9\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s10\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s11\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s12\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s13\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s14\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s15\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s16\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s17\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s18\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s19\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s20\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s21\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s22\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s23\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s24\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s25\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s26\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s27\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s28\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s29\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s30\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s31\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s32\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s33\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s34\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s35\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s36\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s37\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s38\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s39\\",
		"D:\\pcanet\\data-model\\att\\datas\\train\\s40\\"
	};
	
	char path[DIR_LENGTH];
	IplImage* img;
	IplImage *change;
	vector<cv::Mat> InImgs;
	cv::Mat* bmtx;
	cv::Mat* histo; // histogram equalizaion  
	
	const int train_num = 5;
	const int NUM = DIR_NUM * train_num;

	
	float *labels = new float[NUM]; 
	int x = 0;
	for(int i=1; i<train_num + 1; i++){
		for(int j=1; j<=DIR_NUM; j++){
			sprintf(path, "%s%d%s", dir[j-1], i, ".pgm");
			cout << path << endl;
			img = cvLoadImage(path, CV_LOAD_IMAGE_GRAYSCALE);
			
			change = cvCreateImage(cvGetSize(img), IPL_DEPTH_64F, img->nChannels);
			cvConvertScale(img, change, 1.0/255, 0);
			bmtx = new cv::Mat(change);
			InImgs.push_back(*bmtx);
			labels[x] = (float)(j-1);
			x++;
		}
	}
	
	
	vector<int> NumFilters;
	NumFilters.push_back(25);
	NumFilters.push_back(8);
	vector<int> blockSize;
	blockSize.push_back(8);   //  height / 4
	blockSize.push_back(8);    //  width / 4
	

	PCANet pcaNet = {
		2,
		5,
		NumFilters,
		blockSize,
		0//0.5
	};
	
	cout <<"\n ====== PCANet Training ======= \n"<<endl;
	int64 e1 = cv::getTickCount();
	PCA_Train_Result* result = PCANet_train(InImgs, &pcaNet, true);
	int64 e2 = cv::getTickCount();
	double time = (e2 - e1)/ cv::getTickFrequency();
	cout <<" PCANet Training time: "<<time<<endl;

	
	/*FileStorage fs("..\\model\\all_age_filters.xml", FileStorage::WRITE);  
	fs<<"filter1"<<result->Filters[0]<<"filter2"<<result->Filters[1];  
	fs.release();*/

	//prepare data for softmax
	//train
	float *new_labels = new float[NUM];
	int size = result->feature_idx.size();
    vector<double> vecY;
	//cout << size << endl;
	for(int i=0; i<size; i++) {
		new_labels[i] = labels[result->feature_idx[i]];
        vecY.push_back(new_labels[i]);
	}
	//result->Features.convertTo(result->Features, CV_32F);
	result->Features.convertTo(result->Features, CV_64F);

	//test
	vector<Mat> testImg;
	vector<double> testLabel;
	vector<string> names;
	string *t;

	int testNum = 5;

	for(int i=6; i<11; i++){
		for(int j=0; j<DIR_NUM; j++){
			sprintf(path, "%s%d%s", test_dir[j], i, ".pgm");
			img = cvLoadImage(path, CV_LOAD_IMAGE_GRAYSCALE);
			t = new string(path);
			names.push_back(*t);
			change = cvCreateImage(cvGetSize(img), IPL_DEPTH_64F, img->nChannels);
			cvConvertScale(img, change, 1.0/255, 0);
			bmtx = new cv::Mat(change);
			testImg.push_back(*bmtx);
			testLabel.push_back(j);
		}
	}
	int testSIze = testImg.size();
	Hashing_Result* hashing_r;
	PCA_Out_Result *out;
	Mat matTX=Mat::zeros(testSIze,result->Features.cols,CV_64F);

	float all = DIR_NUM * testNum;
	int coreNum = omp_get_num_procs();//获得处理器个数
	cout << "coreNum = " << coreNum << endl;

	e1 = cv::getTickCount();
# pragma omp parallel for default(none) num_threads(coreNum) private(out, hashing_r) shared(names, corrs, correct, testLabel, SVM, pcaNet, testSIze, testImg, result)
	for(int i=0; i<testSIze; i++){
		cout << i << endl;
		out = new PCA_Out_Result;
		out->OutImgIdx.push_back(0);
		out->OutImg.push_back(testImg[i]);
		out = PCA_output(out->OutImg, out->OutImgIdx, pcaNet.PatchSize, 
			pcaNet.NumFilters[0], result->Filters[0], 2);
		for(int j=1; j<pcaNet.NumFilters[0]; j++)
			out->OutImgIdx.push_back(j);

		out = PCA_output(out->OutImg, out->OutImgIdx, pcaNet.PatchSize, 
			pcaNet.NumFilters[1], result->Filters[1], 2);		
		hashing_r = HashingHist(&pcaNet, out->OutImgIdx, out->OutImg);	
		//hashing_r->Features.convertTo(hashing_r->Features, CV_32F);
		hashing_r->Features.convertTo(hashing_r->Features, CV_64F);
		//matTX.at<double>(i)=hashing_r->Features.at<double>(0);
		//matTX.row(i) = hashing_r->Features.row(0);
		hashing_r->Features.row(0).copyTo(matTX.row(i));
#pragma omp critical 
		delete out;
	}
	//cout << hashing_r->Features.rows << "," << hashing_r->Features.cols << endl;

	/*Mat xxt=result->Features.t();
	Mat yyt=matTX.t();
    softmax(xxt, vecY, yyt, testLabel);*/
	normalize(result->Features,result->Features,0,1,NORM_MINMAX);
	normalize(matTX,matTX,0,1,NORM_MINMAX);
	//cout << result->Features.t() << endl;
	//cout << result->Features.at<double>(0) << endl;
	//cout << matTX.at<double>(0) << endl;
	//cout << result->Features.row(0) << endl;
	//cout << matTX.row(0) << endl;
    softmax(result->Features.t(), vecY, matTX.t(), testLabel);

	return 0;
}
