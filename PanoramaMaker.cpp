//
//  main.cpp
//  PanoramaMaker
//
//  Created by Dar Mehta on 2015-12-31.
//  Copyright © 2015 Dar Mehta. All rights reserved.
//

#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <unistd.h>

using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {
    if (argc < 3){
        cout << "Invalid arguments. Input more images." << endl;
        return -1;
    }

    vector<Mat> srcImg_Grayed, descriptors;
    vector<vector<KeyPoint> > keypoints;
    Mat drawnKeypoints;
    vector<vector<DMatch> > matches;
    namedWindow("Display Test");
    
    for (int i = 1; i < argc; i++){
        Mat grayedImg;
        cvtColor(imread(argv[i]), grayedImg, CV_RGB2GRAY);
        srcImg_Grayed.push_back(grayedImg);
    }
    
    int minHessian = 1000;
    Ptr<xfeatures2d::SURF> surfDetector = xfeatures2d::SURF::create(minHessian);
    surfDetector->detect(srcImg_Grayed, keypoints);
    
    Ptr<xfeatures2d::SurfDescriptorExtractor> surfExtractor = xfeatures2d::SurfDescriptorExtractor::create();
    surfExtractor->compute(srcImg_Grayed, keypoints, descriptors);
    
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
    for (int i = 0; i < argc - 2; i ++){
        cout << "Hello " << i << endl;
        matcher->match(descriptors[i], descriptors[i+1], matches[i]);
        cout << "End " << i << endl;
    }

    drawMatches(srcImg_Grayed[0], keypoints[0], srcImg_Grayed[1], keypoints[1], matches, drawnKeypoints, Scalar(0, 0, 255), Scalar(255, 0, 0));
    
    imshow("Display Test", drawnKeypoints);
    waitKey(0);
}
