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

    vector<Mat> srcImg_Grayed;
    namedWindow("Display Test");
    
    for (int i = 1; i < argc; i++){
        Mat grayedImg;
        cvtColor(imread(argv[i]), grayedImg, CV_RGB2GRAY);
        srcImg_Grayed.push_back(grayedImg);
    }
}
