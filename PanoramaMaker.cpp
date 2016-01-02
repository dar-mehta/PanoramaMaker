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

void makePanorama(Mat, Mat, Mat &);
void findKeypointsAndDescriptors (Mat, Mat, vector<KeyPoint> &, Mat &, vector<KeyPoint> &, Mat &);
void findMatchedPoints (vector<KeyPoint> &, Mat &, vector<KeyPoint> &, Mat &, vector<DMatch> &, vector<Point2f> &, vector<Point2f> &);
void stitchTogether (Mat, Mat, vector<Point2f> &, vector<Point2f> &, Mat &);
void testCropB (Mat &);

int main(int argc, const char * argv[]) {
    if (argc < 3){
        cout << "Invalid arguments. Input more images." << endl;
        return -1;
    }
    
    vector<Mat> srcImg_Grayed;
    Mat result, result1, result2;
    namedWindow("Panorama");
    std::stringstream sr, si1, si2;
    
    for (int i = 1; i < argc; i++){
        Mat grayedImg;
        srcImg_Grayed.push_back(imread(argv[i]));
    }
    
    makePanorama(srcImg_Grayed[1], srcImg_Grayed[0], result1);
    testCropB(result1);
    makePanorama(srcImg_Grayed[3], srcImg_Grayed[2], result2);
    testCropB(result2);
    
    makePanorama(result2, result1, result);
    imshow("Panorama", result);
    waitKey(0);
    
}

void makePanorama(Mat imgObj, Mat imgScene, Mat & result){
    
    vector<KeyPoint> objKeypoints, sceneKeypoints;
    Mat drawnKeypoints, objDescriptors, sceneDescriptors;
    vector<DMatch> matches;
    vector<Point2f> objMatchedPoints, sceneMatchedPoints;
    
    findKeypointsAndDescriptors(imgObj, imgScene, objKeypoints, objDescriptors, sceneKeypoints, sceneDescriptors);
    
    findMatchedPoints(objKeypoints, objDescriptors, sceneKeypoints, sceneDescriptors, matches, objMatchedPoints, sceneMatchedPoints);
    
    stitchTogether(imgObj, imgScene, objMatchedPoints, sceneMatchedPoints, result);
}

//Finding keypoints & descriptors using SURF
void findKeypointsAndDescriptors (Mat imgO, Mat imgS, vector<KeyPoint> &oK, Mat &oD, vector<KeyPoint> &sK, Mat &sD){
    
    const int minHessian = 2500;
    Ptr<xfeatures2d::SURF> detectorAndExtractor = xfeatures2d::SURF::create(minHessian);
    
    detectorAndExtractor->detectAndCompute(imgO, noArray(), oK, oD);
    detectorAndExtractor->detectAndCompute(imgS, noArray(), sK, sD);
}

//Finding matches using FLANN
//Filtering matches using David Lowe's Algorithm from his paper "Distinctive Image Features
//from Scale-Invariant Keypoints"
void findMatchedPoints (vector<KeyPoint> &oK, Mat &oD, vector<KeyPoint> &sK, Mat &sD, vector<DMatch> &m, vector<Point2f> &oP, vector<Point2f> &sP){
    
    const float lowe_ratio = 0.45;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
    vector<DMatch> goodMatches;
    goodMatches.reserve(m.size());
    
    matcher->match(oD, sD, m);
    for (int i = 0; i < m.size(); i++){
        if (m[i].distance < lowe_ratio * m[i+1].distance){
            goodMatches.push_back(m[i]);
        }
    }
    
    for (int i = 0; i < goodMatches.size(); i++){
        oP.push_back(oK[goodMatches[i].queryIdx].pt);
        sP.push_back(sK[goodMatches[i].trainIdx].pt);
    }
}

void stitchTogether (Mat imgO, Mat imgS, vector<Point2f> &oP, vector<Point2f> &sP, Mat & result){
    
    Mat homographyMat = findHomography(oP, sP, CV_RANSAC);
    warpPerspective(imgO,result,homographyMat,cv::Size(imgO.cols+imgS.cols,imgO.rows));
    cv::Mat half(result,cv::Rect(0,0,imgS.cols,imgS.rows));
    imgS.copyTo(half);
}

void testCropB(cv::Mat& image)
{
    cv::Mat gray;
    cvtColor(image, gray, CV_BGR2GRAY);
    
    int minCol = 0;
    int minRow = 0;
    int maxCol = gray.cols;
    int maxRow = gray.rows;
    bool noBlackPixels = true;
    
    cout << "A: " << minCol << " " << maxCol << " " << endl;
    cout << "A: " << minRow << " " << maxRow << " " << endl;
    
    for (int i = gray.cols; i > 0; i--)
    {
        noBlackPixels = true;
        for (int j = 0; j < gray.rows && noBlackPixels; j++)
        {
            if (gray.at<char>(j, i) == 0)
            {
                noBlackPixels = false;
            }
        }
        if (!noBlackPixels && maxCol <= gray.cols){
            maxCol--;
        }
        
    }
    
    cout << "B: " << minCol << " " << maxCol << " " << endl;
    cout << "B: " << minRow << " " << maxRow << " " << endl;
    
    cv::Rect cropRect = Rect(minCol, minRow, maxCol, gray.rows);
    image = image(cropRect);
}
