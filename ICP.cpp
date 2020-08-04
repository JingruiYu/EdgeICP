/*************************************************************************
	> File Name: findcontour.cpp
	> Author: 
	> Mail: 
	> Created Time: Wed 29 Jul 2020 09:45:31 PM EDT
 ************************************************************************/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/surface_matching.hpp>
#include <opencv2/surface_matching/ppf_match_3d.hpp>

using namespace std;
using namespace cv;

void LoadDataset(const string &strFile, vector<string> &vstrcontourFilenames,
                vector<cv::Vec3d> &vodomPose, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            double x,y,theta;
            string image;
            ss >> t;
            vTimestamps.push_back(t);
            ss>>x>>y>>theta;
            vodomPose.push_back(cv::Vec3d(x,y,theta));
            ss >> image;
            vstrcontourFilenames.push_back("contour/"+image);
        }
    }
}


cv::Mat getPCL(cv::Mat &Img)
{
    cv::Mat Pcl = Mat::zeros( Img.size(), CV_8UC3 );

    return Pcl.clone();
}


cv::Mat convert(cv::Mat &Pcl, cv::Matx44d &pose)
{
    cv::Mat PclNew = Mat::zeros( Pcl.size(), CV_8UC3 );

    return PclNew.clone();
}


void showResult(cv::Mat &refPcl, cv::Mat &curPcl, cv::Mat &refPclNew, cv::Matx44d &pose)
{
    imshow("refPcl",refPcl);
    imshow("refPcl",curPcl);
    imshow("refPcl",refPclNew);

    waitKey(0);
}


int main(int argc, char const *argv[])
{
    vector<string> vstrcontourFilenames;
	vector<double> vTimestamps;
	vector<cv::Vec3d> vodomPose;

	string DataStrFile = string(argv[1])+"/associate.txt";
	
    LoadDataset(DataStrFile, vstrcontourFilenames, vodomPose, vTimestamps);

    cv::Mat refImg = imread(string(argv[1])+"/"+vstrcontourFilenames[0],CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat refPcl = getPCL(refImg);
    for (size_t i = 1; i < vstrcontourFilenames.size(); i++)
    {
        cv::Mat curImg = imread(string(argv[1])+"/"+vstrcontourFilenames[i],CV_LOAD_IMAGE_UNCHANGED);

        cv::Mat curPcl = getPCL(curImg);

        double res;
        cv::Matx44d pose;
        cv::ppf_match_3d::ICP icp(100, 0.005f, 2.5f, 8);
        int isuc = icp.registerModelToScene(refPcl,curPcl,res,pose);

        //int isuc = cv::ppf_match_3d::ICP::registerModelToScene(refPcl,curPcl,res,pose);

        cv::Mat refPclNew = convert(refPcl,pose);

        showResult(refPcl, curPcl, refPclNew, pose);

        refImg = curImg.clone();
        refPcl = curPcl.clone();
        
        
        // https://github.com/opencv/opencv_contrib/blob/master/modules/surface_matching/samples/ppf_load_match.cpp
        // https://docs.opencv.org/3.4.5/dc/d9b/classcv_1_1ppf__match__3d_1_1ICP.html#accd9744cedf9cd9cd175d2c5bd77951e
    }
    

    return 0;
}
