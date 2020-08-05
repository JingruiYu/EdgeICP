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

// #define Debug

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
    vector<Point> vPts;
    for (size_t i = 0; i < Img.cols; i++)
    {
        for (size_t j = 0; j < Img.rows; j++)
        {
            uchar rgb  = Img.at<uchar>(i,j);
            
            if (rgb > 250)
            {
                Point pt(i,j); 
                vPts.push_back(pt);
            }
        }
    }

    cv::Mat Pcl = Mat::zeros( vPts.size(), 6, CV_32F );

#ifdef Debug
    cv::Mat tmpimg = Mat::zeros( Img.size(), Img.type());
    for (size_t i = 0; i < vPts.size(); i++)
    {
        tmpimg.at<uchar>(vPts[i].x,vPts[i].y) = 255;
    }
    imshow("tmpImg", tmpimg);
    waitKey(0);

    cout << "vPts.size()" << vPts.size() << endl;
    cout << "Pcl.size()" << Pcl.size() << endl;
#endif

    for (size_t i = 0; i < vPts.size(); i++)
    {
        Pcl.at<float>(i,0) = vPts[i].x;
        Pcl.at<float>(i,1) = vPts[i].y;
        Pcl.at<float>(i,2) = 1.0;
        Pcl.at<float>(i,5) = 1.0;

#ifdef Debug        
        cout << "vPts[i]: " << vPts[i] << endl << endl;
        cout << Pcl << endl;
        
        getchar();
#endif

    }
    
    return Pcl.clone();
}


cv::Mat convert(cv::Mat &Pcl, cv::Matx44d &pose)
{
    cv::Mat Tpose = cv::Mat::zeros(4,4, CV_32F);

#ifdef Debug
    cout << "Tpose: " << endl << Tpose << endl;
#endif

    for (size_t i = 0; i < 4; i++)
    {
        for (size_t j = 0; j < 4; j++)
        {
            Tpose.at<float>(i,j) = pose(i,j);
        }
    }

#ifdef Debug
    cout << "Tpose: " << endl << Tpose << endl;
#endif

    cv::Mat PclNew = cv::Mat::zeros(Pcl.size(), Pcl.type());
    Pcl.copyTo(PclNew);
    for (size_t i = 0; i < Pcl.rows; i++)
    {
        cv::Mat subPt = Mat::zeros( 1, 4, CV_32F );
        subPt.at<float>(0,0) = Pcl.at<float>(i,0);
        subPt.at<float>(0,1) = Pcl.at<float>(i,1);
        subPt.at<float>(0,2) = Pcl.at<float>(i,2);

#ifdef Debug
        cout << " subPt: " << subPt << endl;
        cout << " pose: " << pose << endl;
#endif
        
        subPt = subPt * Tpose;

#ifdef Debug
        cout << " subPt: " << subPt << endl;
        getchar(); 
#endif

        PclNew.at<float>(i,0) = subPt.at<float>(0,0);
        PclNew.at<float>(i,1) = subPt.at<float>(0,1);
        PclNew.at<float>(i,2) = subPt.at<float>(0,2);
    }  

    return PclNew.clone();
}


void showResult(cv::Mat &curImg, cv::Mat &refPcl, cv::Mat &curPcl, cv::Mat &refPclNew, cv::Matx44d &pose)
{
    cv::Mat drawOldMat = cv::Mat::zeros(curImg.size(), curImg.type());
    cv::Mat drawNewMat = cv::Mat::zeros(curImg.size(), curImg.type());

    for (size_t i = 0; i < refPcl.rows; i++)
    {
        drawOldMat.at<uchar>(refPcl.at<float>(i,0),refPcl.at<float>(i,1)) = 255;
        drawNewMat.at<uchar>(refPclNew.at<float>(i,0),refPclNew.at<float>(i,1)) = 255;
    }
    

    



    imshow("refOldPcl",drawOldMat);
    imshow("refNewPcl",drawNewMat);
    imshow("curImg",curImg);

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

        // cout << "refPcl.type():" << refPcl.type() << endl;
        // cout << "refPcl.size():" << refPcl.size() << endl;

        int isuc = icp.registerModelToScene(refPcl,curPcl,res,pose);

        cout << " isuc: " << isuc << endl;
        cout << " res: " << res << endl;
        cout << " pose: " << pose << endl;
        //int isuc = cv::ppf_match_3d::ICP::registerModelToScene(refPcl,curPcl,res,pose);

        cv::Mat refPclNew = convert(refPcl,pose);

        showResult(curImg, refPcl, curPcl, refPclNew, pose);

        refImg = curImg.clone();
        refPcl = curPcl.clone();
        
        
        // https://github.com/opencv/opencv_contrib/blob/master/modules/surface_matching/samples/ppf_load_match.cpp
        // https://docs.opencv.org/3.4.5/dc/d9b/classcv_1_1ppf__match__3d_1_1ICP.html#accd9744cedf9cd9cd175d2c5bd77951e
    }
    

    return 0;
}
