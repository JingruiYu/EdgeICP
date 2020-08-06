/*************************************************************************
	> File Name: usingPcl.cpp
	> Author: 
	> Mail: 
	> Created Time: 2020年08月06日 星期四 19时06分46秒
 ************************************************************************/

#include<iostream>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

using namespace std;
using namespace cv;


static const double vehicle_length = 4.63;
static const double vehicle_width = 1.901;
static const double rear_axle_to_center = 1.393;
static const double pixel2meter = 0.03984;


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


void getCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_out)
{
	// Fill in the CloudIn data
    cloud_in->width    = 5;
    cloud_in->height   = 1;
    cloud_in->is_dense = false;
    cloud_in->points.resize (cloud_in->width * cloud_in->height);
    std::cout << "cloud_in->points.size (): " << cloud_in->points.size () << std::endl;
    int idx = 1;
    for (size_t i = 0; i < cloud_in->points.size (); ++i)
    {
        float x = idx * 1;//1024 * rand () / (RAND_MAX + 1.0f);
        float y = idx * 2;//1024 * rand () / (RAND_MAX + 1.0f);
        float z = idx * 3;//1024 * rand () / (RAND_MAX + 1.0f);
        cloud_in->points[i].x = x;
        cloud_in->points[i].y = y;
        cloud_in->points[i].z = z;
        std::cout << x << " " << y << " " << z << std::endl;
        idx++;
    }


    *cloud_out = *cloud_in;

    //performs a simple rigid transform on the point cloud
    for (size_t i = 0; i < cloud_in->points.size (); ++i)
        cloud_out->points[i].x = cloud_in->points[i].x + 1.5f;

}


void getResult(pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> &icp)
{
	//Creates a pcl::PointCloud<pcl::PointXYZ> to which the IterativeClosestPoint can save the resultant cloud after applying the algorithm
    pcl::PointCloud<pcl::PointXYZ> Final;

    //Call the registration algorithm which estimates the transformation and returns the transformed source (input) as output.
    icp.align(Final);

    //Return the state of convergence after the last align run. 
    //If the two PointClouds align correctly then icp.hasConverged() = 1 (true). 
    std::cout << "has converged: " << icp.hasConverged() <<std::endl;

    //Obtain the Euclidean fitness score (e.g., sum of squared distances from the source to the target) 
    std::cout << "score: " <<icp.getFitnessScore() << std::endl; 
    std::cout << "----------------------------------------------------------"<< std::endl;

    //Get the final transformation matrix estimated by the registration method. 
    std::cout << icp.getFinalTransformation() << std::endl;

}


void convertContourImageToCloud(Mat &image, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) 
{

  // convert to grayscale image
  if (image.channels() == 3)
    cvtColor(image, image, CV_RGB2GRAY);

  // get image info
  int frame_width = image.cols;
  int frame_height = image.rows;

  // convert pixels to points
  for (int row = 0; row < image.rows; ++row)
    for (int col = 0; col < image.cols; ++col) {
      if (image.at<uchar>(row, col) > 10) {
        pcl::PointXYZ point;
        point.x = (frame_height / 2 - row) * pixel2meter + rear_axle_to_center;
        point.y = (frame_width / 2 - col) * pixel2meter;
        point.z = 0.0;

        cloud->points.push_back(point);
      }
    }

  cloud->width = cloud->points.size();
  cloud->height = 1;
  cloud->is_dense = true;
}


int main(int argc, char const *argv[])
{
	vector<string> vstrcontourFilenames;
	vector<double> vTimestamps;
	vector<cv::Vec3d> vodomPose;

	string DataStrFile = string(argv[1])+"/associate.txt";
	
    LoadDataset(DataStrFile, vstrcontourFilenames, vodomPose, vTimestamps);

    for (size_t i = 0; i < vstrcontourFilenames.size()-1; i++)
    {
        cout << "i: " << i << endl;

        Mat refImg = imread(string(argv[1])+"/"+vstrcontourFilenames[i], CV_LOAD_IMAGE_GRAYSCALE);
        Mat curImg = imread(string(argv[1])+"/"+vstrcontourFilenames[i+1], CV_LOAD_IMAGE_GRAYSCALE);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZ>);

        convertContourImageToCloud(refImg, cloud_in);
        convertContourImageToCloud(curImg, cloud_out);

        //creates an instance of an IterativeClosestPoint and gives it some useful information
        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
        icp.setInputCloud(cloud_in);
        icp.setInputTarget(cloud_out);

        getResult(icp);

    }

	return 0;
}
