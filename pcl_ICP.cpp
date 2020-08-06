#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

int main (int argc, char** argv)
{
    //Creates two pcl::PointCloud<pcl::PointXYZ> boost shared pointers and initializes them
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZ>);

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

    //creates an instance of an IterativeClosestPoint and gives it some useful information
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputCloud(cloud_in);
    icp.setInputTarget(cloud_out);

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

    return (0);
}