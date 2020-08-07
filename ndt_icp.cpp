#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include <Eigen/Dense>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <pcl/point_types_conversion.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>
#include <pcl/console/time.h> 

#include <pclomp/ndt_omp.h>

typedef pcl::PointXYZ Pt;
typedef pcl::PointCloud<pcl::PointXYZ> Cloud;

using namespace std;
using namespace cv;
using namespace Eigen;

static const double vehicle_length = 4.63;
static const double vehicle_width = 1.901;
static const double rear_axle_to_center = 1.393;
static const double pixel2meter = 0.03984;

void convertContourImageToCloud(Mat &image, Cloud::Ptr cloud) {

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
        Pt point;
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

int main(int argc, char **argv) {

  if (argc != 3) {
    cerr << "Usage: ./birdeye_semantic_odometry ref_image.jpg query_image.jpg"
         << endl;
    return 1;
  }

  // read images
  Mat ref_image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
  Mat query_image = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);

  // convert to point cloud
  Cloud::Ptr ref_cloud(new Cloud());
  Cloud::Ptr query_cloud(new Cloud());

  convertContourImageToCloud(ref_image, ref_cloud);
  convertContourImageToCloud(query_image, query_cloud);

  // estimate transformation matrix
  // -- create ndt object
  pclomp::NormalDistributionsTransform<Pt, Pt>::Ptr ndt_omp(
      new pclomp::NormalDistributionsTransform<Pt, Pt>());
  ndt_omp->setStepSize(0.1);
  ndt_omp->setResolution(1.0);

  // -- align
  Matrix4f initTransform = Matrix4f::Identity();
  ndt_omp->setInputSource(query_cloud);
  ndt_omp->setInputTarget(ref_cloud);

  cout << "initial guess: \n" << initTransform << endl;

  Cloud::Ptr aligned_cloud(new Cloud());
  ndt_omp->align(*aligned_cloud, initTransform);

  Matrix4f finalTransform = ndt_omp->getFinalTransformation();
  cout << "final transform: \n" << finalTransform << endl;
  cout << "score: " << ndt_omp->getFitnessScore() << endl;

  // visualization
  pcl::visualization::PCLVisualizer vis("viewer");
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> ref_handler(
      ref_cloud, 0.0, 255.0, 0.0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
      aligned_handler(query_cloud, 255.0, 0.0, 0.0);
  vis.addPointCloud(ref_cloud, ref_handler, "ref_cloud");
  vis.addPointCloud(aligned_cloud, aligned_handler, "aligned_cloud");

  vis.addCube(-(vehicle_length / 2 - rear_axle_to_center),
              vehicle_length / 2 + rear_axle_to_center, -vehicle_width / 2,
              vehicle_width / 2, 0.0, 0.2, 255.0, 140.0, 0.0, "vehicle");
  vis.addCoordinateSystem(1.0, 0.0, 0.0, 0.3);
  vis.spin();

  return 0;
}