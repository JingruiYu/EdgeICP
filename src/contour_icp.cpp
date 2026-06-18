#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

struct FrameRecord {
  double timestamp = 0.0;
  double odometry_x = 0.0;
  double odometry_y = 0.0;
  double odometry_yaw = 0.0;
  std::string contour_image_path;
};

struct RigidTransform2D {
  double rotation = 0.0;
  cv::Point2d translation = cv::Point2d(0.0, 0.0);
};

struct IcpResult {
  bool converged = false;
  int iterations = 0;
  double mean_error = std::numeric_limits<double>::infinity();
  RigidTransform2D transform;
};

constexpr double kRearAxleToVehicleCenterMeters = 1.393;
constexpr double kPixelToMeter = 0.03984;
constexpr int kMaximumPointCount = 3000;
constexpr int kMaximumIterations = 40;
constexpr double kConvergenceTolerance = 1e-5;

std::vector<FrameRecord> LoadFrameRecords(const std::string& dataset_directory) {
  const std::string associate_path = dataset_directory + "/associate.txt";
  std::ifstream input_stream(associate_path.c_str());
  if (!input_stream.is_open()) {
    throw std::runtime_error("Failed to open associate file: " + associate_path);
  }

  std::vector<FrameRecord> frame_records;
  std::string line;
  while (std::getline(input_stream, line)) {
    if (line.empty()) {
      continue;
    }

    std::istringstream line_stream(line);
    FrameRecord frame_record;
    std::string image_name;
    line_stream >> frame_record.timestamp >> frame_record.odometry_x >> frame_record.odometry_y >>
        frame_record.odometry_yaw >> image_name;
    if (line_stream.fail() || image_name.empty()) {
      std::cerr << "Skip malformed associate line: " << line << std::endl;
      continue;
    }

    frame_record.contour_image_path = dataset_directory + "/contours/" + image_name;
    frame_records.push_back(frame_record);
  }
  return frame_records;
}

std::vector<cv::Point2d> ContourImageToPoints(const cv::Mat& contour_image) {
  cv::Mat gray_image;
  if (contour_image.channels() == 3) {
    cv::cvtColor(contour_image, gray_image, cv::COLOR_BGR2GRAY);
  } else {
    gray_image = contour_image;
  }

  std::vector<cv::Point2d> points;
  const int image_width = gray_image.cols;
  const int image_height = gray_image.rows;

  for (int row = 0; row < gray_image.rows; ++row) {
    for (int column = 0; column < gray_image.cols; ++column) {
      if (gray_image.at<unsigned char>(row, column) <= 10) {
        continue;
      }
      const double x = (image_height / 2.0 - row) * kPixelToMeter + kRearAxleToVehicleCenterMeters;
      const double y = (image_width / 2.0 - column) * kPixelToMeter;
      points.push_back(cv::Point2d(x, y));
    }
  }

  if (static_cast<int>(points.size()) <= kMaximumPointCount) {
    return points;
  }

  std::vector<cv::Point2d> sampled_points;
  sampled_points.reserve(kMaximumPointCount);
  const double step = static_cast<double>(points.size()) / static_cast<double>(kMaximumPointCount);
  for (int index = 0; index < kMaximumPointCount; ++index) {
    sampled_points.push_back(points[static_cast<size_t>(index * step)]);
  }
  return sampled_points;
}

cv::Point2d TransformPoint(const cv::Point2d& point, const RigidTransform2D& transform) {
  const double cos_theta = std::cos(transform.rotation);
  const double sin_theta = std::sin(transform.rotation);
  return cv::Point2d(cos_theta * point.x - sin_theta * point.y + transform.translation.x,
                     sin_theta * point.x + cos_theta * point.y + transform.translation.y);
}

RigidTransform2D ComposeTransform(const RigidTransform2D& increment, const RigidTransform2D& transform) {
  RigidTransform2D composed;
  composed.rotation = increment.rotation + transform.rotation;
  composed.translation = TransformPoint(transform.translation, increment);
  return composed;
}

int FindNearestPointIndex(const cv::Point2d& point, const std::vector<cv::Point2d>& target_points) {
  int nearest_index = -1;
  double nearest_distance = std::numeric_limits<double>::infinity();
  for (int index = 0; index < static_cast<int>(target_points.size()); ++index) {
    const cv::Point2d difference = point - target_points[index];
    const double distance = difference.dot(difference);
    if (distance < nearest_distance) {
      nearest_distance = distance;
      nearest_index = index;
    }
  }
  return nearest_index;
}

RigidTransform2D EstimateRigidTransform(const std::vector<cv::Point2d>& source_points,
                                         const std::vector<cv::Point2d>& target_points) {
  if (source_points.size() != target_points.size() || source_points.empty()) {
    throw std::runtime_error("Point correspondence lists must have the same non-zero size.");
  }

  cv::Point2d source_centroid(0.0, 0.0);
  cv::Point2d target_centroid(0.0, 0.0);
  for (size_t index = 0; index < source_points.size(); ++index) {
    source_centroid += source_points[index];
    target_centroid += target_points[index];
  }
  source_centroid *= 1.0 / static_cast<double>(source_points.size());
  target_centroid *= 1.0 / static_cast<double>(target_points.size());

  double numerator = 0.0;
  double denominator = 0.0;
  for (size_t index = 0; index < source_points.size(); ++index) {
    const cv::Point2d centered_source = source_points[index] - source_centroid;
    const cv::Point2d centered_target = target_points[index] - target_centroid;
    numerator += centered_source.x * centered_target.y - centered_source.y * centered_target.x;
    denominator += centered_source.x * centered_target.x + centered_source.y * centered_target.y;
  }

  RigidTransform2D transform;
  transform.rotation = std::atan2(numerator, denominator);
  RigidTransform2D rotation_only_transform;
  rotation_only_transform.rotation = transform.rotation;
  rotation_only_transform.translation = cv::Point2d(0.0, 0.0);
  transform.translation = target_centroid - TransformPoint(source_centroid, rotation_only_transform);
  return transform;
}

IcpResult AlignContours(const std::vector<cv::Point2d>& source_points,
                        const std::vector<cv::Point2d>& target_points) {
  if (source_points.empty() || target_points.empty()) {
    throw std::runtime_error("Cannot align empty point sets.");
  }

  IcpResult result;
  double previous_error = std::numeric_limits<double>::infinity();
  RigidTransform2D total_transform;

  for (int iteration = 0; iteration < kMaximumIterations; ++iteration) {
    std::vector<cv::Point2d> transformed_source_points;
    std::vector<cv::Point2d> matched_target_points;
    transformed_source_points.reserve(source_points.size());
    matched_target_points.reserve(source_points.size());

    double mean_error = 0.0;
    for (const cv::Point2d& source_point : source_points) {
      const cv::Point2d transformed_source_point = TransformPoint(source_point, total_transform);
      const int nearest_index = FindNearestPointIndex(transformed_source_point, target_points);
      if (nearest_index < 0) {
        continue;
      }
      transformed_source_points.push_back(transformed_source_point);
      matched_target_points.push_back(target_points[nearest_index]);
      const cv::Point2d difference = transformed_source_point - target_points[nearest_index];
      mean_error += std::sqrt(difference.dot(difference));
    }

    if (transformed_source_points.empty()) {
      break;
    }
    mean_error /= static_cast<double>(transformed_source_points.size());

    const RigidTransform2D incremental_transform = EstimateRigidTransform(transformed_source_points, matched_target_points);
    total_transform = ComposeTransform(incremental_transform, total_transform);

    result.iterations = iteration + 1;
    result.mean_error = mean_error;
    result.transform = total_transform;

    if (std::abs(previous_error - mean_error) < kConvergenceTolerance) {
      result.converged = true;
      break;
    }
    previous_error = mean_error;
  }

  return result;
}

void RunPairwiseContourIcp(const std::vector<FrameRecord>& frame_records, int max_pairs) {
  if (frame_records.size() < 2) {
    throw std::runtime_error("Need at least two contour frames to run pairwise ICP.");
  }

  const int available_pairs = static_cast<int>(frame_records.size()) - 1;
  const int pair_count = max_pairs > 0 ? std::min(max_pairs, available_pairs) : available_pairs;

  for (int pair_index = 0; pair_index < pair_count; ++pair_index) {
    const FrameRecord& reference_record = frame_records[pair_index];
    const FrameRecord& current_record = frame_records[pair_index + 1];

    const cv::Mat reference_image = cv::imread(reference_record.contour_image_path, cv::IMREAD_GRAYSCALE);
    const cv::Mat current_image = cv::imread(current_record.contour_image_path, cv::IMREAD_GRAYSCALE);
    if (reference_image.empty() || current_image.empty()) {
      std::cerr << "Skip pair " << pair_index << " because one contour image cannot be loaded." << std::endl;
      continue;
    }

    const std::vector<cv::Point2d> reference_points = ContourImageToPoints(reference_image);
    const std::vector<cv::Point2d> current_points = ContourImageToPoints(current_image);
    const IcpResult result = AlignContours(reference_points, current_points);

    std::cout << "pair " << pair_index << " " << reference_record.timestamp << " -> "
              << current_record.timestamp << std::endl;
    std::cout << "  reference points: " << reference_points.size()
              << ", current points: " << current_points.size() << std::endl;
    std::cout << "  converged: " << result.converged << ", iterations: " << result.iterations
              << ", mean error: " << result.mean_error << std::endl;
    std::cout << "  rotation(rad): " << result.transform.rotation << ", translation(m): ["
              << result.transform.translation.x << ", " << result.transform.translation.y << "]" << std::endl;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "Usage: " << argv[0] << " <dataset_directory> [max_pairs]" << std::endl;
    std::cerr << "Example: " << argv[0] << " examples/contour_sequence 5" << std::endl;
    return 1;
  }

  const std::string dataset_directory = argv[1];
  const int max_pairs = argc == 3 ? std::atoi(argv[2]) : 0;

  try {
    const std::vector<FrameRecord> frame_records = LoadFrameRecords(dataset_directory);
    RunPairwiseContourIcp(frame_records, max_pairs);
  } catch (const std::exception& exception) {
    std::cerr << "Error: " << exception.what() << std::endl;
    return 1;
  }

  return 0;
}
