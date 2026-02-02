// Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
// Technology All rights reserved.

// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree
#include <config_utilities/config_utilities.h>
#include <config_utilities/parsing/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <semantic_inference/image_recolor.h>
#include <semantic_inference/panoptic_extractor.h>
#include <semantic_inference_msgs/FeatureImage.h>
#include <semantic_inference_msgs/FeatureVectorStamped.h>

#include <memory>
#include <opencv2/opencv.hpp>

namespace semantic_inference {

class RepublishNodelet : public nodelet::Nodelet {
 public:
  struct Config {
    ImageRecolor::Config recolor;
    bool open_vocab = false;
  };
  virtual void onInit() override;

  virtual ~RepublishNodelet();

 private:
  void runGtRepublishNodelet(const sensor_msgs::ImageConstPtr& msg);

  Config config_;
  std::unique_ptr<image_transport::ImageTransport> transport_;
  image_transport::Subscriber sub_;
  image_transport::Publisher panoptic_pub_;
  struct Publishers {
    ros::Publisher feature_image;
    ros::Publisher feature_vector;
  } pubs_;

  std::unique_ptr<PanopticExtractor> panoptic_extractor_;
  std::unique_ptr<ImageRecolor> image_recolor_;
  cv_bridge::CvImagePtr panoptic_image_;
  cv_bridge::CvImagePtr label_image_;
};

void declare_config(RepublishNodelet::Config& config) {
  using namespace config;
  name("RepublishNodelet::Config");
  field(config.recolor, "recolor");
  field(config.open_vocab, "open_vocab");
}

void RepublishNodelet::onInit() {
  ros::NodeHandle nh = getPrivateNodeHandle();
  config_ = config::fromRos<RepublishNodelet::Config>(nh);
  config::checkValid(config_);
  image_recolor_ = std::make_unique<ImageRecolor>(config_.recolor);
  panoptic_extractor_ = std::make_unique<PanopticExtractor>();
  transport_ = std::make_unique<image_transport::ImageTransport>(nh);
  sub_ = transport_->subscribe(
      "semantic/image_raw", 1, &RepublishNodelet::runGtRepublishNodelet, this);
  ROS_INFO_STREAM("Subscribed to:" << sub_.getTopic());
  if (!config_.open_vocab) {
    pubs_.feature_image = nh.advertise<semantic_inference_msgs::FeatureImage>(
        "semantic_color/feature_image", 1);
    pubs_.feature_vector =
        nh.advertise<semantic_inference_msgs::FeatureVectorStamped>("image_feature", 1);
  }
  panoptic_pub_ = transport_->advertise("panoptic/image_raw", 1);
}

RepublishNodelet::~RepublishNodelet() {}

void RepublishNodelet::runGtRepublishNodelet(const sensor_msgs::ImageConstPtr& msg) {
  // Extract panoptic IDs and publish
  if (!panoptic_image_) {
    panoptic_image_.reset(new cv_bridge::CvImage());
    panoptic_image_->encoding = "16SC1";
    panoptic_image_->image = cv::Mat(msg->height, msg->width, CV_16SC1);
  }
  if (!label_image_) {
    label_image_.reset(new cv_bridge::CvImage());
    label_image_->encoding = "16SC1";
    label_image_->image = cv::Mat(msg->height, msg->width, CV_16SC1);
  }
  panoptic_image_->header = msg->header;
  image_recolor_->labelImage(cv_bridge::toCvShare(msg)->image, label_image_->image);
  cv::Mat label_image_reshaped;
  label_image_->image.convertTo(label_image_reshaped, CV_32S);
  panoptic_extractor_->extract(label_image_reshaped, panoptic_image_->image);
  panoptic_pub_.publish(panoptic_image_->toImageMsg());

  // Publish feature image and feature vector
  if (!config_.open_vocab) {
    semantic_inference_msgs::FeatureImage feature_image;
    semantic_inference_msgs::FeatureVectorStamped feature_vector;
    feature_image.header = msg->header;
    feature_image.image = *msg;
    feature_vector.header = msg->header;
    pubs_.feature_image.publish(feature_image);
    pubs_.feature_vector.publish(feature_vector);
  }
}

}  // namespace semantic_inference

PLUGINLIB_EXPORT_CLASS(semantic_inference::RepublishNodelet, nodelet::Nodelet)
