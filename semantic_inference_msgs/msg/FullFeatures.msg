# Message containing full features for an image, including segmentation features and full image feature vector

std_msgs/Header header
semantic_inference_msgs/FeatureImage segmentation_features  # Segmentation features for the image
semantic_inference_msgs/FeatureVectorStamped full_image_feature  # Full image feature vector with timestamp
sensor_msgs/Image pixelwise_feature_map  # Pixel-wise feature map for the image
