# Message containing an image masked by instance ids and associated feature vectors

std_msgs/Header header                              # Header of image that the features were sourced from
int32[] mask_ids                                    # Mask IDs if non-sequential (should be empty otherwise)
int32[] panoptic_ids                                # Panoptic IDs
int32 unknown_id                                    # ID used for unknown segments
sensor_msgs/Image image                             # Masked image by segment/feature ID
semantic_inference_msgs/FeatureVector[] features    # Feature vectors for each segment. Size must be equal to the number of masks
