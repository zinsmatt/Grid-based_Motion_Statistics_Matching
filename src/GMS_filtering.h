#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

std::vector<cv::DMatch> GMS_match_filtering(const std::vector<cv::KeyPoint>& kps1, const cv::Size2i& size1,
                                            const std::vector<cv::KeyPoint>& kps2, const cv::Size2i& size2, const std::vector<cv::DMatch>& matches);