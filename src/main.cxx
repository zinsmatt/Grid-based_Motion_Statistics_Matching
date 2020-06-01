#include <filesystem>
#include <iostream>


#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "GMS_filtering.h"

namespace fs = std::filesystem;

enum Matches_Filtering_Method
{
    None, Best1K, RatioTest, GMS
};

std::vector<cv::DMatch> matches_filtering_ratio_test(const std::vector<std::vector<cv::DMatch>>& matches)
{
    const double ratio_thresh = 0.7;
    std::vector<cv::DMatch> good_matches;
    for (const auto& m : matches)
    {
        if (m[0].distance < ratio_thresh * m[1].distance)
            good_matches.push_back(m[0]);
    }
    return good_matches;
}



cv::Mat DrawInlier(cv::Mat &src1, cv::Mat &src2, std::vector<cv::KeyPoint> &kpt1, std::vector<cv::KeyPoint> &kpt2, std::vector<cv::DMatch> &inlier) {
	const int height = std::max(src1.rows, src2.rows);
	const int width = src1.cols + src2.cols;
	cv::Mat output(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
	src1.copyTo(output(cv::Rect(0, 0, src1.cols, src1.rows)));
	src2.copyTo(output(cv::Rect(src1.cols, 0, src2.cols, src2.rows)));

    for (size_t i = 0; i < inlier.size(); i++)
    {
        cv::Point2f left = kpt1[inlier[i].queryIdx].pt;
        cv::Point2f right = (kpt2[inlier[i].trainIdx].pt + cv::Point2f((float)src1.cols, 0.f));
        cv::line(output, left, right, cv::Scalar(0, 255, 255));
    }

	return output;
}


std::tuple<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>, std::vector<cv::DMatch>>
match_images(const cv::Mat& img1, const cv::Mat& img2, Matches_Filtering_Method method=Matches_Filtering_Method::None)
{
    cv::Ptr<cv::ORB> detector = cv::ORB::create(10000);
    detector->setFastThreshold(0);
    std::vector<cv::KeyPoint> kps1, kps2;
    cv::Mat desc1, desc2;
    
    detector->detectAndCompute(img1, cv::Mat(), kps1, desc1);
    detector->detectAndCompute(img2, cv::Mat(), kps2, desc2);

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    
    std::vector<cv::DMatch> good_matches;

    if (method == Matches_Filtering_Method::None)
    {
        matcher.match(desc1, desc2, good_matches);
    }
    else if (method == Matches_Filtering_Method::Best1K)
    {
        std::vector<cv::DMatch> matches;
        matcher.match(desc1, desc2, matches);
        std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b){
            return a.distance < b.distance;
        });
        good_matches = std::vector<cv::DMatch>(matches.begin(), matches.begin()+1000);
    }
    else if (method == Matches_Filtering_Method::RatioTest)
    {
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher.knnMatch(desc1, desc2, knn_matches, 2);
        good_matches = matches_filtering_ratio_test(knn_matches);
    }
    else if (method == Matches_Filtering_Method::GMS)
    {
        std::vector<cv::DMatch> matches;
        matcher.match(desc1, desc2, matches);
        good_matches = GMS_match_filtering(kps1, img1.size(), kps2, img2.size(), matches);
    }
    else
    {
        throw std::runtime_error("Unknown filtering method");
    }

    return make_tuple(kps1, kps2, good_matches);
}


int main(int argc, char* argv[])
{
	std::cout << "Grid-based Motion Statistics Matching\n";


    std::string img_1_file = "../data/01.jpg";
    std::string img_2_file = "../data/02.jpg";

    if (argc == 3)
    {
        img_1_file = std::string(argv[1]);
        img_2_file = std::string(argv[2]);

        cv::Mat img1, img2;
        img1 = cv::imread(img_1_file, cv::IMREAD_UNCHANGED);
        img2 = cv::imread(img_2_file, cv::IMREAD_UNCHANGED);
        
        //cv::resize(img1, img1, cv::Size(640, 480), 0, 0, cv::INTER_AREA);
        //cv::resize(img2, img2, cv::Size(640, 480), 0, 0, cv::INTER_AREA);

        auto [kps1, kps2, good_matches] = match_images(img1, img2, Matches_Filtering_Method::GMS);

        cv::Mat img_matches;
        cv::drawMatches(img1, kps1, img2, kps2, good_matches, img_matches, cv::Scalar(0, 255, 255), cv::Scalar(200, 200, 200), {}, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        img_matches = DrawInlier(img1, img2, kps1, kps2, good_matches);

        cv::namedWindow("fen", cv::WINDOW_AUTOSIZE);
        cv::imshow("fen", img_matches);
        cv::waitKey(-1);

        // cv::imwrite("../best_1k_nearest.png", img_matches);
        // cv::imwrite("../ratio_test.png", img_matches);
        // cv::imwrite("../GMS_results.png", img_matches);
    }
    else if (argc == 2)
    {
        std::string folder = std::string(argv[1]);

        std::vector<std::string> images_filenames;
        for (const auto & file : fs::directory_iterator(folder))
        {
            if (file.path().extension() == ".png")
            {
                images_filenames.push_back(file.path().string());
            }
        }
        std::sort(images_filenames.begin(), images_filenames.end());
        std::cout << "Found " << images_filenames.size() << " images.\n";

        cv::Mat ref_img = cv::imread(images_filenames.front(), cv::IMREAD_UNCHANGED);
        cv::Mat img_matches;
        cv::namedWindow("fen", cv::WINDOW_AUTOSIZE);

        // use first images as reference
        for (unsigned int i = 1; i < images_filenames.size(); i+=4)
        {
            cv::Mat img = cv::imread(images_filenames[i], cv::IMREAD_UNCHANGED);

            auto [kps1, kps2, good_matches] = match_images(ref_img, img, Matches_Filtering_Method::RatioTest);
            img_matches = DrawInlier(ref_img, img, kps1, kps2, good_matches);
            cv::imshow("fen", img_matches);
            cv::waitKey(10);
        }

        cv::destroyAllWindows();
    }

	return 0;
}
