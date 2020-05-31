#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>



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



std::vector<cv::DMatch> GMS_match_filtering(const std::vector<cv::KeyPoint>& kps1, const cv::Size2i& size1, const std::vector<cv::KeyPoint>& kps2, const cv::Size2i& size2, const std::vector<cv::DMatch>& matches)
{
    // TODO: add support of rotation and scale

    // both images are divided into G cells
    const int g_h = 20;
    const int g_v = 20;
    const int G = g_h * g_v;

    const double alpha = 6;
    std::vector<bool> matches_mask(matches.size(), false);
    for (const auto& shift : std::vector<std::vector<double>>({{0.0, 0.0}, {0.5, 0.0}, {0.0, 0.5}, {0.5, 0.5}}))
    {

        // compute the number of matches between each pair of cells
        std::vector<std::vector<std::vector<int>>> nb_matches(G, std::vector<std::vector<int>>(G));
        std::vector<int> nb_features_per_cell(G, 0);
        std::vector<std::vector<int>> matches_indices_per_cell(G, std::vector<int>());
        for (int mi = 0; mi < matches.size(); ++mi)
        {
            const auto& m = matches[mi];
            const cv::KeyPoint& kp1 = kps1[m.queryIdx];
            const cv::KeyPoint& kp2 = kps2[m.trainIdx];

            int xi1 = static_cast<int>(g_h * kp1.pt.x / size1.width + shift[0]);
            int yi1 = static_cast<int>(g_v * kp1.pt.y / size1.height + shift[0]);

            int xi2 = static_cast<int>(g_h * kp2.pt.x / size2.width + shift[0]);
            int yi2 = static_cast<int>(g_v * kp2.pt.y / size2.height + shift[1]);

            if (xi1 >= g_h || xi2 >= g_h || yi1 >= g_v || yi2 >= g_v)
                continue;

            nb_matches[yi1 * g_h + xi1][xi2 + g_h * yi2].push_back(mi);
            nb_features_per_cell[yi1 * g_h + xi1]++;
        }

        for (int y = 0; y < g_v; ++y)
        {
            for (int x = 0; x < g_h; ++x)
            {
                int score = 0;
                int i = y*g_h + x;

                int max_count = 0;
                int best_j = 0;
                for (int j = 0; j < G; ++j)
                {
                    if (nb_matches[i][j].size() > max_count)
                    {
                        max_count = nb_matches[i][j].size();
                        best_j = j;
                    }
                }
                if (nb_matches[i][best_j].size() == 0) {
                    continue;
                }

                int best_x = best_j % g_h;
                int best_y = best_j / g_h;
                int nb_features_in_neighourhood = 0;
                int count_valid_neighbours = 0;
                for (int dx : {-1, 0, 1})
                {
                    for (int dy : {-1, 0, 1})
                    {
                        int xx = x + dx;
                        int yy = y + dy;
                        int best_xx = best_x + dx;
                        int best_yy = best_y + dy;
                        if (xx >= 0 && xx < g_h && yy >=0 && yy < g_v &&
                            best_xx >= 0 && best_xx < g_h && best_yy >=0 && best_yy < g_v)
                        {
                            int ii = xx + yy * g_h;
                            int jj = best_xx + best_yy * g_h;
                            score += nb_matches[ii][jj].size();
                            nb_features_in_neighourhood += nb_features_per_cell[ii];
                            count_valid_neighbours++;
                        }
                    }
                }

                double thresh = alpha * std::sqrt(static_cast<double>(nb_features_in_neighourhood) / count_valid_neighbours);

                if (score > thresh)
                {
                    for (auto idx : nb_matches[i][best_j])
                        matches_mask[idx] = true;
                }
            }
        }
    }


    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < matches.size(); ++i)
    {
        if (matches_mask[i])
            good_matches.push_back(matches[i]);
    }

    std::cout << good_matches.size() << " good matches.\n";
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


int main(int argc, char* argv[])
{

    std::string img_1_file = "../data/01.jpg";
    std::string img_2_file = "../data/02.jpg";

    if (argc == 3)
    {
        img_1_file = std::string(argv[1]);
        img_2_file = std::string(argv[2]);
    }
    cv::Mat img1, img2;
    img1 = cv::imread(img_1_file, cv::IMREAD_UNCHANGED);
    img2 = cv::imread(img_2_file, cv::IMREAD_UNCHANGED);
    
    cv::resize(img1, img1, cv::Size(640, 480), 0, 0, cv::INTER_AREA);
    cv::resize(img2, img2, cv::Size(640, 480), 0, 0, cv::INTER_AREA);

    cv::Ptr<cv::ORB> detector = cv::ORB::create(10000);
    detector->setFastThreshold(0);
    std::vector<cv::KeyPoint> kps1, kps2;
    cv::Mat desc1, desc2;
    
    detector->detectAndCompute(img1, cv::Mat(), kps1, desc1);
    detector->detectAndCompute(img2, cv::Mat(), kps2, desc2);

    // Apparently FLANN needs the descriptor to have float values (not binary) => not working with ORB
    // cv::Ptr<cv::DescriptorMatcher> matcher_flann = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    // std::vector<std::vector<cv::DMatch>> knn_matches_flann;
    // matcher_flann->knnMatch(desc1, desc2, knn_matches_flann, 2);


    // ratio test filtering
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(desc1, desc2, matches);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(desc1, desc2, knn_matches, 2);
    

    std::vector<cv::DMatch> good_matches;

    // sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b){
    //     return a.distance < b.distance;
    // });
    // good_matches = std::vector<cv::DMatch>(matches.begin(), matches.begin()+1000);

    // good_matches = matches_filtering_ratio_test(knn_matches);
    good_matches = GMS_match_filtering(kps1, img1.size(), kps2, img2.size(), matches);



    

    cv::Mat img_matches;
    cv::drawMatches(img1, kps1, img2, kps2, good_matches, img_matches, cv::Scalar(0, 255, 255), cv::Scalar(200, 200, 200), {}, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    img_matches = DrawInlier(img1, img2, kps1, kps2, good_matches);

    cv::namedWindow("fen", cv::WINDOW_AUTOSIZE);
    cv::imshow("fen", img_matches);
    cv::waitKey(-1);

    // cv::imwrite("../best_1k_nearest.png", img_matches);
    // cv::imwrite("../ratio_test.png", img_matches);
    cv::imwrite("../GMS_results.png", img_matches);

	std::cout << "Grid-based Motion Statistics Matching\n";
	return 0;
}


// Salut, comment t'as deviné ? ;)
// Comment ça va?
