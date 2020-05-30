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

    // both images are divided into G cells

    const int g_h = 15;
    const int g_v = 15;
    const int G = g_h * g_v;
    // div by w (or h) and mult by g_h

    const double alpha = 6;

    // compute the number of matches between each pair of cells
    std::vector<std::vector<std::vector<cv::DMatch>>> nb_matches(G, std::vector<std::vector<cv::DMatch>>(G));
    std::vector<int> nb_features_per_cell(G, 0);
    for (const auto& m : matches)
    {
        const cv::KeyPoint& kp1 = kps1[m.queryIdx];
        const cv::KeyPoint& kp2 = kps2[m.trainIdx];

        int xi1 = static_cast<int>(g_h * kp1.pt.x / size1.width);
        int yi1 = static_cast<int>(g_v * kp1.pt.y / size1.height);

        int xi2 = static_cast<int>(g_h * kp2.pt.x / size2.width);
        int yi2 = static_cast<int>(g_v * kp2.pt.y / size2.height);

        nb_matches[yi1 * g_h + xi1][yi2 + g_h * yi2].push_back(m);
        nb_features_per_cell[yi1 * g_h + xi1]++;
    }

    std::vector<cv::DMatch> good_matches;
    for (int y = 0; y < g_v; ++y)
    {
        for (int x = 0; x < g_h; ++x)
        {
            int score = 0;
            int i = y*g_h + x;
            int best_j = std::distance(nb_matches[i].begin(), std::max_element(nb_matches[i].begin(), nb_matches[i].end()));

            int best_x = best_j % g_h;
            int best_y = best_j / g_h;
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
                    }
                }
            }

            double thresh = alpha * std::sqrt(nb_features_per_cell[i]);

            if (score > thresh)
            {
                good_matches.insert(good_matches.end(), nb_matches[i][best_j].begin(), nb_matches[i][best_j].end());
            }
        }
    }
    return good_matches;
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


    cv::Ptr<cv::ORB> detector = cv::ORB::create(100000);
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
    //cv::BFMatcher matcher(cv::NORM_HAMMING);
    //std::vector<cv::DMatch> matches;
    //matcher.match(desc1, desc2, matches);
    //std::vector<std::vector<cv::DMatch>> knn_matches;
    //matcher.knnMatch(desc1, desc2, knn_matches, 2);
    
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(desc1, desc2, matches);
    

    std::vector<cv::DMatch> good_matches;
    // good_matches = matches_filtering_ratio_test(knn_matches_flann);
    good_matches = GMS_match_filtering(kps1, img1.size(), kps2, img2.size(), matches);

    cv::Mat img_matches;
    cv::drawMatches(img1, kps1, img2, kps2, good_matches, img_matches, cv::Scalar(0, 255, 255), cv::Scalar(200, 200, 200), {}, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


    cv::namedWindow("fen", cv::WINDOW_AUTOSIZE);
    cv::imshow("fen", img_matches);
    cv::waitKey(-1);


	std::cout << "Grid-based Motion Statistics Matching\n";
	return 0;
}


// Salut, comment t'as deviné ? ;)
// Comment ça va?
