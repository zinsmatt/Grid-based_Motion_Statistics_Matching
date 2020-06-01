#include "GMS_filtering.h"



std::vector<cv::DMatch> GMS_match_filtering(const std::vector<cv::KeyPoint>& kps1, const cv::Size2i& size1, 
                                            const std::vector<cv::KeyPoint>& kps2, const cv::Size2i& size2, const std::vector<cv::DMatch>& matches)
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
        for (unsigned int mi = 0; mi < matches.size(); ++mi)
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

                unsigned int max_count = 0;
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
    for (unsigned int i = 0; i < matches.size(); ++i)
    {
        if (matches_mask[i])
            good_matches.push_back(matches[i]);
    }

    std::cout << good_matches.size() << " good matches.\n";
    return good_matches;
}
