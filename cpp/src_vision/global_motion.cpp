#include "iha/global_motion.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace iha {

std::optional<GlobalMotion> GlobalMotionEstimator::update(const cv::Mat& gray_full, const std::vector<Box>& exclude) {
    CV_Assert(gray_full.type() == CV_8UC1);
    cv::Mat gray;
    if (p_.downscale != 1.f)
        cv::resize(gray_full, gray, cv::Size(), p_.downscale, p_.downscale, cv::INTER_AREA);
    else
        gray = gray_full;

    if (prev_.empty() || prev_.size() != gray.size()) {
        prev_ = gray.clone();
        return std::nullopt;
    }

    cv::Mat mask(prev_.size(), CV_8UC1, cv::Scalar(255));
    for (const auto& b : exclude) {
        const float m = 8.f;  // kutunun etrafinda pay
        cv::Rect r(cv::Point(static_cast<int>((b.x - m) * p_.downscale), static_cast<int>((b.y - m) * p_.downscale)),
                   cv::Point(static_cast<int>((b.x + b.w + m) * p_.downscale),
                             static_cast<int>((b.y + b.h + m) * p_.downscale)));
        r &= cv::Rect(0, 0, mask.cols, mask.rows);
        if (r.area() > 0) mask(r).setTo(0);
    }

    std::vector<cv::Point2f> p0, p1;
    cv::goodFeaturesToTrack(prev_, p0, p_.max_corners, p_.quality, p_.min_distance * p_.downscale, mask);
    std::optional<GlobalMotion> out;
    if (static_cast<int>(p0.size()) >= p_.min_inliers) {
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(prev_, gray, p0, p1, status, err, cv::Size(p_.lk_window, p_.lk_window), p_.lk_levels);
        std::vector<cv::Point2f> a, b;
        for (size_t i = 0; i < p0.size(); ++i)
            if (status[i]) {
                a.push_back(p0[i]);
                b.push_back(p1[i]);
            }
        if (static_cast<int>(a.size()) >= p_.min_inliers) {
            std::vector<uchar> inl;
            const cv::Mat M = cv::estimateAffinePartial2D(a, b, inl, cv::RANSAC, p_.ransac_px * p_.downscale);
            const int n_in = cv::countNonZero(inl);
            if (!M.empty() && n_in >= p_.min_inliers) {
                GlobalMotion g;
                // Kucuk olcekte kestirilen donusumu tam cozunurluge tasi: L ayni, t / downscale.
                g.A = {static_cast<float>(M.at<double>(0, 0)), static_cast<float>(M.at<double>(0, 1)),
                       static_cast<float>(M.at<double>(0, 2) / p_.downscale), static_cast<float>(M.at<double>(1, 0)),
                       static_cast<float>(M.at<double>(1, 1)), static_cast<float>(M.at<double>(1, 2) / p_.downscale)};
                g.inliers = n_in;
                g.tracked = static_cast<int>(a.size());
                out = g;
            }
        }
    }
    prev_ = gray.clone();
    return out;
}

}  // namespace iha
