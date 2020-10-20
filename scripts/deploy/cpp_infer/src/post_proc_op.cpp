
#include "include/post_proc_op.hpp"

std::vector<std::vector<std::vector<int>>>
DBPostProcess::BoxesFromBitmap(const cv::Mat pred, 
                               const cv::Mat bitmap,
                               const float &box_thresh,
                               const float &unclip_ratio){
    
    int width  = bitmap.cols;
    int height = bitmap.rows;

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(bitmap, contours, hierarchy, cv::RETR_LIST,
                    cv::CHAIN_APPROX_SIMPLE);

    int num_contours = contours.size();
    if (num_contours > this->max_candidates_)
        num_contours = this->max_candidates_;
    
    std::vector<std::vector<std::vector<int>>> boxes;
    for (int _i = 0; _i < num_contours; _i++) {
        if (contours[_i].size() <= 2)
            continue;
        float ssid;
        cv::RotatedRect box = cv::minAreaRect(contours[_i]);
        ssid = std::min(box.size.width, box.size.height);
        auto array = GetMiniBoxes(box);

        if (ssid < this->min_size_) 
            continue;

        float score;
        score = BoxScoreFast(array, pred);
        if (score < box_thresh)
            continue;

        // start for unclip
        cv::RotatedRect points = UnClip(array, unclip_ratio);
        ssid = std::min(points.size.width, points.size.height);
        auto cliparray = GetMiniBoxes(points);

        if (ssid < this->min_size_ + 2)
            continue;

        int dest_width  = pred.cols;
        int dest_height = pred.rows;
        std::vector<std::vector<int>> intcliparray;

        for (int num_pt = 0; num_pt < 4; num_pt++) {
            int x = clamp(int(cliparray[num_pt][0]), 0, width);
            int y = clamp(int(cliparray[num_pt][1]), 0, height);
            std::vector<int> a{x,y};
            intcliparray.push_back(a);
        }
        boxes.push_back(intcliparray);
    } // end for

    return boxes;

}

void DBPostProcess::GetContourArea(const std::vector<std::vector<float>> &box,
                                   float unclip_ratio, float &distance) {
  int pts_num = 4;
  float area = 0.0f;
  float dist = 0.0f;
  for (int i = 0; i < pts_num; i++) {
    area += box[i][0] * box[(i + 1) % pts_num][1] -
            box[i][1] * box[(i + 1) % pts_num][0];
    dist += sqrtf((box[i][0] - box[(i + 1) % pts_num][0]) *
                      (box[i][0] - box[(i + 1) % pts_num][0]) +
                  (box[i][1] - box[(i + 1) % pts_num][1]) *
                      (box[i][1] - box[(i + 1) % pts_num][1]));
  }
  area = fabs(float(area / 2.0));

  distance = area * unclip_ratio / dist;
}

cv::RotatedRect DBPostProcess::UnClip(std::vector<std::vector<float>> box,
                                      const float &unclip_ratio) {
    float distance = 1.0;

    GetContourArea(box, unclip_ratio, distance);

    ClipperLib::ClipperOffset offset;
    ClipperLib::Path p;
    p << ClipperLib::IntPoint(int(box[0][0]), int(box[0][1]))
        << ClipperLib::IntPoint(int(box[1][0]), int(box[1][1]))
        << ClipperLib::IntPoint(int(box[2][0]), int(box[2][1]))
        << ClipperLib::IntPoint(int(box[3][0]), int(box[3][1]));
    offset.AddPath(p, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

    ClipperLib::Paths soln;
    offset.Execute(soln, distance);
    std::vector<cv::Point2f> points;

    for (int j = 0; j < soln.size(); j++) {
        for (int i = 0; i < soln[soln.size() - 1].size(); i++) {
        points.emplace_back(soln[j][i].X, soln[j][i].Y);
        }
    }
    cv::RotatedRect res;
    if (points.size() <= 0) {
        res = cv::RotatedRect(cv::Point2f(0, 0), cv::Size2f(1, 1), 0);
    } else {
        res = cv::minAreaRect(points);
    }
    return res;
}

std::vector<std::vector<float>> DBPostProcess::GetMiniBoxes(cv::RotatedRect box) {
  
  cv::Mat points;
  cv::boxPoints(box, points);

  auto array = Mat2Vector(points);
  std::sort(array.begin(), array.end(), Xsort<float>);

  std::vector<float> idx1 = array[0], idx2 = array[1], idx3 = array[2],
                     idx4 = array[3];
  if (array[3][1] <= array[2][1]) {
    idx2 = array[3];
    idx3 = array[2];
  } else {
    idx2 = array[2];
    idx3 = array[3];
  }
  if (array[1][1] <= array[0][1]) {
    idx1 = array[1];
    idx4 = array[0];
  } else {
    idx1 = array[0];
    idx4 = array[1];
  }

  array[0] = idx1;
  array[1] = idx2;
  array[2] = idx3;
  array[3] = idx4;

  return array;
}

float DBPostProcess::BoxScoreFast(std::vector<std::vector<float>> box_array,
                                  cv::Mat pred) {
  auto array = box_array;
  int width = pred.cols;
  int height = pred.rows;

  float box_x[4] = {array[0][0], array[1][0], array[2][0], array[3][0]};
  float box_y[4] = {array[0][1], array[1][1], array[2][1], array[3][1]};

  int xmin = clamp(int(std::floor(*(std::min_element(box_x, box_x + 4)))), 0,
                   width - 1);
  int xmax = clamp(int(std::ceil(*(std::max_element(box_x, box_x + 4)))), 0,
                   width - 1);
  int ymin = clamp(int(std::floor(*(std::min_element(box_y, box_y + 4)))), 0,
                   height - 1);
  int ymax = clamp(int(std::ceil(*(std::max_element(box_y, box_y + 4)))), 0,
                   height - 1);

  cv::Mat mask;
  mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

  cv::Point root_point[4];
  root_point[0] = cv::Point(int(array[0][0]) - xmin, int(array[0][1]) - ymin);
  root_point[1] = cv::Point(int(array[1][0]) - xmin, int(array[1][1]) - ymin);
  root_point[2] = cv::Point(int(array[2][0]) - xmin, int(array[2][1]) - ymin);
  root_point[3] = cv::Point(int(array[3][0]) - xmin, int(array[3][1]) - ymin);
  const cv::Point *ppt[1] = {root_point};
  int npt[] = {4};
  cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

  cv::Mat croppedImg;
  pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
      .copyTo(croppedImg);

  auto score = cv::mean(croppedImg, mask)[0];
  return score;
}


