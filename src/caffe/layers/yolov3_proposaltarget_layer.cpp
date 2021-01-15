#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "caffe/layers/detection_evaluate_layer.hpp"
#include "caffe/layers/yolov3_proposaltarget_layer.hpp"
#include "caffe/util/bbox_util.hpp"

namespace caffe {
	
template <typename Dtype>
void Yolov3ProposalTargetLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const DetectionEvaluateParameter& detection_evaluate_param =
      this->layer_param_.detection_evaluate_param();
  CHECK(detection_evaluate_param.has_num_classes())
      << "Must provide num_classes.";
  num_classes_ = detection_evaluate_param.num_classes();
  num_attr_classes_ = detection_evaluate_param.num_attr_classes();
  background_label_id_ = detection_evaluate_param.background_label_id();
  overlap_threshold_ = detection_evaluate_param.overlap_threshold();
  CHECK_GT(overlap_threshold_, 0.) << "overlap_threshold must be non negative.";
  //evaluate_difficult_gt_ = detection_evaluate_param.evaluate_difficult_gt();
  
  //count_ = 0;
  // If there is no name_size_file provided, use normalized bbox to evaluate.
  use_normalized_bbox_ = false;

  // Retrieve resize parameter if there is any provided.
  has_resize_ = false;
  if (has_resize_) {
    resize_param_ = detection_evaluate_param.resize_param();
  }
}

template <typename Dtype>
void Yolov3ProposalTargetLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), 1);
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[0]->width(), 7);
  CHECK_EQ(bottom[1]->num(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->width(), 8);

  // num() and channels() are 1.
  int num_pos_classes = background_label_id_ == -1 ?
      num_classes_ : num_classes_ - 1;
  int num_valid_det = 0;
  const Dtype* det_data = bottom[0]->cpu_data();
  for (int i = 0; i < bottom[0]->height(); ++i) {
    if (det_data[1] != -1) {
      ++num_valid_det;
    }
    det_data += 7;
  }
  
  //vector<int> top_shape(2, 1);
  //top_shape.push_back(num_pos_classes + num_valid_det);
  // Each row is a 5 dimension vector, which stores
  // [image_id, label, confidence, true_pos, false_pos]
  //top_shape.push_back(5);
  //top[0]->Reshape(top_shape);
  
  
    // [label, x, y, w, h]
  vector<int> top_shape(4);
	top_shape[0] = num_valid_det;
	top_shape[1] = 5;
	top_shape[2] = 1;
	top_shape[3] = 1;
  top[0]->Reshape(top_shape);

  if (num_attr_classes_ > 0)
  {
	// attribute labels
	  vector<int> top_attr_shape(2);
	  top_attr_shape[0] = num_valid_det;
	  top_attr_shape[1] = 1;
	  top[1]->Reshape(top_attr_shape);
  }
}

template <typename Dtype>
void Yolov3ProposalTargetLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* det_data = bottom[0]->cpu_data();
  const Dtype* gt_data = bottom[1]->cpu_data();
  const Dtype *p_img_info_cpu = bottom[2]->cpu_data();
  const Dtype img_height = p_img_info_cpu[0];
  const Dtype img_width = p_img_info_cpu[1];
  sizes_.clear();
  sizes_.push_back(std::make_pair(img_height, img_width));
  

  // Retrieve all detection results.
  map<int, LabelBBox> all_detections;
  GetDetectionResults(det_data, bottom[0]->height(), background_label_id_,
                      &all_detections);

  // Retrieve all ground truth (including difficult ones).
  map<int, LabelBBox> all_gt_bboxes;
  map<int, LabelAttr> all_gt_attrs;
  GetGroundTruth(gt_data, bottom[1]->height(), background_label_id_,
                 true, &all_gt_bboxes, &all_gt_attrs);
  
  int num_det = 0;
 // Dtype* top_data = top[0]->mutable_cpu_data();
 // caffe_set(top[0]->count(), Dtype(0.), top_data);
 // 
 // 
 // Dtype* attr_labels = NULL;
 // if (num_attr_classes_ > 0) {
	//	attr_labels = top[1]->mutable_cpu_data();
	//	caffe_set(top[1]->count(), (Dtype)-1, attr_labels);
	//}

  vector< NormalizedBBox > predicts;

  // Insert detection evaluate status.
  for (map<int, LabelBBox>::iterator it = all_detections.begin();
       it != all_detections.end(); ++it) {
    int image_id = it->first;
    LabelBBox& detections = it->second;
    if (all_gt_bboxes.find(image_id) == all_gt_bboxes.end()) {
		std::cout <<"xxxxxxxxxxxxxxxxxxxxxxxxx" <<std::endl;
   //   // No ground truth for current image. All detections become false_pos.
   //   for (LabelBBox::iterator iit = detections.begin();
   //        iit != detections.end(); ++iit) {
   //     int label = iit->first;
   //     if (label == -1) {
   //       continue;
   //     }
   //     const vector<NormalizedBBox>& bboxes = iit->second;
   //     for (int i = 0; i < bboxes.size(); ++i) {
   //       top_data[num_det * 5] = label;
		 // top_data[num_det * 5 + 1] = bboxes[i].xmin();
		 // top_data[num_det * 5 + 2] = bboxes[i].ymin();
		 // top_data[num_det * 5 + 3] = bboxes[i].xmax();
		 // top_data[num_det * 5 + 4] = bboxes[i].ymax();
		 // if (num_attr_classes_ > 0) {
			//attr_labels[num_det] = -1;
		 // }
   //       ++num_det;
   //     }
   //   }
    } else {
      LabelBBox& label_bboxes = all_gt_bboxes.find(image_id)->second;
	  LabelAttr& label_attrs = all_gt_attrs.find(image_id)->second;
      for (LabelBBox::iterator iit = detections.begin();
           iit != detections.end(); ++iit) {
        int label = iit->first;
        if (label == -1) {
          continue;
        }
        vector<NormalizedBBox>& bboxes = iit->second;
        if (label_bboxes.find(label) == label_bboxes.end()) {
			std::cout << "there is not in gt: " << label << std::endl;

   //       // No ground truth for current label. All detections become false_pos.
   //       for (int i = 0; i < bboxes.size(); ++i) {
   //         top_data[num_det * 5] = label;
			//top_data[num_det * 5 + 1] = bboxes[i].xmin();
			//top_data[num_det * 5 + 2] = bboxes[i].ymin();
			//top_data[num_det * 5 + 3] = bboxes[i].xmax();
			//top_data[num_det * 5 + 4] = bboxes[i].ymax();
			//if (num_attr_classes_ > 0) {
			//  attr_labels[num_det] = -1;
			//}
			//++num_det;
   //       }
        } else {
		  std::cout << "ggggggggggggggg" << std::endl;

          vector<NormalizedBBox>& gt_bboxes = label_bboxes.find(label)->second;
		  vector<int>& gt_attrs = label_attrs.find(label)->second;
          // Scale ground truth if needed.
          if (!use_normalized_bbox_) {
            for (int i = 0; i < gt_bboxes.size(); ++i) {
              OutputBBox(gt_bboxes[i], sizes_[0], has_resize_,
                         resize_param_, &(gt_bboxes[i]));
            }
          }

          // Sort detections in descend order based on scores.
          //std::sort(bboxes.begin(), bboxes.end(), SortBBoxDescend);
          for (int i = 0; i < bboxes.size(); ++i) {
            if (!use_normalized_bbox_) {
              OutputBBox(bboxes[i], sizes_[0], has_resize_,
                         resize_param_, &(bboxes[i]));
            }

			if (bboxes[i].xmin() < 0 || bboxes[i].ymin() < 0)
			{
				continue;
			}
			if (bboxes[i].xmax()>= img_width || bboxes[i].ymax() >= img_height)
			{
				continue;
			}
            // Compare with each ground truth bbox.
            float overlap_max = -1;
            int jmax = -1;
            for (int j = 0; j < gt_bboxes.size(); ++j) {
              float overlap = JaccardOverlap(bboxes[i], gt_bboxes[j],
                                             use_normalized_bbox_);
              if (overlap > overlap_max) {
                overlap_max = overlap;
                jmax = j;
              }
            }
			
			std::cout <<"xxxxxxx" << overlap_max << bboxes[i].xmin() << bboxes[i].ymin() <<" " << bboxes[i].xmax() <<" " << bboxes[i].ymax()<<std::endl;
			
            if (overlap_max >= overlap_threshold_) {
              {
					NormalizedBBox predict;
     //             // true positive.
     //             top_data[num_det * 5] = label;
				 // top_data[num_det * 5 + 1] = bboxes[i].xmin();
				 // top_data[num_det * 5 + 2] = bboxes[i].ymin();
				 // top_data[num_det * 5 + 3] = bboxes[i].xmax();
				 // top_data[num_det * 5 + 4] = bboxes[i].ymax();
				 // if (num_attr_classes_ > 0) {
					//attr_labels[num_det] = gt_attrs[jmax];
				 // }

				  predict.set_xmin(bboxes[i].xmin());
				  predict.set_ymin(bboxes[i].ymin());
				  predict.set_xmax(bboxes[i].xmax());
				  predict.set_ymax(bboxes[i].ymax());
				  predict.set_label(label);
				  predict.set_difficult(gt_attrs[jmax]);
				  predicts.push_back(predict);

				  std::cout <<"xxxxxxx" << label << bboxes[i].xmin() << bboxes[i].ymin() <<" " << bboxes[i].xmax() <<" " << bboxes[i].ymax()<<std::endl;
              }
            } else {
     //         // false positive.
     //         top_data[num_det * 5] = label;
			  //top_data[num_det * 5 + 1] = bboxes[i].xmin();
			  //top_data[num_det * 5 + 2] = bboxes[i].ymin();
			  //top_data[num_det * 5 + 3] = bboxes[i].xmax();
			  //top_data[num_det * 5 + 4] = bboxes[i].ymax();
			  //if (num_attr_classes_ > 0) {
					//attr_labels[num_det] = -1;
				 // }
            }
            ++num_det;
          }
        }
      }
    }
  }

  vector<int> top_shape(4);
  top_shape[0] = predicts.size();
  top_shape[1] = 5;
  top_shape[2] = 1;
  top_shape[3] = 1;
  top[0]->Reshape(top_shape);

  if (num_attr_classes_ > 0)
  {
	  // attribute labels
	  vector<int> top_attr_shape(2);
	  top_attr_shape[0] = predicts.size();
	  top_attr_shape[1] = 1;
	  top[1]->Reshape(top_attr_shape);
  }

  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0.), top_data);

  Dtype* attr_labels = NULL;
  if (num_attr_classes_ > 0) {
	  attr_labels = top[1]->mutable_cpu_data();
	  caffe_set(top[1]->count(), (Dtype)-1, attr_labels);
  }

  for (int num_det = 0; num_det < predicts.size(); num_det++) {
	top_data[num_det * 5] = predicts[num_det].label();
	top_data[num_det * 5 + 1] = predicts[num_det].xmin();
	top_data[num_det * 5 + 2] = predicts[num_det].ymin();
	top_data[num_det * 5 + 3] = predicts[num_det].xmax();
	top_data[num_det * 5 + 4] = predicts[num_det].ymax();
	if (num_attr_classes_ > 0) {
		attr_labels[num_det] = predicts[num_det].difficult();
	}
  }
  
  //std::cout <<"Yolov3ProposalTargetLayer fw" <<std::endl;
  
}


#ifdef CPU_ONLY
//STUB_GPU_FORWARD(DetectionOutputLayer, Forward);
#endif

INSTANTIATE_CLASS(Yolov3ProposalTargetLayer);
REGISTER_LAYER_CLASS(Yolov3ProposalTarget);

}  // namespace caffe
