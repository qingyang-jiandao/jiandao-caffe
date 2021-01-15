#include "caffe/layers/proposal_and_detection_layer.hpp"
#include "caffe/util/nms.hpp"
#include <iostream>

#define ROUND(x) ((int)((x) + (Dtype)0.5))

using std::max;
using std::min;

namespace caffe
{

template <typename Dtype>
static int transform_box(Dtype box[],
                         const Dtype dx, const Dtype dy,
                         const Dtype d_log_w, const Dtype d_log_h,
                         const Dtype img_W, const Dtype img_H,
                         const Dtype min_box_W, const Dtype min_box_H)
{
  // width & height of box
  const Dtype w = box[2] - box[0] + (Dtype)1;
  const Dtype h = box[3] - box[1] + (Dtype)1;
  // center location of box
  const Dtype ctr_x = box[0] + (Dtype)0.5 * w;
  const Dtype ctr_y = box[1] + (Dtype)0.5 * h;

  // new center location according to gradient (dx, dy)
  const Dtype pred_ctr_x = dx * w + ctr_x;
  const Dtype pred_ctr_y = dy * h + ctr_y;
  // new width & height according to gradient d(log w), d(log h)
  const Dtype pred_w = exp(d_log_w) * w;
  const Dtype pred_h = exp(d_log_h) * h;

  // update upper-left corner location
  box[0] = pred_ctr_x - (Dtype)0.5 * pred_w;
  box[1] = pred_ctr_y - (Dtype)0.5 * pred_h;
  // update lower-right corner location
  box[2] = pred_ctr_x + (Dtype)0.5 * pred_w;
  box[3] = pred_ctr_y + (Dtype)0.5 * pred_h;

  // adjust new corner locations to be within the image region,
  box[0] = std::max((Dtype)0, std::min(box[0], img_W - (Dtype)1));
  box[1] = std::max((Dtype)0, std::min(box[1], img_H - (Dtype)1));
  box[2] = std::max((Dtype)0, std::min(box[2], img_W - (Dtype)1));
  box[3] = std::max((Dtype)0, std::min(box[3], img_H - (Dtype)1));

  // recompute new width & height
  const Dtype box_w = box[2] - box[0] + (Dtype)1;
  const Dtype box_h = box[3] - box[1] + (Dtype)1;

  // check if new box's size >= threshold
  return (box_w >= min_box_W) * (box_h >= min_box_H);
}

template <typename Dtype>
static int enumerate_blob_cpu(const Dtype bottom4d[],
	const Dtype d_anchor4d[],
	const Dtype d_targetsocore4d[],
	const Dtype anchors[],
	const int num_anchors,
	Dtype proposals[],
	Dtype filter_scores[],
	Dtype target_filter_scores[],
	const int bottom_H, const int bottom_W,
	const Dtype img_H, const Dtype img_W,
	const Dtype min_box_H, const Dtype min_box_W,
	const int feat_stride,
	const int num_cls)
{
	Dtype *p_proposal = proposals;
	Dtype *p_filter_scores = filter_scores;
	Dtype *p_target_filter_scores = target_filter_scores;

	int valid_proposal_cnt = 0;
	for (int h = 0; h < bottom_H; ++h)
	{
		for (int w = 0; w < bottom_W; ++w)
		{
			const Dtype x = w * feat_stride;
			const Dtype y = h * feat_stride;
			const Dtype *p_score = bottom4d + h * (bottom_W*num_anchors) + w * num_anchors;
			const Dtype *p_box = d_anchor4d + h * (bottom_W*num_anchors*4) + w * num_anchors*4;
			const Dtype *p_target_score = d_targetsocore4d + h * (bottom_W*num_anchors*num_cls) + w * num_anchors*num_cls;
			for (int k = 0; k < num_anchors; ++k)
			{
				if (p_score[k] < 0.05) continue;
				else
					valid_proposal_cnt++;

				const Dtype dx = p_box[(k * 4 + 0)];
				const Dtype dy = p_box[(k * 4 + 1)];
				const Dtype d_log_w = p_box[(k * 4 + 2)];
				const Dtype d_log_h = p_box[(k * 4 + 3)];

				p_proposal[0] = x + anchors[k * 4 + 0];
				p_proposal[1] = y + anchors[k * 4 + 1];
				p_proposal[2] = x + anchors[k * 4 + 2];
				p_proposal[3] = y + anchors[k * 4 + 3];
				p_filter_scores[0] = transform_box(p_proposal, dx, dy, d_log_w, d_log_h,
					img_W, img_H, min_box_W, min_box_H) * p_score[k];
				//p_filter_scores[0] = p_score[k];

				for (int t = 0; t < num_cls; ++t)
				{
					p_target_filter_scores[t] = p_target_score[(k * num_cls + t)];
				}

				p_proposal += 4;
				p_filter_scores += 1;
				p_target_filter_scores += num_cls;
			} // endfor k
		}   // endfor w
	}     // endfor h
	return valid_proposal_cnt;
}

template <typename Dtype>
static void sort_box(Dtype list_cpu[], const int start, const int end,
                     const int num_top)
{
  const Dtype pivot_score = list_cpu[start * 5 + 4];
  int left = start + 1, right = end;
  Dtype temp[5];
  while (left <= right)
  {
    while (left <= end && list_cpu[left * 5 + 4] >= pivot_score)
      ++left;
    while (right > start && list_cpu[right * 5 + 4] <= pivot_score)
      --right;
    if (left <= right)
    {
      for (int i = 0; i < 5; ++i)
      {
        temp[i] = list_cpu[left * 5 + i];
      }
      for (int i = 0; i < 5; ++i)
      {
        list_cpu[left * 5 + i] = list_cpu[right * 5 + i];
      }
      for (int i = 0; i < 5; ++i)
      {
        list_cpu[right * 5 + i] = temp[i];
      }
      ++left;
      --right;
    }
  }

  if (right > start)
  {
    for (int i = 0; i < 5; ++i)
    {
      temp[i] = list_cpu[start * 5 + i];
    }
    for (int i = 0; i < 5; ++i)
    {
      list_cpu[start * 5 + i] = list_cpu[right * 5 + i];
    }
    for (int i = 0; i < 5; ++i)
    {
      list_cpu[right * 5 + i] = temp[i];
    }
  }

  if (start < right - 1)
  {
    sort_box(list_cpu, start, right - 1, num_top);
  }
  if (right + 1 < num_top && right + 1 < end)
  {
    sort_box(list_cpu, right + 1, end, num_top);
  }
}

template <typename Dtype>
static void generate_anchors(int base_size,
                             const Dtype ratios[],
                             const Dtype scales[],
                             const int num_ratios,
                             const int num_scales,
                             Dtype anchors[])
{
  // base box's width & height & center location
  const Dtype base_area = (Dtype)(base_size * base_size);
  const Dtype center = (Dtype)0.5 * (base_size - (Dtype)1);

  // enumerate all transformed boxes
  Dtype *p_anchors = anchors;
  for (int i = 0; i < num_ratios; ++i)
  {
    // transformed width & height for given ratio factors
    const Dtype ratio_w = (Dtype)(sqrt(base_area / ratios[i]));
    const Dtype ratio_h = (Dtype)(ratio_w * ratios[i]);

    for (int j = 0; j < num_scales; ++j)
    {
      // transformed width & height for given scale factors
      const Dtype scale_w = (Dtype)0.5 * ROUND(ratio_w * scales[j] - (Dtype)1);
      const Dtype scale_h = (Dtype)0.5 * ROUND(ratio_h * scales[j] - (Dtype)1);

      // (x1, y1, x2, y2) for transformed box
      p_anchors[0] = center - scale_w;
      p_anchors[1] = center - scale_h;
      p_anchors[2] = center + scale_w;
      p_anchors[3] = center + scale_h;
      p_anchors += 4;
    } // endfor j
  }
}

//template <typename Dtype>
//static vector<Dtype> whctrs(vector<Dtype> anchor)
//{
//	vector<Dtype> result;
//	result.push_back(anchor[2] - anchor[0] + 1); //w
//	result.push_back(anchor[3] - anchor[1] + 1); //h
//	result.push_back((anchor[2] + anchor[0]) / 2); //ctrx
//	result.push_back((anchor[3] + anchor[1]) / 2); //ctry
//	return result;
//}
//
//template <typename Dtype>
//static vector<Dtype> mkanchor(Dtype w, Dtype h, Dtype x_ctr, Dtype y_ctr)
//{
//	vector<Dtype> tmp;
//	tmp.push_back(x_ctr - 0.5*(w - 1));
//	tmp.push_back(y_ctr - 0.5*(h - 1));
//	tmp.push_back(x_ctr + 0.5*(w - 1));
//	tmp.push_back(y_ctr + 0.5*(h - 1));
//	return tmp;
//}

//template <typename Dtype>
//static void generate_anchors(int base_size,
//	const Dtype ratios[],
//	const Dtype scales[],
//	const int num_ratios,
//	const int num_scales,
//	Dtype anchors[])
//{
//	Dtype *p_anchors = anchors;
//
//	//generate base anchor
//	vector<Dtype> base_anchor;
//	base_anchor.push_back((Dtype)0);
//	base_anchor.push_back((Dtype)0);
//	base_anchor.push_back(base_size - 1);
//	base_anchor.push_back(base_size - 1);
//	//enum ratio anchors
//	vector<vector<Dtype>> ratio_anchors;
//	vector<Dtype> base_reform_anchor = whctrs(base_anchor);
//	Dtype x_ctr = base_reform_anchor[2];
//	Dtype y_ctr = base_reform_anchor[3];
//	Dtype size = base_reform_anchor[0] * base_reform_anchor[1];
//	for (int i = 0; i < num_ratios; ++i)
//	{
//		Dtype size_ratios = size / ratios[i];
//		Dtype ws = round(sqrt(size_ratios));
//		Dtype hs = round(ws*ratios[i]);
//		vector<Dtype> tmp = mkanchor(ws, hs, x_ctr, y_ctr);
//		ratio_anchors.push_back(tmp);
//	}
//	//enum scale anchors
//	for (int i = 0; i < ratio_anchors.size(); ++i)
//	{
//		vector<Dtype> reform_anchor = whctrs(ratio_anchors[i]);
//		Dtype x_ctr = reform_anchor[2];
//		Dtype y_ctr = reform_anchor[3];
//		Dtype w = reform_anchor[0];
//		Dtype h = reform_anchor[1];
//		for (int j = 0; j < num_scales; ++j)
//		{
//			Dtype scaled_w = w * scales[j];
//			Dtype scaled_h = h * scales[j];
//
//			p_anchors[0] = (x_ctr - (Dtype)0.5*(scaled_w - (Dtype)1));
//			p_anchors[1] = (y_ctr - (Dtype)0.5*(scaled_h - (Dtype)1));
//			p_anchors[2] = (x_ctr + (Dtype)0.5*(scaled_w - (Dtype)1));
//			p_anchors[3] = (y_ctr + (Dtype)0.5*(scaled_h - (Dtype)1));
//			//std::cout << "anchors[0] " << p_anchors[0] << " anchors[1] " << p_anchors[1] << " anchors[2] " << p_anchors[2] << " anchors[3] " << p_anchors[3] << std::endl;
//			p_anchors += 4;
//		}
//	}
//}

template <typename Dtype>
static int enumerate_proposals_cpu(const Dtype bottom4d[],
                                    const Dtype d_anchor4d[],
                                    const Dtype anchors[],
                                    Dtype proposals[],
                                    const int num_anchors,
                                    const int bottom_H, const int bottom_W,
                                    const Dtype img_H, const Dtype img_W,
                                    const Dtype min_box_H, const Dtype min_box_W,
                                    const int feat_stride)
{
  Dtype *p_proposal = proposals;
  const int bottom_area = bottom_H * bottom_W;

    int valid_proposal_cnt=0;
  for (int h = 0; h < bottom_H; ++h)
  {
    for (int w = 0; w < bottom_W; ++w)
    {
      const Dtype x = w * feat_stride;
      const Dtype y = h * feat_stride;
      const Dtype *p_box = d_anchor4d + h * bottom_W + w;
      const Dtype *p_score = bottom4d + h * bottom_W + w;
      for (int k = 0; k < num_anchors; ++k)
      {
        if (p_score[k*bottom_area]<0.05) continue;
        else
          valid_proposal_cnt++; 
        const Dtype dx = p_box[(k * 4 + 0) * bottom_area];
        const Dtype dy = p_box[(k * 4 + 1) * bottom_area];
        const Dtype d_log_w = p_box[(k * 4 + 2) * bottom_area];
        const Dtype d_log_h = p_box[(k * 4 + 3) * bottom_area];

        p_proposal[0] = x + anchors[k * 4 + 0];
        p_proposal[1] = y + anchors[k * 4 + 1];
        p_proposal[2] = x + anchors[k * 4 + 2];
        p_proposal[3] = y + anchors[k * 4 + 3];
        p_proposal[4] = transform_box(p_proposal,
                                      dx, dy, d_log_w, d_log_h,
                                      img_W, img_H, min_box_W, min_box_H) *
                        p_score[k * bottom_area];
        p_proposal += 5;
      } // endfor k
    }   // endfor w
  }     // endfor h
  return valid_proposal_cnt;
}

template <typename Dtype>
static void retrieve_rois_cpu(const int num_rois,
                              const int item_index,
                              const Dtype proposals[],
                              const int roi_indices[],
                              Dtype rois[],
                              Dtype roi_scores[])
{
  for (int i = 0; i < num_rois; ++i)
  {
    const Dtype *const proposals_index = proposals + roi_indices[i] * 5;
    rois[i * 5 + 0] = item_index;
    rois[i * 5 + 1] = proposals_index[0];
    rois[i * 5 + 2] = proposals_index[1];
    rois[i * 5 + 3] = proposals_index[2];
    rois[i * 5 + 4] = proposals_index[3];
    if (roi_scores)
    {
      roi_scores[i] = proposals_index[4];
    }
  }
}

template <typename Dtype>
void ProposalAndDetectionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*> &bottom,
                                      const vector<Blob<Dtype>*> &top)
{

  ProposalAndDetectionParameter param = this->layer_param_.proposal_detection_param();

  base_size_ = param.base_size();
  feat_stride_ = param.feat_stride();
  pre_nms_topn_ = param.pre_nms_topn();
  post_nms_topn_ = param.post_nms_topn();
  nms_thresh_ = param.nms_thresh();
  min_size_ = param.min_size();
  numcls_ = param.numcls();

  vector<Dtype> ratios(param.ratio_size());
  for (int i = 0; i < param.ratio_size(); ++i)
  {
    ratios[i] = param.ratio(i);
  }
  vector<Dtype> scales(param.scale_size());
  for (int i = 0; i < param.scale_size(); ++i)
  {
    scales[i] = param.scale(i);
  }

  vector<int> anchors_shape(2);
  anchors_shape[0] = ratios.size() * scales.size();
  anchors_shape[1] = 4;
  anchors_.Reshape(anchors_shape);
  generate_anchors(base_size_, &ratios[0], &scales[0],
                   ratios.size(), scales.size(),
                   anchors_.mutable_cpu_data());

  vector<int> roi_indices_shape(1);
  roi_indices_shape[0] = post_nms_topn_;
  roi_indices_.Reshape(roi_indices_shape);

  vector<int> top_shape(2);
  top_shape[0] = bottom[0]->shape(0) * post_nms_topn_;
  top_shape[1] = 5;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void ProposalAndDetectionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top)
{
	int num_anchors = anchors_.shape(0);

	//=====================================
	Blob<Dtype>* scores = bottom[0];

	Blob<Dtype> scores_trans;
	scores_trans.Reshape(scores->num(), scores->height(), scores->width(),
		scores->channels() - num_anchors);

	Dtype* scores_trans_data = scores_trans.mutable_cpu_data();
	for (int n = 0; n < scores_trans.num(); ++n) {
		for (int c = 0; c < scores_trans.channels(); ++c) {
			for (int h = 0; h < scores_trans.height(); ++h) {
				for (int w = 0; w < scores_trans.width(); ++w) {
					scores_trans_data[scores_trans.offset(n, c, h, w)] =
						scores->data_at(n, w + num_anchors, c, h);
				}
			}
		}
	}

	//=====================================
	Blob<Dtype> bbox_delta_trans;
	bbox_delta_trans.Reshape(bottom[1]->num(), bottom[1]->height(),
		bottom[1]->width(), bottom[1]->channels());
	Dtype* bbox_delta_trans_data = bbox_delta_trans.mutable_cpu_data();
	for (int n = 0; n < bbox_delta_trans.num(); ++n) {
		for (int c = 0; c < bbox_delta_trans.channels(); ++c) {
			for (int h = 0; h < bbox_delta_trans.height(); ++h) {
				for (int w = 0; w < bbox_delta_trans.width(); ++w) {
					bbox_delta_trans_data[bbox_delta_trans.offset(n, c, h, w)] =
						bottom[1]->data_at(n, w, c, h);
				}
			}
		}
	}

	//only num = 1
	//bbox_delta_trans.Reshape(1, 1, bbox_delta_trans.count() / 4, 4);

	bbox_delta_trans_data = bbox_delta_trans.mutable_cpu_data();

	//=====================================
	const int bottom_H = bottom[0]->height();
	const int bottom_W = bottom[0]->width();
	// input image height & width
	const Dtype *p_img_info_cpu = bottom[2]->cpu_data();
	const Dtype img_H = p_img_info_cpu[0];
	const Dtype img_W = p_img_info_cpu[1];
	// scale factor for height & width
	const Dtype scale_H = p_img_info_cpu[2];
	const Dtype scale_W = p_img_info_cpu[3];
	// minimum box width & height
	const Dtype min_box_H = min_size_ * scale_H;
	const Dtype min_box_W = min_size_ * scale_W;

	const int num_proposals = anchors_.shape(0) * bottom_H * bottom_W;

	Blob<Dtype>* target_scores = bottom[3];
	target_scores->Reshape(numcls_, anchors_.shape(0), bottom_H, bottom_W);

	Blob<Dtype> target_scores_trans;
	target_scores_trans.Reshape(target_scores->height(), target_scores->width(), target_scores->channels(), target_scores->num());

	Dtype* target_scores_trans_data = target_scores_trans.mutable_cpu_data();
	for (int n = 0; n < target_scores_trans.num(); ++n) {
		for (int c = 0; c < target_scores_trans.channels(); ++c) {
			for (int h = 0; h < target_scores_trans.height(); ++h) {
				for (int w = 0; w < target_scores_trans.width(); ++w) {
					target_scores_trans_data[target_scores_trans.offset(n, c, h, w)] =
						target_scores->data_at(w, h, n, c);
				}
			}
		}
	}

	//std::cout << "bbox_delta_trans cnt " << bbox_delta_trans.count() << " scores_trans " << scores_trans.count() << " target_scores_trans " << target_scores_trans.count() << std::endl;

	Blob<Dtype> filter_scores;
	vector<int> scores_shape(2);
	scores_shape[0] = num_proposals;
	scores_shape[1] = 1;
	filter_scores.Reshape(scores_shape);

	Blob<Dtype> pred_boxes;
	vector<int> pred_boxes_shape(2);
	pred_boxes_shape[0] = num_proposals;
	pred_boxes_shape[1] = 4;
	pred_boxes.Reshape(pred_boxes_shape);

	Blob<Dtype> target_filter_scores;
	vector<int> target_scores_shape(2);
	target_scores_shape[0] = num_proposals;
	target_scores_shape[1] = numcls_;
	target_filter_scores.Reshape(target_scores_shape);

	int valid_proposal_cnt = enumerate_blob_cpu(scores_trans_data, bbox_delta_trans_data, target_scores_trans_data,
		anchors_.cpu_data(), anchors_.shape(0),
		pred_boxes.mutable_cpu_data(), filter_scores.mutable_cpu_data(), target_filter_scores.mutable_cpu_data(),
		bottom_H, bottom_W, img_H, img_W, min_box_H, min_box_W,
		feat_stride_, numcls_);

	//std::cout << "valid_proposal_cnt " << valid_proposal_cnt << std::endl;
	//std::cout << "pred_boxes cnt " << pred_boxes.count() << " filter_scores " << filter_scores.count() << " target_filter_scores " << target_filter_scores.count() << std::endl;

	scores_shape[0] = valid_proposal_cnt;
	filter_scores.Reshape(scores_shape);

	pred_boxes_shape[0] = valid_proposal_cnt;
	pred_boxes.Reshape(pred_boxes_shape);

	target_scores_shape[0] = valid_proposal_cnt;
	target_filter_scores.Reshape(target_scores_shape);

	top[0]->Reshape(pred_boxes.shape());
	top[0]->CopyFrom(pred_boxes);

	top[1]->Reshape(filter_scores.shape());
	top[1]->CopyFrom(filter_scores);

	top[2]->Reshape(target_filter_scores.shape());
	top[2]->CopyFrom(target_filter_scores);
}

template <typename Dtype>
void ProposalAndDetectionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	this->Forward_cpu(bottom, top);
}

#ifdef CPU_ONLY
STUB_GPU(ProposalAndDetectionLayer);
#endif

INSTANTIATE_CLASS(ProposalAndDetectionLayer);
REGISTER_LAYER_CLASS(ProposalAndDetection);

} // namespace caffe
