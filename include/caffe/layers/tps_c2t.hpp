#ifndef TPS_BEFORET_HPP_
#define TPS_BEFORET_HPP_

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class CToParaLayer : public Layer<Dtype> {

public:
	explicit CToParaLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {
		K = 20;
		margin_x = margin_y = 0.05;
		test_defined_count = 0;
	}
					
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "CToPara"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	string transform_type_;
	string sampler_type_;
	 
private:
	int N, C, K;
	int test_defined_count;
	Dtype margin_x, margin_y;
	Blob<Dtype> inv_delta_c;
	Blob<Dtype> full_y;
	Blob<Dtype> dC_tmp;	// used for back propagation part in GPU implementation
	Blob<Dtype> all_ones_2;	// used for back propagation part in GPU implementation
};

}  // namespace caffe

#endif  // CAFFE_COMMON_HPP_

			