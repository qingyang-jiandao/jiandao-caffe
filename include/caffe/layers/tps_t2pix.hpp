#ifndef TPS_AfterT_HPP_
#define TPS_AfterT_HPP_

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
class TPStransformerLayer : public Layer<Dtype> {
				
public:
		explicit TPStransformerLayer(const LayerParameter& param)
		: Layer<Dtype>(param) {
			K = 20;
			margin_x = margin_y = 0.05;
			to_compute_dU_ = false;
			global_debug = false; 
			test_defined_count = 0;
		}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
														const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
													const vector<Blob<Dtype>*>& top);
				
		virtual inline const char* type() const { return "TPStransformer"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
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
				
private:
		inline Dtype abs(Dtype x) {
				if(x < 0) return -x; return x;
		}
		inline Dtype max(Dtype x, Dtype y) {
				if(x < y) return y; return x;
		}
				
		Dtype transform_forward_cpu(const Dtype* pic, Dtype px, Dtype py);
		void transform_backward_cpu(Dtype dV, const Dtype* U, const Dtype px,
																const Dtype py, Dtype* dU, Dtype& dpx, Dtype& dpy);
				
						 
		int output_H_;
		int output_W_;

		int test_defined_count;
				
		int N, C, H, W, K, mat_dim;

		Dtype margin_x, margin_y;
				
		bool global_debug;
		bool to_compute_dU_;
				
		Blob<Dtype> dT_tmp;	// used for back propagation part in GPU implementation
		Blob<Dtype> all_ones_2;	// used for back propagation part in GPU implementation
				
		//Blob<Dtype> full_T;	// used for storing data and diff for full six-dim T
				
		Blob<Dtype> output_grid; // standard output coordinate2 system, [0, 1) by [0, 1).---------
		Blob<Dtype> input_grid;	// corresponding coordinate on input image after projection for each output pixel.
};
		
}  // namespace caffe

#endif  // CAFFE_COMMON_HPP_
