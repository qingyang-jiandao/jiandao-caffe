#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

//template <typename Dtype>
//void EuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
//    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
//  for (int i = 0; i < 2; ++i) {
//    if (propagate_down[i]) {
//      const Dtype sign = (i == 0) ? 1 : -1;
//      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
//      caffe_gpu_axpby(
//          bottom[i]->count(),              // count
//          alpha,                              // alpha
//          diff_.gpu_data(),                   // a
//          Dtype(0),                           // beta
//          bottom[i]->mutable_gpu_diff());  // b
//    }
//  }
//}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	int N = bottom[0]->shape(0);
	int C = bottom[0]->shape(1);
	int H = bottom[0]->shape(2);
	int W = bottom[0]->shape(3);

	if (bottom.size() == 2) {
		for (int i = 0; i < 2; ++i) {
			if (propagate_down[i]) {
				const Dtype sign = (i == 0) ? 1 : -1;
				const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
				caffe_gpu_axpby(
					bottom[i]->count(),              // count
					alpha,                              // alpha
					diff_.gpu_data(),                   // a
					Dtype(0),                           // beta
					bottom[i]->mutable_gpu_diff());  // b
			}
		}
	}

	//std::cout << N << " " << C << " " << H << " " << W << std::endl;
	//const Dtype* db_dtheta = bottom[2]->cpu_data();
	//for(int i=0; i<bottom[2]->count(); ++i) {
	//std::cout << db_dtheta[i] << " ";
	//}
	//std::cout<<std::endl;

	//Dtype* break_ptr = 0;
	//*break_ptr = 1;

	if (bottom.size() == 3) {
		for (int i = 0; i < 2; ++i) {
			const Dtype* in_diff_data = diff_.gpu_data();
			Dtype* out_diff_data = bottom[i]->mutable_gpu_diff();
			const Dtype* in_lossWeight_data = bottom[2]->cpu_data();
			if (propagate_down[i]) {
				const Dtype sign = (i == 0) ? 1 : -1;
				const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
				for (int n = 0; n < N; ++n) {
					Dtype* curr_out_diff_data = out_diff_data + C*H*W*n;
					const Dtype* curr_in_diff_data = in_diff_data + C*H*W*n;
					//std::cout << "================ " << std::endl;
					Dtype curr_alpha = alpha;
                    if (in_lossWeight_data[n]<0.99)curr_alpha = alpha*(in_lossWeight_data[n]-0.01);
					//std::cout << "================ " << std::endl;
					//std::cout << in_lossWeight_data[n] << " " << alpha << std::endl;

					caffe_gpu_axpby(
						C*H*W,              // count
						curr_alpha,                              // alpha
						curr_in_diff_data,                   // a
						Dtype(0),                           // beta
						curr_out_diff_data);  // b
				}
			}
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossLayer);

}  // namespace caffe
