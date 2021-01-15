#include <vector>

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/morn_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MornLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	string prefix = "\t\tMornLayer Layer:: LayerSetUp: \t";

	if(this->layer_param_.morn_param().transform_type() == "affine") {
		transform_type_ = "affine";
	} else {
		CHECK(false) << prefix << "Transformation type only supports affine now!" << std::endl;
	}

	if(this->layer_param_.morn_param().sampler_type() == "bilinear") {
		sampler_type_ = "bilinear";
	} else {
		CHECK(false) << prefix << "Sampler type only supports bilinear now!" << std::endl;
	}

	if(this->layer_param_.morn_param().to_compute_du()) {
		to_compute_dU_ = true;
	}

	std::cout<<prefix<<"Getting output_H_ and output_W_"<<std::endl;

	output_H_ = bottom[0]->shape(2);
	if(this->layer_param_.morn_param().has_output_h()) {
		output_H_ = this->layer_param_.morn_param().output_h();
	}
	output_W_ = bottom[0]->shape(3);
	if(this->layer_param_.morn_param().has_output_w()) {
		output_W_ = this->layer_param_.morn_param().output_w();
	}

	std::cout<<prefix<<"output_H_ = "<<output_H_<<", output_W_ = "<<output_W_<<std::endl;

	std::cout<<prefix<<"Getting pre-defined parameters"<<std::endl;

	// initialize the matrix for output grid
	std::cout<<prefix<<"Initializing the matrix for output grid"<<std::endl;

	vector<int> shape_output(2);
	shape_output[0] = output_H_ * output_W_; shape_output[1] = 2;
	output_grid.Reshape(shape_output);
    
	Dtype* data = output_grid.mutable_cpu_data();
	for(int i=0; i<output_H_ * output_W_; ++i) {
		data[2 * i] = (i / output_W_) * 1.0 / output_H_ * 2 - 1;
		data[2 * i + 1] = (i % output_W_) * 1.0 / output_W_ * 2 - 1;
	}

	// initialize the matrix for input grid
	std::cout<<prefix<<"Initializing the matrix for input grid"<<std::endl;

	vector<int> shape_input(3);
	shape_input[0] = bottom[1]->shape(0); shape_input[1] = output_H_ * output_W_; shape_input[2] = 2;
	input_grid.Reshape(shape_input);

	std::cout<<prefix<<"Initialization finished."<<std::endl;
}

template <typename Dtype>
void MornLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	string prefix = "\t\tMornLayer:: Reshape: \t";

	if(global_debug) std::cout<<prefix<<"Starting!"<<std::endl;

	N = bottom[0]->shape(0);
	C = bottom[0]->shape(1);
	H = bottom[0]->shape(2);
	W = bottom[0]->shape(3);

	offset_N = bottom[1]->shape(0);
	offset_C = bottom[1]->shape(1);
	offset_H = bottom[1]->shape(2);
	offset_W = bottom[1]->shape(3);
	
	output_H_ = H;
	output_W_ = W;
	vector<int> shape_output(2);
	shape_output[0] = output_H_ * output_W_; shape_output[1] = 2;
	output_grid.Reshape(shape_output);

	Dtype* data = output_grid.mutable_cpu_data();
	for (int i = 0; i < output_H_ * output_W_; ++i) {
		data[2 * i] = (i / output_W_) * 1.0 / output_H_ * 2 - 1;
		data[2 * i + 1] = (i % output_W_) * 1.0 / output_W_ * 2 - 1;
	}

	vector<int> shape_input(3);
	shape_input[0] = bottom[1]->shape(0); shape_input[1] = output_H_ * output_W_; shape_input[2] = 2;
	input_grid.Reshape(shape_input);

	// reshape V
	vector<int> shape(4);

	shape[0] = N;
	shape[1] = C;
	shape[2] = output_H_;
	shape[3] = output_W_;

	top[0]->Reshape(shape);

	// reshape dOffset_tmp
	vector<int> dOffset_tmp_shape(3);
	dOffset_tmp_shape[0] = bottom[1]->shape(0); dOffset_tmp_shape[1] = bottom[1]->shape(2) * bottom[1]->shape(3); dOffset_tmp_shape[2] = bottom[1]->shape(1);
	offsets_trans.Reshape(dOffset_tmp_shape);

	//Dtype* offsets_trans_data = offsets_trans.mutable_cpu_data();
	//for (int n = 0; n < bottom[1]->num(); ++n) {
	//	for (int c = 0; c < bottom[1]->channels(); ++c) {
	//		for (int h = 0; h < bottom[1]->height(); ++h) {
	//			for (int w = 0; w < bottom[1]->width(); ++w) {
	//				offsets_trans_data[offsets_trans.offset(n, h*bottom[1]->width() + w, c)] =
	//					bottom[1]->data_at(n, c, h, w);
	//			}
	//		}
	//	}
	//}

	if(global_debug) std::cout<<prefix<<"Finished."<<std::endl;
}

template <typename Dtype>
Dtype MornLayer<Dtype>::transform_forward_cpu(const Dtype* pic, Dtype px, Dtype py) {

	bool debug = false;

	string prefix = "\t\tMornLayer:: transform_forward_cpu: \t";

	if(debug) std::cout<<prefix<<"Starting!\t"<<std::endl;
	if(debug) std::cout<<prefix<<"(px, py) = ("<<px<<", "<<py<<")"<<std::endl;

	Dtype res = (Dtype)0.;

	Dtype x = (px + 1) / 2 * H; Dtype y = (py + 1) / 2 * W;

	if(debug) std::cout<<prefix<<"(x, y) = ("<<x<<", "<<y<<")"<<std::endl;

	int m, n; Dtype w;

	m = floor(x); n = floor(y); w = 0;
	if (m >= 0 && m < H && n >= 0 && n < W) {
		w = (1 - (x - m)) * (1 - (y - n));
		res += w * pic[m * W + n];
	}

	m = floor(x) + 1; n = floor(y); w = 0;
	if (m >= 0 && m < H && n >= 0 && n < W) {
		w = (1 - (m - x)) * (1 - (y - n));
		res += w * pic[m * W + n];
	}

	m = floor(x); n = floor(y) + 1; w = 0;
	if (m >= 0 && m < H && n >= 0 && n < W) {
		w = (1 - (x - m)) * (1 - (n - y));
		res += w * pic[m * W + n];
	}

	m = floor(x) + 1; n = floor(y) + 1; w = 0;
	if (m >= 0 && m < H && n >= 0 && n < W) {
		w = (1 - (m - x)) * (1 - (n - y));
		res += w * pic[m * W + n];
	}

	if(debug) std::cout<<prefix<<"Finished. \tres = "<<res<<std::endl;

	return res;
}

template <typename Dtype>
void MornLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	string prefix = "\t\tMornLayer:: Forward_cpu: \t";

	if(global_debug) std::cout<<prefix<<"Starting!"<<std::endl;

	//Blob<Dtype> offsets_trans;
	//vector<int> shape_offset(3);
	//shape_offset[0] = bottom[1]->shape(0); shape_offset[1] = bottom[1]->shape(2) * bottom[1]->shape(3); shape_offset[2] = bottom[1]->shape(1);
	//offsets_trans.Reshape(shape_offset);

	Dtype* offsets_trans_data = offsets_trans.mutable_cpu_data();
	for (int n = 0; n < bottom[1]->num(); ++n) {
		for (int c = 0; c < bottom[1]->channels(); ++c) {
			for (int h = 0; h < bottom[1]->height(); ++h) {
				for (int w = 0; w < bottom[1]->width(); ++w) {
					offsets_trans_data[offsets_trans.offset(n, h*bottom[1]->width()+w, c)] =
						bottom[1]->data_at(n, c, h, w);
				}
			}
		}
	}

	const Dtype* U = bottom[0]->cpu_data();
	const Dtype* offset_data = offsets_trans.cpu_data();
	const Dtype* output_grid_data = output_grid.cpu_data();

	Dtype* input_grid_data = input_grid.mutable_cpu_data();
	Dtype* V = top[0]->mutable_cpu_data();

	caffe_set(input_grid.count(), (Dtype)0, input_grid_data);
	caffe_set(top[0]->count(), (Dtype)0, V);

	// for each input
	for(int i = 0; i < N; ++i) {

		Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 2) * i;
		const Dtype* curr_offsets = offset_data + (output_H_ * output_W_ * 2) * i;

		const int grid_count = output_H_ * output_W_ * 2;
		caffe_axpy<Dtype>(grid_count, 1.0, output_grid_data, coordinates);
		caffe_axpy<Dtype>(grid_count, 1.0, curr_offsets, coordinates);

		int row_idx; Dtype px, py;

		for(int j = 0; j < C; ++j)
			for(int s = 0; s < output_H_; ++s)
				for(int t = 0; t < output_W_; ++t) {

					row_idx = output_W_ * s + t;

					px = coordinates[row_idx * 2];
					py = coordinates[row_idx * 2 + 1];

					V[top[0]->offset(i, j, s, t)] = transform_forward_cpu(
							U + bottom[0]->offset(i, j, 0, 0), px, py);
				}
	}

	//std::cout << "input_grid" << std::endl;
	//const Dtype* debug_input_grid = input_grid.cpu_data();
	//for(int i=0; i<input_grid.count(); ++i) {
	//	std::cout << debug_input_grid[i] << " ";
	//}
	//std::cout<<std::endl << std::endl;

	//std::cout << "vvvvvvvvvvv" << std::endl;
	//for (int i = 0; i<top[0]->count(); ++i) {
	//	std::cout << V[i] << " ";
	//}
	//std::cout << std::endl << std::endl;

	if(global_debug) std::cout<<prefix<<"Finished."<<std::endl;
}

template <typename Dtype>
void MornLayer<Dtype>::transform_backward_cpu(Dtype dV, const Dtype* U, const Dtype px,
		const Dtype py, Dtype* dU, Dtype& dpx, Dtype& dpy) {

	bool debug = false;

	string prefix = "\t\tMornLayer:: transform_backward_cpu: \t";

	if(debug) std::cout<<prefix<<"Starting!"<<std::endl;

	Dtype x = (px + 1) / 2 * H; Dtype y = (py + 1) / 2 * W;
	if(debug) std::cout<<prefix<<"(x, y) = ("<<x<<", "<<y<<")"<<std::endl;

	int m, n; Dtype w;

	// left-bottom neighbor
	m = floor(x); n = floor(y);
	if (m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));
		dU[m * W + n] += w * dV;

		dpx -= (1 - (y - n)) * U[m * W + n] * dV * H / 2;
		dpy -= (1 - (x - m)) * U[m * W + n] * dV * W / 2;
	}

	// left-top neighbor
	m = floor(x); n = floor(y) + 1;
	if (m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));
		dU[m * W + n] += w * dV;

		dpx -= (1 - (n - y)) * U[m * W + n] * dV * H / 2;
		dpy += (1 - (x - m)) * U[m * W + n] * dV * W / 2;
	}

	// right-bottom neighbor
	m = floor(x) + 1; n = floor(y);
	if (m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));
		dU[m * W + n] += w * dV;

		dpx += (1 - (y - n)) * U[m * W + n] * dV * H / 2;
		dpy -= (1 - (m - x)) * U[m * W + n] * dV * W / 2;
	}

	// right-top neighbor
	m = floor(x) + 1; n = floor(y) + 1;
	if (m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));
		dU[m * W + n] += w * dV;

		dpx += (1 - (n - y)) * U[m * W + n] * dV * H / 2;
		dpy += (1 - (m - x)) * U[m * W + n] * dV * W / 2;
	}

	if(debug) std::cout<<prefix<<"Finished."<<std::endl;
}

template <typename Dtype>
void MornLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

		string prefix = "\t\tMornLayer:: Backward_cpu: \t";

		if(global_debug) std::cout<<prefix<<"Starting!"<<std::endl;

		const Dtype* dV = top[0]->cpu_diff();
		const Dtype* input_grid_data = input_grid.cpu_data();
		const Dtype* U = bottom[0]->cpu_data();

		Dtype* dU = bottom[0]->mutable_cpu_diff();
		Dtype* dOffset_diff = bottom[1]->mutable_cpu_diff();
		Dtype* input_grid_diff = input_grid.mutable_cpu_diff();

		caffe_set(bottom[0]->count(), (Dtype)0, dU);
		caffe_set(bottom[1]->count(), (Dtype)0, dOffset_diff);
		caffe_set(input_grid.count(), (Dtype)0, input_grid_diff);

		for(int i = 0; i < N; ++i) {

			const Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 2) * i;
			Dtype* coordinates_diff = input_grid_diff + (output_H_ * output_W_ * 2) * i;

			int row_idx; Dtype px, py, dpx, dpy, delta_dpx, delta_dpy;

			for(int s = 0; s < output_H_; ++s)
				for(int t = 0; t < output_W_; ++t) {

					row_idx = output_W_ * s + t;

					px = coordinates[row_idx * 2];
					py = coordinates[row_idx * 2 + 1];

					for(int j = 0; j < C; ++j) {

						delta_dpx = delta_dpy = (Dtype)0.;

						transform_backward_cpu(dV[top[0]->offset(i, j, s, t)], U + bottom[0]->offset(i, j, 0, 0),
								px, py, dU + bottom[0]->offset(i, j, 0, 0), delta_dpx, delta_dpy);

						coordinates_diff[row_idx * 2] += delta_dpx;
						coordinates_diff[row_idx * 2 + 1] += delta_dpy;
					}

					dpx = coordinates_diff[row_idx * 2];
					dpy = coordinates_diff[row_idx * 2 + 1];

					dOffset_diff[bottom[1]->offset(i, 0, s, t)] = (dpx * 2.0 / output_H_);
					dOffset_diff[bottom[1]->offset(i, 1, s, t)] = (dpy * 2.0 / output_W_);
				}
		}

		//std::cout << "input_grid" << std::endl;
		//const Dtype* debug_input_grid = input_grid.cpu_data();
		//for (int i = 0; i<input_grid.count(); ++i) {
		//	std::cout << debug_input_grid[i] << " ";
		//}
		//std::cout << std::endl << std::endl;

		//const Dtype* db_doffset_diff = bottom[1]->cpu_diff();
		//for (int n = 0; n < bottom[1]->num(); ++n) {
		//	for (int c = 0; c < bottom[1]->channels(); ++c) {
		//		std::cout << "ccccccccccccccccc" << std::endl;
		//		for (int h = 0; h < bottom[1]->height(); ++h) {
		//			for (int w = 0; w < bottom[1]->width(); ++w) {
		//				std::cout << db_doffset_diff[bottom[1]->offset(n, c, h, w)] << " ";
		//			}
		//			std::cout << std::endl;
		//		}
		//		std::cout << std::endl << std::endl << std::endl;
		//	}
		//}

		//std::cout << std::endl << std::endl;

		//Dtype* break_ptr = 0;
		//*break_ptr = 1;

		if(global_debug) std::cout<<prefix<<"Finished."<<std::endl;
}

#ifdef CPU_ONLY
STUB_GPU(MornLayer);
#endif

INSTANTIATE_CLASS(MornLayer);
REGISTER_LAYER_CLASS(Morn);

}  // namespace caffe
