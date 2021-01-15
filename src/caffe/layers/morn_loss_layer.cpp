#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
//#include "caffe/vision_layers.hpp"
#include "caffe/morn_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void MornLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	LossLayer<Dtype>::LayerSetUp(bottom, top);

	string prefix = "\t\tMorn Loss Layer:: LayerSetUp: \t";

	std::cout<<prefix<<"Getting output_H_ and output_W_"<<std::endl;

	output_H_ = bottom[0]->shape(2);
	if (this->layer_param_.morn_loss_param().has_output_h()) {
		output_H_ = this->layer_param_.morn_loss_param().output_h();
	}
	output_W_ = bottom[0]->shape(3);
	if (this->layer_param_.morn_loss_param().has_output_w()) {
		output_W_ = this->layer_param_.morn_loss_param().output_w();
	}

	std::cout<<prefix<<"output_H_ = "<<output_H_<<", output_W_ = "<<output_W_<<std::endl;

	// initialize the matrix for output grid
	std::cout << prefix << "Initializing the matrix for output grid" << std::endl;

	vector<int> shape_output(2);
	shape_output[0] = 2; shape_output[1] = output_H_ * output_W_;
	output_grid.Reshape(shape_output);

	Dtype* data = output_grid.mutable_cpu_data();
	for (int i = 0; i<output_H_ * output_W_; ++i) {
		data[i] = (i / output_W_) * 1.0 / output_H_ * 2 - 1;
		data[output_H_ * output_W_ + i] = (i % output_W_) * 1.0 / output_W_ * 2 - 1;
	}

	// initialize the matrix for input grid
	std::cout << prefix << "Initializing the matrix for input grid" << std::endl;

	vector<int> shape_input(3);
	shape_input[0] = bottom[0]->shape(0); shape_input[1] = 2; shape_input[2] = output_H_ * output_W_;
	input_grid.Reshape(shape_input);

	std::cout << prefix << "Initialization finished." << std::endl;
}

template <typename Dtype>
void MornLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	vector<int> tot_loss_shape(0);  // Loss layers output a scalar; 0 axes.
	top[0]->Reshape(tot_loss_shape);

	N = bottom[0]->shape(0);
	int H = bottom[0]->shape(2);
	int W = bottom[0]->shape(3);

	output_H_ = H;
	output_W_ = W;

	vector<int> shape_output(2);
	shape_output[0] = 2; shape_output[1] = output_H_ * output_W_;
	output_grid.Reshape(shape_output);

	Dtype* data = output_grid.mutable_cpu_data();
	for (int i = 0; i<output_H_ * output_W_; ++i) {
		data[i] = (i / output_W_) * 1.0 / output_H_ * 2 - 1;
		data[output_H_ * output_W_ + i] = (i % output_W_) * 1.0 / output_W_ * 2 - 1;
	}

	vector<int> shape_input(3);
	shape_input[0] = bottom[0]->shape(0); shape_input[1] = 2; shape_input[2] = output_H_ * output_W_;
	input_grid.Reshape(shape_input);

	vector<int> loss_shape(3);
	loss_shape[0] = N;
	loss_shape[1] = output_H_;
	loss_shape[2] = output_W_;
	loss_.Reshape(loss_shape);
}

template <typename Dtype>
void MornLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	/* not implemented */
	CHECK(false) << "Error: not implemented.";
}

template <typename Dtype>
void MornLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	/* not implemented */
	CHECK(false) << "Error: not implemented.";
}

#ifdef CPU_ONLY
STUB_GPU(MornLossLayer);
#endif

INSTANTIATE_CLASS(MornLossLayer);
REGISTER_LAYER_CLASS(MornLoss);

}  // namespace caffe
