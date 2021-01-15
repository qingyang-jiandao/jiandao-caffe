#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
//#include "caffe/vision_layers.hpp"
#include "caffe/morn_loss_layer.hpp"

namespace caffe {

//template <typename Dtype>
//__global__ void STLossForwardGPU(const int nthreads, int N, 
//		int output_H_, int output_W_, const Dtype* theta, Dtype* loss_array) {
//	
//	CUDA_KERNEL_LOOP(index, nthreads) {
//
//		const int t = index % output_W_;
//		const int s = (index / output_W_) % output_H_;
//		const int i = index / (output_W_ * output_H_);
//
//		Dtype input_x = s * 2.0 / output_H_ - 1;
//		Dtype input_y = t * 2.0 / output_W_ - 1;
//		
//		Dtype output_x = theta[6*i] * input_x + theta[6*i+1] * input_y + theta[6*i+2];
//		Dtype output_y = theta[6*i+3] * input_x + theta[6*i+4] * input_y + theta[6*i+5];
//		
//		Dtype loss = (Dtype)0;
//		
//		if(output_x < -1) {
//			loss += (output_x + 1) * (output_x + 1) / 2;
//		} else if(output_x > 1) {
//			loss += (output_x - 1) * (output_x - 1) / 2;
//		}
//		
//		if(output_y < -1) {
//			loss += (output_y + 1) * (output_y + 1) / 2;
//		} else if(output_y > 1) {
//			loss += (output_y - 1) * (output_y - 1) / 2;
//		}
//		
//		loss_array[index] = loss;
//  }
//}

template <typename Dtype>
__global__ void MornLossForwardGPU(const int nthreads, int N,
	int output_H_, int output_W_, const Dtype* input_grid_data, Dtype* loss_array) {

	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int i = index / (output_W_ * output_H_);

		const Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 2) * i;
		const int row_idx = output_W_ * s + t;

		const Dtype output_x = coordinates[row_idx];
		const Dtype output_y = coordinates[output_H_ * output_W_ + row_idx];

		Dtype loss = (Dtype)0;

		if (output_x < -1) {
			loss += (output_x + 1) * (output_x + 1) / 2;
		}
		else if (output_x > 1) {
			loss += (output_x - 1) * (output_x - 1) / 2;
		}

		if (output_y < -1) {
			loss += (output_y + 1) * (output_y + 1) / 2;
		}
		else if (output_y > 1) {
			loss += (output_y - 1) * (output_y - 1) / 2;
		}

		loss_array[index] = loss;
	}
}

template <typename Dtype>
void MornLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	
	string prefix = "MornLossLayer::Forward_gpu::\t";

	const Dtype* offset_data = bottom[0]->gpu_data();
	const Dtype* output_grid_data = output_grid.mutable_gpu_data();

	Dtype* input_grid_data = input_grid.mutable_gpu_data();
	caffe_gpu_set(input_grid.count(), (Dtype)0, input_grid_data);

	for (int i = 0; i < N; ++i) {

		Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 2) * i;
		const Dtype* curr_offsets = offset_data + (output_H_ * output_W_ * 2) * i;

		const int grid_count = output_H_ * output_W_ * 2;
		caffe_gpu_axpy<Dtype>(grid_count, 1.0, output_grid_data, coordinates);
		caffe_gpu_axpy<Dtype>(grid_count, 1.0, curr_offsets, coordinates);
	}


	Dtype* loss_array = loss_.mutable_gpu_data();
	caffe_gpu_set(loss_.count(), (Dtype)0, loss_array);
	
	const int nthreads = N * output_H_ * output_W_;
	MornLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
	     CAFFE_CUDA_NUM_THREADS>>>(nthreads, N, output_H_, output_W_, input_grid_data, loss_array);
	
	Dtype loss;
	caffe_gpu_asum(nthreads, loss_array, &loss);
	loss /= nthreads;
	
	top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
__global__ void MornLossBackwardGPU(const int nthreads, int N, 
		int output_H_, int output_W_, const Dtype* input_grid_data, Dtype* dOffset_diff) {
	
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int i = index / (output_W_ * output_H_);

		Dtype input_x = s * 2.0 / output_H_ - 1;
		Dtype input_y = t * 2.0 / output_W_ - 1;

		const Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 2) * i;
		const int row_idx = output_W_ * s + t;

		const Dtype output_x = coordinates[row_idx];
		const Dtype output_y = coordinates[output_H_ * output_W_ + row_idx];
		
		Dtype d1 = (Dtype)0, d2 = (Dtype)0;
		
		if(output_x < -1) {
			d1 = output_x + 1;
		} else if(output_x > 1) {
			d1 = output_x - 1;
		}
		
		if(output_y < -1) {
			d2 = output_y + 1;
		} else if(output_y > 1) {
			d2 = output_y - 1;
		}

		Dtype* curr_Offset_diff = dOffset_diff + (output_H_ * output_W_ * 2) * i;
		curr_Offset_diff[row_idx] = d1 * input_x;
		curr_Offset_diff[output_H_ * output_W_ + row_idx] = d2 * input_y;
  }
}

template <typename Dtype>
void MornLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	
	const Dtype* offset_data = bottom[0]->gpu_data();
	Dtype* dOffset_diff = bottom[0]->mutable_gpu_diff();
	const Dtype* input_grid_data = input_grid.gpu_data();

	caffe_gpu_set(bottom[0]->count(), (Dtype)0, dOffset_diff);

	const int nthreads = N * output_H_ * output_W_;
	MornLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
	     CAFFE_CUDA_NUM_THREADS>>>(nthreads, N, output_H_, output_W_, input_grid_data, dOffset_diff);
	     
	caffe_gpu_scal(bottom[0]->count(), top[0]->cpu_diff()[0] / nthreads, dOffset_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(MornLossLayer);

}  // namespace caffe
