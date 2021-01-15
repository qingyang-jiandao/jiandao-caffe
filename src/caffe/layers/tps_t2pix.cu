#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/layers/tps_t2pix.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
__global__ void TPStransformerForwardGPU(const int nthreads, int N, int C,
		int output_H_, int output_W_, int H, int W,
		const Dtype* input_grid_data, const Dtype* U, Dtype* V) {
	
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int j = (index / (output_W_ * output_H_)) % C;
		const int i = index / (output_W_ * output_H_ * C);

		const Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 2) * i;
		const int row_idx = output_W_ * s + t;

	  	const Dtype px = coordinates[row_idx * 2];
	  	const Dtype py = coordinates[row_idx * 2 + 1];

	  	const int V_offset = index;

	  	V[V_offset] = (Dtype)0.;

	  	const Dtype x = (px) * H;
	  	const Dtype y = (py) * W;

	  	int m, n; Dtype w;
	  	const Dtype* pic = U + i * (C * H * W) + j * (H * W);

	  	m = floor(x); n = floor(y); w = 0;
	  	if(m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (x - m)) * (1 - (y - n));
	  		V[V_offset] += w * pic[m * W + n];
	  	}

	  	m = floor(x) + 1; n = floor(y); w = 0;
	  	if(m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (m - x)) * (1 - (y - n));
	  		V[V_offset] += w * pic[m * W + n];
	  	}

	  	m = floor(x); n = floor(y) + 1; w = 0;
	  	if(m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (x - m)) * (1 - (n - y));
	  		V[V_offset] += w * pic[m * W + n];
	  	}

	  	m = floor(x) + 1; n = floor(y) + 1; w = 0;
	  	if(m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (m - x)) * (1 - (n - y));
	  		V[V_offset] += w * pic[m * W + n];
	  	}
  }
}

template <typename Dtype>
void TPStransformerLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	string prefix = "TPStransformerLayer::Forward_gpu::\t";
	//std::cout << prefix << "Starting!" << std::endl;
	//test_defined_count++;
	//Forward_cpu(bottom, top);

	
	const Dtype* U = bottom[0]->gpu_data();
	const Dtype* T = bottom[1]->gpu_data();
	const Dtype* output_grid_data = output_grid.mutable_gpu_data();
	Dtype* input_grid_data = input_grid.mutable_gpu_data();
	Dtype* V = top[0]->mutable_gpu_data();

	caffe_gpu_set(input_grid.count(), (Dtype)0, input_grid_data);
	caffe_gpu_set(top[0]->count(), (Dtype)0, V);

	int coff_num = bottom[1]->shape(1);

	// compute out input_grid_data
	for(int i = 0; i < N; ++i) {
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, output_H_ * output_W_, 2, mat_dim, (Dtype)1.,
				output_grid_data, T + coff_num * i, (Dtype)0.,
				input_grid_data + (output_H_ * output_W_ * 2) * i);
	}

	const int nthreads = N * C * output_H_ * output_W_;
	TPStransformerForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
	      CAFFE_CUDA_NUM_THREADS>>>(nthreads, N, C, output_H_, output_W_, H, W, input_grid_data, U, V);
    

	//std::cout << prefix << "ending!" << std::endl;


	//if (test_defined_count == 100)
	//{
	//	std::cout << "gpu input_grid" << std::endl;
	//	//const Dtype* input_grid_data_test = input_grid.cpu_data();
	//	//for (int index = 0; index < output_H_ * output_W_; ++index) {
	//	//	Dtype wy = input_grid_data_test[2 * index + 1];
	//	//	std::cout << wy << " ";
	//	//}
	//	//std::cout << std::endl << std::endl;

	//	const Dtype* V_test = top[0]->cpu_data();
	//	for (int index = 0; index < output_H_ * output_W_; ++index) {
	//		Dtype wy = V_test[3 * index + 1];
	//		std::cout << wy << " ";
	//	}
	//	std::cout << std::endl << std::endl;

	//	Dtype* break_ptr = 0;
	//	*break_ptr = 1;
	//}
}

template <typename Dtype>
__global__ void TPStransformerBackwardGPU_dTheta(const int nthreads, int C,
		int output_H_, int output_W_, int H, int W, int mat_dim,
		const Dtype* input_grid_data, const Dtype* output_grid_data, const Dtype* dV_array, const Dtype* U_array,
		Dtype* dTheta_tmp_diff) {
	
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int j = (index / (output_W_ * output_H_)) % C;
		const int i = index / (output_W_ * output_H_ * C);

		const Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 2) * i;
		const int row_idx = output_W_ * s + t;

		const Dtype px = coordinates[row_idx * 2];
		const Dtype py = coordinates[row_idx * 2 + 1];
		
		Dtype delta_dpx = (Dtype)0.;
		Dtype delta_dpy = (Dtype)0.;

		const Dtype x = px * H;
		const Dtype y = py * W;
		const int dV_offset = index;
		const Dtype dV = dV_array[dV_offset];

		int m, n; 
		const Dtype* U = U_array + i * (C * H * W) + j * (H * W);

		// left-bottom neighbor
		m = floor(x); n = floor(y); 
		if(m >= 0 && m < H && n >= 0 && n < W) {
			delta_dpx -= (1 - (y - n)) * U[m * W + n] * dV * H;
			delta_dpy -= (1 - (x - m)) * U[m * W + n] * dV * W;
		}
		
		// left-top neighbor
		m = floor(x); n = floor(y) + 1; 
		if(m >= 0 && m < H && n >= 0 && n < W) {
			delta_dpx -= (1 - (n - y)) * U[m * W + n] * dV * H;
			delta_dpy += (1 - (x - m)) * U[m * W + n] * dV * W;
		}

		// right-bottom neighbor
		m = floor(x) + 1; n = floor(y); 
		if(m >= 0 && m < H && n >= 0 && n < W) {
			delta_dpx += (1 - (y - n)) * U[m * W + n] * dV * H;
			delta_dpy -= (1 - (m - x)) * U[m * W + n] * dV * W;
		}
		
		// right-top neighbor
		m = floor(x) + 1; n = floor(y) + 1; 
		if(m >= 0 && m < H && n >= 0 && n < W) {
			delta_dpx += (1 - (n - y)) * U[m * W + n] * dV * H;
			delta_dpy += (1 - (m - x)) * U[m * W + n] * dV * W;
		}

		int idx = j * (output_H_ * output_W_) + s * output_W_ + t;

		for (int v = 0; v < mat_dim; v++) {
			dTheta_tmp_diff[(2 * mat_dim * i + v) * (output_H_ * output_W_ * C) + idx] += delta_dpx * output_grid_data[row_idx * mat_dim + v];
		}

		for (int v = mat_dim; v < mat_dim*2; v++) {
			dTheta_tmp_diff[(2 * mat_dim * i + v) * (output_H_ * output_W_ * C) + idx] += delta_dpy * output_grid_data[row_idx * mat_dim + v - mat_dim];
		}
	}
}

template <typename Dtype>
__global__ void TPStransformerBackwardGPU_dU(const int nthreads, const int C,
	const int W,  const int H, const int output_H_, const int output_W_, 
	const Dtype* input_grid_data, const Dtype* dV, Dtype* dU) {
	
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int j = (index / (output_W_ * output_H_)) % C;
		const int i = index / (output_W_ * output_H_ * C);

		const Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 2) * i;
		const int row_idx = output_W_ * s + t;

		const Dtype px = coordinates[row_idx * 2];
		const Dtype py = coordinates[row_idx * 2 + 1];

	  	const int V_offset = index;

		const Dtype x = px * H;
		const Dtype y = py * W;

	  	int m, n; Dtype w;
	  	Dtype* pic = dU + i * (C * H * W) + j * (H * W);

	  	m = floor(x); n = floor(y); w = 0;
	  	if(m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (x - m)) * (1 - (y - n));
			caffe_gpu_atomic_add(w * dV[V_offset], pic + (m * W + n));
	  	}

	  	m = floor(x) + 1; n = floor(y); w = 0;
	  	if(m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (m - x)) * (1 - (y - n));
			caffe_gpu_atomic_add(w * dV[V_offset], pic + (m * W + n));
	  	}

	  	m = floor(x); n = floor(y) + 1; w = 0;
	  	if(m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (x - m)) * (1 - (n - y));
			caffe_gpu_atomic_add(w * dV[V_offset], pic + (m * W + n));
	  	}

	  	m = floor(x) + 1; n = floor(y) + 1; w = 0;
	  	if(m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (m - x)) * (1 - (n - y));
			caffe_gpu_atomic_add(w * dV[V_offset], pic + (m * W + n));
	  	}
	}
}

template <typename Dtype>
void TPStransformerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	string prefix = "TPStransformerLayer::Backward_gpu::\t";
	//std::cout << prefix << "Starting!" << std::endl;
	//test_defined_count++;
	//Backward_cpu(top, propagate_down, bottom);

	
	const Dtype* dV = top[0]->gpu_diff();
	const Dtype* U = bottom[0]->gpu_data();
	const Dtype* input_grid_data = input_grid.gpu_data();
	const Dtype* output_grid_data = output_grid.gpu_data();

	Dtype* dT = bottom[1]->mutable_gpu_diff();
	Dtype* dT_tmp_diff = dT_tmp.mutable_gpu_diff();

	caffe_gpu_set(bottom[1]->count(), (Dtype)0, dT);
	caffe_gpu_set(dT_tmp.count(), (Dtype)0., dT_tmp_diff);

	const int mat_count = bottom[1]->shape(1);
	const int local_mat_dim = mat_count/2;
	//std::cout << prefix << "processing!" << std::endl;
	//std::cout << "local_mat_dim:" << local_mat_dim << std::endl;
	const int nthreads = N * C * output_H_ * output_W_;
	TPStransformerBackwardGPU_dTheta<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
			CAFFE_CUDA_NUM_THREADS>>>(nthreads, C, output_H_, output_W_, H, W, local_mat_dim, input_grid_data, output_grid_data,
					dV, U, dT_tmp_diff);

	Dtype* all_ones_2_data = all_ones_2.mutable_gpu_data();
	caffe_gpu_set(all_ones_2.count(), (Dtype)1., all_ones_2_data);
	
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, bottom[1]->count(), 1, output_H_ * output_W_ * C,
			(Dtype)1., dT_tmp_diff, all_ones_2_data, (Dtype)0., dT);

	//std::cout << prefix << "ending!" << std::endl;
	

	//if (test_defined_count == 1000)
	//{
	//	std::cout << "----------------dT gpu-------------------" << std::endl;
	//	const Dtype* dT_test = bottom[1]->cpu_diff();
	//	for (int i = 0; i < bottom[1]->count(); ++i) {
	//		std::cout << dT_test[i] << " ";
	//	}
	//	std::cout << std::endl;

	//	Dtype* break_ptr = 0;
	//	*break_ptr = 1;
	//}

	//if(to_compute_dU_) {
	//	Dtype* dU = bottom[0]->mutable_gpu_diff();
	//	caffe_gpu_set(bottom[0]->count(), (Dtype)0., dU);
	//	const int nthreads = N * C * output_H_ * output_W_;
	//	TPStransformerBackwardGPU_dU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
	//		CAFFE_CUDA_NUM_THREADS>>>(nthreads, C, W, H, output_H_, output_W_, input_grid_data, dV, dU);
	//}
}

INSTANTIATE_LAYER_GPU_FUNCS(TPStransformerLayer);

}	// namespace caffe
