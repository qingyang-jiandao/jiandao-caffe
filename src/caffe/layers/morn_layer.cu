#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/morn_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
__global__ void set_value_to_constant(const int nthreads, Dtype value, int size, 
	int i, Dtype* dst) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		dst[index * size + i] = value;
	}
}

template <typename Dtype>
__global__ void copy_values(const int nthreads, int size_src, int k, 
	const Dtype* src, int size_dst, int i, Dtype* dst) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		dst[index * size_dst + i] = src[index * size_src + k];
	}
}

template <typename Dtype>
__global__ void MornForwardGPU(const int nthreads, int N, int C,
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

	  	const Dtype x = (px + 1) / 2 * H;
	  	const Dtype y = (py + 1) / 2 * W;

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

//template <typename Dtype>
//__global__ void MornForwardGPU(const int nthreads, int N, int C,
//	int output_H_, int output_W_, int H, int W,
//	const Dtype* coordinates, const Dtype* U, Dtype* V) {
//
//	CUDA_KERNEL_LOOP(index, nthreads) {
//
//		const int t = index % output_W_;
//		const int s = (index / output_W_) % output_H_;
//		const int j = (index / (output_W_ * output_H_)) % C;
//
//		const int row_idx = output_W_ * s + t;
//
//		const Dtype px = coordinates[row_idx * 2];
//		const Dtype py = coordinates[row_idx * 2 + 1];
//
//		const int V_offset = index;
//
//		V[V_offset] = (Dtype)0.;
//
//		const Dtype x = (px + 1) / 2 * H;
//		const Dtype y = (py + 1) / 2 * W;
//
//		int m, n; Dtype w;
//		const Dtype* pic = U + j * (H * W);
//
//		m = floor(x); n = floor(y); w = 0;
//		if (m >= 0 && m < H && n >= 0 && n < W) {
//			w = (1 - (x - m)) * (1 - (y - n));
//			V[V_offset] += w * pic[m * W + n];
//		}
//
//		m = floor(x) + 1; n = floor(y); w = 0;
//		if (m >= 0 && m < H && n >= 0 && n < W) {
//			w = (1 - (m - x)) * (1 - (y - n));
//			V[V_offset] += w * pic[m * W + n];
//		}
//
//		m = floor(x); n = floor(y) + 1; w = 0;
//		if (m >= 0 && m < H && n >= 0 && n < W) {
//			w = (1 - (x - m)) * (1 - (n - y));
//			V[V_offset] += w * pic[m * W + n];
//		}
//
//		m = floor(x) + 1; n = floor(y) + 1; w = 0;
//		if (m >= 0 && m < H && n >= 0 && n < W) {
//			w = (1 - (m - x)) * (1 - (n - y));
//			V[V_offset] += w * pic[m * W + n];
//		}
//	}
//}

template <typename Dtype>
__global__ void transform_offset(const int nthreads, int N,
	int output_H_, int output_W_, const Dtype* offset_data, Dtype* offset_data_trans) {

	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int i = index / (output_W_ * output_H_);

		const Dtype* i_offsets = offset_data + (output_H_ * output_W_ * 2) * i;
		const int row_idx = output_W_ * s + t;

		const Dtype offset_x = i_offsets[row_idx];
		const Dtype offset_y = i_offsets[output_H_ * output_W_ + row_idx];

		Dtype* i_offsets_trans = offset_data_trans + (output_H_ * output_W_ * 2) * i;
		i_offsets_trans[row_idx * 2] = offset_x;
		i_offsets_trans[row_idx * 2 + 1] = offset_y;
	}
}

template <typename Dtype>
void MornLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	string prefix = "MornLayer::Forward_gpu::\t";

	//const Dtype* U = bottom[0]->gpu_data();
	//const Dtype* offset_data = offsets_trans.gpu_data();
	//const Dtype* output_grid_data = output_grid.gpu_data();

	//Dtype* input_grid_data = input_grid.mutable_gpu_data();
	//Dtype* V = top[0]->mutable_gpu_data();

	//caffe_gpu_set(input_grid.count(), (Dtype)0, input_grid_data);
	//caffe_gpu_set(top[0]->count(), (Dtype)0, V);

	//// for each input
	//for (int i = 0; i < N; ++i) {

	//	Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 2) * i;
	//	const Dtype* coordinate_offsets = offset_data + (output_H_ * output_W_ * 2) * i;

	//	const int grid_count = output_H_ * output_W_ * 2;
	//	caffe_gpu_axpy<Dtype>(grid_count, 1.0, output_grid_data, coordinates);
	//	caffe_gpu_axpy<Dtype>(grid_count, 1.0, coordinate_offsets, coordinates);

	//	const int nthreads = C * output_H_ * output_W_;

	//	const Dtype* curr_U = U + i * (C * H * W);
	//	Dtype* curr_V = V + i * (C * output_H_ * output_W_);

	//	MornForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
	//	      CAFFE_CUDA_NUM_THREADS>>>(nthreads, i, C, output_H_, output_W_, H, W, coordinates, curr_U, curr_V);
	//}

	const Dtype* U = bottom[0]->gpu_data();
    const Dtype* offset_data = bottom[1]->gpu_data();
	Dtype* offset_data_trans = offsets_trans.mutable_gpu_data();
	Dtype* output_grid_data = output_grid.mutable_gpu_data();
	
	Dtype* input_grid_data = input_grid.mutable_gpu_data();
	Dtype* V = top[0]->mutable_gpu_data();

	caffe_gpu_set(input_grid.count(), (Dtype)0, input_grid_data);
	caffe_gpu_set(top[0]->count(), (Dtype)0, V);

	const int gpu_nthreads = N * output_H_ * output_W_;
	transform_offset<Dtype> << <CAFFE_GET_BLOCKS(gpu_nthreads),
		CAFFE_CUDA_NUM_THREADS >> >(gpu_nthreads, N, output_H_, output_W_, offset_data, offset_data_trans);
	
	for (int i = 0; i < N; ++i) {

		Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 2) * i;
		const Dtype* curr_offsets = offset_data_trans + (output_H_ * output_W_ * 2) * i;

		const int grid_count = output_H_ * output_W_ * 2;
		caffe_gpu_axpy<Dtype>(grid_count, 1.0, output_grid_data, coordinates);
		caffe_gpu_axpy<Dtype>(grid_count, 1.0, curr_offsets, coordinates);
	}

	const int nthreads = N * C * output_H_ * output_W_;

	MornForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
	      CAFFE_CUDA_NUM_THREADS>>>(nthreads, N, C, output_H_, output_W_, H, W, input_grid_data, U, V);

	//std::cout << "input_grid" << std::endl;
	//const Dtype* debug_input_grid = input_grid.cpu_data();
	//for (int i = 0; i<input_grid.count(); ++i) {
	//	std::cout << debug_input_grid[i] << " ";
	//}
	//std::cout << std::endl << std::endl;

	//std::cout << "vvvvvvvvvvv" << std::endl;
	//const Dtype* debug_V = top[0]->cpu_data();
	//for (int i = 0; i<top[0]->count(); ++i) {
	//	std::cout << debug_V[i] << " ";
	//}
	//std::cout << std::endl << std::endl;
}

template <typename Dtype>
__global__ void MornBackwardGPU_dOffset(const int nthreads, int C,
	int output_H_, int output_W_, int H, int W,
	const Dtype* input_grid_data, const Dtype* dV_array, const Dtype* U_array,
	Dtype* dOffset_diff) {

	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;

		const Dtype* coordinates = input_grid_data;

		const int row_idx = output_W_ * s + t;

		const Dtype px = coordinates[row_idx * 2];
		const Dtype py = coordinates[row_idx * 2 + 1];

		Dtype delta_dpx = (Dtype)0.;
		Dtype delta_dpy = (Dtype)0.;

		const Dtype x = (px + 1) / 2 * H;
		const Dtype y = (py + 1) / 2 * W;
		const int dV_offset = index;
		const Dtype dV = dV_array[dV_offset];

		int m, n;
		const Dtype* U = U_array;

		m = floor(x); n = floor(y); 
		if(m >= 0 && m < H && n >= 0 && n < W) {
			delta_dpx -= (1 - (y - n)) * U[m * W + n] * dV * H / 2;
			delta_dpy -= (1 - (x - m)) * U[m * W + n] * dV * W / 2;
		}
		
		m = floor(x); n = floor(y) + 1; 
		if(m >= 0 && m < H && n >= 0 && n < W) {
			delta_dpx -= (1 - (n - y)) * U[m * W + n] * dV * H / 2;
			delta_dpy += (1 - (x - m)) * U[m * W + n] * dV * W / 2;
		}

		m = floor(x) + 1; n = floor(y); 
		if(m >= 0 && m < H && n >= 0 && n < W) {
			delta_dpx += (1 - (y - n)) * U[m * W + n] * dV * H / 2;
			delta_dpy -= (1 - (m - x)) * U[m * W + n] * dV * W / 2;
		}
		
		m = floor(x) + 1; n = floor(y) + 1; 
		if(m >= 0 && m < H && n >= 0 && n < W) {
			delta_dpx += (1 - (n - y)) * U[m * W + n] * dV * H / 2;
			delta_dpy += (1 - (m - x)) * U[m * W + n] * dV * W / 2;
		}

		int idx = s * output_W_ + t;

		dOffset_diff[idx] += (delta_dpx * 2.0 / output_H_);
		dOffset_diff[output_H_ * output_W_ + idx] += (delta_dpy * 2.0 / output_W_);
	}
}


template <typename Dtype>
__global__ void MornBackwardGPU_dU(const int nthreads, const int C, 
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

	  	const Dtype x = (px + 1) / 2 * H;
	  	const Dtype y = (py + 1) / 2 * W;

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
__global__ void MornBackward(const int n, const Dtype* in_diff, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index];
  }
}

template <typename Dtype>
void MornLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	string prefix = "MornLayer::Backward_GPU::\t";

	Dtype* dV = top[0]->mutable_gpu_diff();
	const Dtype* input_grid_data = input_grid.gpu_data();
	const Dtype* U = bottom[0]->gpu_data();

	Dtype* dOffset_diff = bottom[1]->mutable_gpu_diff();

	caffe_gpu_set(bottom[1]->count(), (Dtype)0, dOffset_diff);

	// for each input
	for (int i = 0; i < N; ++i) {

		const Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 2) * i;
		Dtype* curr_dOffset_diff = dOffset_diff + (output_H_ * output_W_ * 2) * i;

		for (int j = 0; j < C; ++j) {

			const int nthreads = output_H_ * output_W_;

			const Dtype* curr_dV = dV + i * (C * output_H_ * output_W_) + j * (output_H_ * output_W_);
			const Dtype* curr_U = U + i * (C * H * W) + j * (H * W);

			MornBackwardGPU_dOffset<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
					CAFFE_CUDA_NUM_THREADS>>>(nthreads, C, output_H_, output_W_, H, W, coordinates,
						curr_dV, curr_U, curr_dOffset_diff);
		}
	}

	//const int nthreads = N * C * output_H_ * output_W_;
	// bug bug
	//MornBackwardGPU_dOffset<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
	//		CAFFE_CUDA_NUM_THREADS>>>(nthreads, C, output_H_, output_W_, H, W, input_grid_data,
	//				dV, U, dOffset_diff);

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

	//std::cout << "U U U" << std::endl;
	//const Dtype* db_U = bottom[0]->cpu_data();
	//for (int i = 0; i<bottom[0]->count(); ++i) {
	//	std::cout << db_U[i] << " ";
	//}
	//std::cout << std::endl << std::endl;

	//Dtype* break_ptr = 0;
	//*break_ptr = 1;
	
	if(to_compute_dU_) {
		Dtype* dU = bottom[0]->mutable_gpu_diff();
		caffe_gpu_set(bottom[0]->count(), (Dtype)0., dU);
		const int nthreads = N * C * output_H_ * output_W_;
		MornBackwardGPU_dU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
			CAFFE_CUDA_NUM_THREADS>>>(nthreads, C, W, H, output_H_, output_W_, input_grid_data, dV, dU);

		//const int count = bottom[0]->count();
		//// NOLINT_NEXT_LINE(whitespace/operators)
		//MornBackward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
		//	count, dV, dU);
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(MornLayer);

}	// namespace caffe
