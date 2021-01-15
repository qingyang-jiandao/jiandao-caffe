#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/st_pt_op_layer.hpp"
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
__global__ void SpatialTransformerForwardGPU(const int nthreads, int N, int C,
	int output_H_, int output_W_, int H, int W,
	const Dtype* input_grid_data, const Dtype* all_theta, const Dtype* U, Dtype* V) {

	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int j = (index / (output_W_ * output_H_)) % C;
		const int i = index / (output_W_ * output_H_ * C);

		const Dtype centerpoint_offset_x = all_theta[11 * i + 9];
		const Dtype centerpoint_offset_y = all_theta[11 * i + 10];

		const Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 3) * i;
		const int row_idx = output_W_ * s + t;

		const Dtype pw = coordinates[row_idx * 3 + 2];
		const Dtype px = coordinates[row_idx * 3] / pw;
		const Dtype py = coordinates[row_idx * 3 + 1] / pw;

		const int V_offset = index;

		V[V_offset] = (Dtype)0.;

		const Dtype x = (px + centerpoint_offset_x) / 2 * H;
		const Dtype y = (py + centerpoint_offset_y) / 2 * W;

		int m, n; Dtype w;
		const Dtype* pic = U + i * (C * H * W) + j * (H * W);

		m = floor(x); n = floor(y); w = 0;
		if (m >= 0 && m < H && n >= 0 && n < W) {
			w = (1 - (x - m)) * (1 - (y - n));
			V[V_offset] += w * pic[m * W + n];
		}

		m = floor(x) + 1; n = floor(y); w = 0;
		if (m >= 0 && m < H && n >= 0 && n < W) {
			w = (1 - (m - x)) * (1 - (y - n));
			V[V_offset] += w * pic[m * W + n];
		}

		m = floor(x); n = floor(y) + 1; w = 0;
		if (m >= 0 && m < H && n >= 0 && n < W) {
			w = (1 - (x - m)) * (1 - (n - y));
			V[V_offset] += w * pic[m * W + n];
		}

		m = floor(x) + 1; n = floor(y) + 1; w = 0;
		if (m >= 0 && m < H && n >= 0 && n < W) {
			w = (1 - (m - x)) * (1 - (n - y));
			V[V_offset] += w * pic[m * W + n];
		}
	}
}

template <typename Dtype>
__global__ void overflow_test(const int nthreads, int N,
	int output_H_, int output_W_, Dtype* output_grid_data) {

	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int i = index / (output_W_ * output_H_);

		Dtype pw = output_grid_data[3 * index + 2];
		if (pw < 0.000001 && pw > -0.000001) {
			if (pw > 0) {
				output_grid_data[3 * index + 2] = 0.0001;
			}
			else {
				output_grid_data[3 * index + 2] = -0.0001;
			}
		}
	}
}

template <typename Dtype>
__global__ void move_grid(const int nthreads, int N,
	int output_H_, int output_W_, const Dtype* all_theta, Dtype* inner_grid_data) {

	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int i = index / (output_W_ * output_H_);

		const Dtype centerpoint_offset_x = all_theta[11 * i + 9];
		const Dtype centerpoint_offset_y = all_theta[11 * i + 10];

		const int row_idx = output_W_ * s + t;
		//Dtype* curr_inner_grid_data = inner_grid_data + (output_H_ * output_W_ * 3) * i;
		//curr_inner_grid_data[row_idx] = s * 1.0 / output_H_ * 2 - centerpoint_offset_x;
		//curr_inner_grid_data[row_idx+1] = t * 1.0 / output_W_ * 2 - centerpoint_offset_y;
		//curr_inner_grid_data[row_idx + 2] = 1;

		inner_grid_data[3 * index] = s * 1.0 / output_H_ * 2 - centerpoint_offset_x;
		inner_grid_data[3 * index + 1] = t * 1.0 / output_W_ * 2 - centerpoint_offset_y;
		inner_grid_data[3 * index + 2] = 1;
	}
}

template <typename Dtype>
void SpatialTransformerPTOPLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	string prefix = "SpatialTransformerPTOPLayer::Forward_gpu::\t";

	const Dtype* U = bottom[0]->gpu_data();
	const Dtype* theta = bottom[1]->gpu_data();
	const Dtype* output_grid_data = output_grid.mutable_gpu_data();
	const int thete_count = bottom[1]->shape(1);

	//std::cout << "output_grid data sync end " << std::endl;
	
	Dtype* full_theta_data = full_theta.mutable_gpu_data();
	Dtype* input_grid_data = input_grid.mutable_gpu_data();
	Dtype* inner_grid_data = inner_grid.mutable_gpu_data();
	Dtype* V = top[0]->mutable_gpu_data();

	caffe_gpu_set(input_grid.count(), (Dtype)0, input_grid_data);
	caffe_gpu_set(top[0]->count(), (Dtype)0, V);
	
	// compute full_theta
	int k = 0; 
	const int num_threads = N;
	for(int i=0; i<9; ++i) {
		if (is_pre_defined_theta[i]) {
			set_value_to_constant<Dtype> << <CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS >> > (
				num_threads, pre_defined_theta[i], 9, i, full_theta_data);
			//std::cout << "Setting value " << pre_defined_theta[i] << " to "<< i << 
			//	"/9 of full_theta_data" << std::endl;
		}
		else {
			copy_values<Dtype> << <CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS >> > (num_threads,
				11 - pre_defined_count, k, theta, 9, i, full_theta_data);
			//std::cout << "Copying " << k << "/" << 9 - pre_defined_count << " of theta to " 
			//	<< i << "/9 of full_theta_data" << std::endl;
			++k;
		}
	}

	const int gpu_nthreads = N * output_H_ * output_W_;
	move_grid<Dtype> << <CAFFE_GET_BLOCKS(gpu_nthreads),
		CAFFE_CUDA_NUM_THREADS >> >(gpu_nthreads, N, output_H_, output_W_, theta, inner_grid_data);

	// compute out input_grid_data
	for(int i = 0; i < N; ++i) {
		Dtype* curr_inner_grid_data = inner_grid_data + (output_H_ * output_W_ * 3) * i;
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, output_H_ * output_W_, 3, 3, (Dtype)1.,
			curr_inner_grid_data, full_theta_data + 9 * i, (Dtype)0.,
				input_grid_data + (output_H_ * output_W_ * 3) * i);
	}

	//for(int i = 0; i < N; ++i) {
	//	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, output_H_ * output_W_, 3, 3, (Dtype)1.,
	//			output_grid_data, full_theta_data + 9 * i, (Dtype)0.,
	//			input_grid_data + (output_H_ * output_W_ * 3) * i);
	//}

	//test_defined_count = test_defined_count + 1;
	//if (test_defined_count == 1000) {
	//	std::cout << "dw dw dw" << std::endl;
	//	const Dtype* input_grid_data_test = input_grid.cpu_data();
	//	for (int index = 0; index < output_H_ * output_W_; ++index) {
	//		Dtype pw = input_grid_data_test[3 * index + 2];
	//		std::cout << pw << " ";
	//	}
	//	std::cout << std::endl << std::endl;
	//}

	//******be care overfitting.********
	//const int gpu_nthreads = N * output_H_ * output_W_;
	//overflow_test<Dtype> << <CAFFE_GET_BLOCKS(gpu_nthreads),
	//	CAFFE_CUDA_NUM_THREADS >> >(gpu_nthreads, N, output_H_, output_W_, input_grid_data);

	//if (test_defined_count == 1000) {
	//	std::cout << "dw dw dw" << std::endl;
	//	const Dtype* input_grid_data_test = input_grid.cpu_data();
	//	for (int index = 0; index < output_H_ * output_W_; ++index) {
	//		Dtype pw = input_grid_data_test[3 * index + 2];
	//		std::cout << pw << " ";
	//	}
	//	std::cout << std::endl << std::endl;

	//	Dtype* break_ptr = 0;
	//	*break_ptr = 1;
	//}

	const int nthreads = N * C * output_H_ * output_W_;

	SpatialTransformerForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
	      CAFFE_CUDA_NUM_THREADS>>>(nthreads, N, C, output_H_, output_W_, H, W, input_grid_data, theta, U, V);
}

template <typename Dtype>
__global__ void SpatialTransformerBackwardGPU_dTheta(const int nthreads, int C,
		int output_H_, int output_W_, int H, int W,
		const Dtype* input_grid_data, const Dtype* all_theta, const Dtype* inner_grid_data, const Dtype* dV_array, const Dtype* U_array,
		Dtype* dTheta_tmp_diff, Dtype* dTheta_tmp_point_xy_diff) {
	
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int j = (index / (output_W_ * output_H_)) % C;
		const int i = index / (output_W_ * output_H_ * C);

		const Dtype* i_offsets = all_theta + 11 * i;
		const Dtype centerpoint_offset_x = i_offsets[9];
		const Dtype centerpoint_offset_y = i_offsets[10];

		const int row_idx = output_W_ * s + t;
		const Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 3) * i;
		const Dtype pw = coordinates[row_idx * 3 + 2];
		const Dtype px = coordinates[row_idx * 3] / pw;
		const Dtype py = coordinates[row_idx * 3 + 1] / pw;

		const Dtype* inner_grid_coordinates = inner_grid_data + (output_H_ * output_W_ * 3) * i;
		const Dtype inner_px = inner_grid_coordinates[row_idx * 3];
		const Dtype inner_py = inner_grid_coordinates[row_idx * 3 + 1];
		
		Dtype delta_dpx = (Dtype)0.;
		Dtype delta_dpy = (Dtype)0.;
		Dtype delta_dpw = (Dtype)0.;

		const Dtype x = (px + centerpoint_offset_x) / 2 * H;
		const Dtype y = (py + centerpoint_offset_y) / 2 * W;
		const int dV_offset = index;
		const Dtype dV = dV_array[dV_offset];

		int m, n; 
		const Dtype* U = U_array + i * (C * H * W) + j * (H * W);

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
		
		delta_dpw = delta_dpx*(-coordinates[row_idx * 3])/ (pw*pw) + delta_dpy*(-coordinates[row_idx * 3+1]) / (pw*pw);
		//******be care overfitting.********
		//delta_dpw = delta_dpx*(-px)/pw + delta_dpy*(-py) / pw;

		int delta_centerpoint_offset_x = delta_dpx + delta_dpx*(-pw*i_offsets[0] / (pw*pw) + coordinates[row_idx * 3] * i_offsets[6] / (pw*pw)) + delta_dpy*(-pw*i_offsets[3] / (pw*pw) + coordinates[row_idx * 3 + 1] * i_offsets[6] / (pw*pw));
		int delta_centerpoint_offset_y = delta_dpy + delta_dpx*(-pw*i_offsets[1] / (pw*pw) + coordinates[row_idx * 3] * i_offsets[7] / (pw*pw)) + delta_dpy*(-pw*i_offsets[4] / (pw*pw) + coordinates[row_idx * 3 + 1] * i_offsets[7] / (pw*pw));

		int idx = j * (output_H_ * output_W_) + s * output_W_ + t;
		
		dTheta_tmp_diff[(9 * i) * (output_H_ * output_W_ * C) + idx] += delta_dpx * inner_px / pw;
		dTheta_tmp_diff[(9 * i + 1) * (output_H_ * output_W_ * C) + idx] += delta_dpx * inner_py / pw;
		dTheta_tmp_diff[(9 * i + 2) * (output_H_ * output_W_ * C) + idx] += delta_dpx / pw;
		dTheta_tmp_diff[(9 * i + 3) * (output_H_ * output_W_ * C) + idx] += delta_dpy * inner_px / pw;
		dTheta_tmp_diff[(9 * i + 4) * (output_H_ * output_W_ * C) + idx] += delta_dpy * inner_py / pw;
		dTheta_tmp_diff[(9 * i + 5) * (output_H_ * output_W_ * C) + idx] += delta_dpy / pw;
		dTheta_tmp_diff[(9 * i + 6) * (output_H_ * output_W_ * C) + idx] += delta_dpw * inner_px;
		dTheta_tmp_diff[(9 * i + 7) * (output_H_ * output_W_ * C) + idx] += delta_dpw * inner_py;
		dTheta_tmp_diff[(9 * i + 8) * (output_H_ * output_W_ * C) + idx] += delta_dpw;

		dTheta_tmp_point_xy_diff[(2 * i) * (output_H_ * output_W_ * C) + idx] += delta_centerpoint_offset_x;
		dTheta_tmp_point_xy_diff[(2 * i + 1) * (output_H_ * output_W_ * C) + idx] += delta_centerpoint_offset_y;
	}
}

template <typename Dtype>
__global__ void SpatialTransformerBackwardGPU_dU(const int nthreads, const int C, 
	const int W,  const int H, const int output_H_, const int output_W_, 
	const Dtype* input_grid_data, const Dtype* all_theta, const Dtype* dV, Dtype* dU) {
	
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int j = (index / (output_W_ * output_H_)) % C;
		const int i = index / (output_W_ * output_H_ * C);

		const Dtype* i_offsets = all_theta + 11 * i;
		const Dtype centerpoint_offset_x = i_offsets[9];
		const Dtype centerpoint_offset_y = i_offsets[10];

		const Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 3) * i;
		const int row_idx = output_W_ * s + t;

		const Dtype pw = coordinates[row_idx * 3 + 2];
		const Dtype px = coordinates[row_idx * 3] / pw;
		const Dtype py = coordinates[row_idx * 3 + 1] / pw;

	  	const int V_offset = index;

	  	const Dtype x = (px + centerpoint_offset_x) / 2 * H;
	  	const Dtype y = (py + centerpoint_offset_y) / 2 * W;

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
__global__ void centerpoint_offset_loss(const int nthreads, int N,
	int output_H_, int output_W_, const Dtype* theta, Dtype* loss_array) {

	CUDA_KERNEL_LOOP(index, nthreads) {

		const Dtype centerpoint_offset_x = theta[11 * index + 9];
		const Dtype centerpoint_offset_y = theta[11 * index + 10];

		Dtype d_x = (Dtype)0, d_y = (Dtype)0;

		if (centerpoint_offset_x < 0) {
			d_x = (centerpoint_offset_x);
		}
		else if (centerpoint_offset_x > 2) {
			d_x = (centerpoint_offset_x - 2);
		}

		if (centerpoint_offset_y < 0) {
			d_y = (centerpoint_offset_y);
		}
		else if (centerpoint_offset_y > 2) {
			d_y = (centerpoint_offset_y - 2);
		}

		loss_array[11 * index + 9] = d_x;
		loss_array[11 * index + 10] = d_y;
	}
}

template <typename Dtype>
void SpatialTransformerPTOPLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	string prefix = "SpatialTransformerPTOPLayer::Backward_GPU::\t";

	const Dtype* dV = top[0]->gpu_diff();
	const Dtype* input_grid_data = input_grid.gpu_data();
	const Dtype* inner_grid_data = inner_grid.gpu_data();
	const Dtype* U = bottom[0]->gpu_data();
	const Dtype* theta = bottom[1]->gpu_data();
	const int thete_count = bottom[1]->shape(1);

	Dtype* dFull_theta_diff = full_theta.mutable_gpu_diff();
	Dtype* dFull_theta_point_xy_diff = full_theta_point_xy.mutable_gpu_diff();
	Dtype* dTheta = bottom[1]->mutable_gpu_diff();
	Dtype* dTheta_tmp_diff = dTheta_tmp.mutable_gpu_diff();
	Dtype* dTheta_tmp_point_xy_diff = dTheta_tmp_point_xy.mutable_gpu_diff();

	caffe_gpu_set(dTheta_tmp.count(), (Dtype)0., dTheta_tmp_diff);
	caffe_gpu_set(dTheta_tmp_point_xy.count(), (Dtype)0., dTheta_tmp_point_xy_diff);

	const int nthreads = N * C * output_H_ * output_W_;
	SpatialTransformerBackwardGPU_dTheta<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
			CAFFE_CUDA_NUM_THREADS>>>(nthreads, C, output_H_, output_W_, H, W, input_grid_data, theta, inner_grid_data,
					dV, U, dTheta_tmp_diff, dTheta_tmp_point_xy_diff);

	Dtype* all_ones_2_data = all_ones_2.mutable_gpu_data();
	caffe_gpu_set(all_ones_2.count(), (Dtype)1., all_ones_2_data);
	
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, full_theta.count(), 1, output_H_ * output_W_ * C,
			(Dtype)1., dTheta_tmp_diff, all_ones_2_data, (Dtype)0., dFull_theta_diff);

	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, full_theta_point_xy.count(), 1, output_H_ * output_W_ * C,
		(Dtype)1., dTheta_tmp_point_xy_diff, all_ones_2_data, (Dtype)0., dFull_theta_point_xy_diff);
			
	/*const Dtype* db_dFull_theta = full_theta.cpu_diff();
	for(int i=0; i<full_theta.count(); ++i) {
		std::cout << db_dFull_theta[i] << " ";
	}
	std::cout<<std::endl;*/
			
	int k = 0;
	const int num_threads = N;
	for (int i = 0; i < 11; ++i) {
		if (!is_pre_defined_theta[i] && i<9) {
			copy_values<Dtype> << <CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS >> > (num_threads,
				9, i, dFull_theta_diff, 11 - pre_defined_count, k, dTheta);
			//std::cout << "Copying " << i << "/9 of dFull_theta to " << k << "/" << 
			//	9 - pre_defined_count << " of dTheta" << std::endl;
			++k;
		}
		else {
			if (!is_pre_defined_theta[i]) {
				copy_values<Dtype> << <CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS >> > (num_threads,
					2, i-9, dFull_theta_point_xy_diff, 11 - pre_defined_count, k, dTheta);
				//std::cout << "Copying " << i << "/9 of dFull_theta to " << k << "/" << 
				//	9 - pre_defined_count << " of dTheta" << std::endl;
				++k;
			}
		}
	}
	
	//centerpoint_offset_loss<Dtype> << <CAFFE_GET_BLOCKS(num_threads),
	//	CAFFE_CUDA_NUM_THREADS >> >(num_threads, N, output_H_, output_W_, theta, dTheta);
	
	/*const Dtype* db_dtheta = bottom[1]->cpu_diff();
	for(int i=0; i<bottom[1]->count(); ++i) {
		std::cout << db_dtheta[i] << " ";
	}
	std::cout<<std::endl;*/
			
	if(to_compute_dU_) {
		Dtype* dU = bottom[0]->mutable_gpu_diff();
		caffe_gpu_set(bottom[0]->count(), (Dtype)0., dU);
		const int nthreads = N * C * output_H_ * output_W_;
		SpatialTransformerBackwardGPU_dU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
			CAFFE_CUDA_NUM_THREADS>>>(nthreads, C, W, H, output_H_, output_W_, input_grid_data, theta, dV, dU);
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(SpatialTransformerPTOPLayer);

}	// namespace caffe
