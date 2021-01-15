#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/st_pt_layer.hpp"
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
__global__ void SpatialTransformerPTForwardGPU(const int nthreads, int N, int C,
	int output_H_, int output_W_, int H, int W,
	const Dtype* input_grid_data, const Dtype* U, Dtype* V) {
	
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int j = (index / (output_W_ * output_H_)) % C;
		const int i = index / (output_W_ * output_H_ * C);

		const Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 3) * i;
		const int row_idx = output_W_ * s + t;

		const Dtype pw = coordinates[row_idx * 3 + 2];
	  	const Dtype px = coordinates[row_idx * 3] / pw;
	  	const Dtype py = coordinates[row_idx * 3 + 1] / pw;

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

		//m = floor(x); n = floor(y); w = (Dtype)0;
		//if (m >= 0 && m < H && n >= 0 && n < W) {
		//	w = max((Dtype)0, (Dtype)1 - abs(x - m)) * max((Dtype)0, (Dtype)1 - abs(y - n));
		//	V[V_offset] += w * pic[m * W + n];
		//}

		//m = floor(x) + 1; n = floor(y); w = (Dtype)0;
		//if (m >= 0 && m < H && n >= 0 && n < W) {
		//	w = max((Dtype)0, (Dtype)1 - abs(x - m)) * max((Dtype)0, (Dtype)1 - abs(y - n));
		//	V[V_offset] += w * pic[m * W + n];
		//}

		//m = floor(x); n = floor(y) + 1; w = (Dtype)0;
		//if (m >= 0 && m < H && n >= 0 && n < W) {
		//	w = max((Dtype)0, (Dtype)1 - abs(x - m)) * max((Dtype)0, (Dtype)1 - abs(y - n));
		//	V[V_offset] += w * pic[m * W + n];
		//}

		//m = floor(x) + 1; n = floor(y) + 1; w = (Dtype)0;
		//if (m >= 0 && m < H && n >= 0 && n < W) {
		//	w = max((Dtype)0, (Dtype)1 - abs(x - m)) * max((Dtype)0, (Dtype)1 - abs(y - n));
		//	V[V_offset] += w * pic[m * W + n];
		//}
  }
}

template <typename Dtype>
__global__ void overflow_test(const int nthreads, int N,
	int output_H_, int output_W_, Dtype* input_grid_data) {

	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int i = index / (output_W_ * output_H_);

		//Dtype pw = input_grid_data[3 * index + 2];
		//if (pw < 0.000001 && pw > -0.000001) {
		//	if (pw > 0) {
		//		input_grid_data[3 * index + 2] = 0.0001;
		//	}
		//	else {
		//		input_grid_data[3 * index + 2] = -0.0001;
		//	}
		//}

		Dtype pw = input_grid_data[index * 3 + 2];
		input_grid_data[index * 3] = input_grid_data[index * 3] / pw;
		input_grid_data[index * 3 + 1] = input_grid_data[index * 3 + 1] / pw;
	}
}

template <typename Dtype>
void SpatialTransformerPTLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	string prefix = "SpatialTransformerPTLayer::Forward_gpu::\t";
	//Forward_cpu(bottom, top);

	const Dtype* U = bottom[0]->gpu_data();
	const Dtype* theta = bottom[1]->gpu_data();
	Dtype* output_grid_data = output_grid.mutable_gpu_data();

	//std::cout << "output_grid data sync end " << std::endl;
	
	Dtype* full_theta_data = full_theta.mutable_gpu_data();
	Dtype* input_grid_data = input_grid.mutable_gpu_data();
	Dtype* V = top[0]->mutable_gpu_data();

	caffe_gpu_set(input_grid.count(), (Dtype)0, input_grid_data);
	caffe_gpu_set(top[0]->count(), (Dtype)0, V);
	
	// compute full_theta
	int k = 0; 
	const int num_threads = N;
	for(int i=0; i<9; ++i) {
		if (is_pre_defined_theta[i]) {
			set_value_to_constant<Dtype> << <CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS >> >(
				num_threads, pre_defined_theta[i], 9, i, full_theta_data);
			//std::cout << "Setting value " << pre_defined_theta[i] << " to "<< i << 
			//	"/9 of full_theta_data" << std::endl;
		}
		else {
			copy_values<Dtype> << <CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS >> >(num_threads,
				9 - pre_defined_count, k, theta, 9, i, full_theta_data);
			//std::cout << "Copying " << k << "/" << 9 - pre_defined_count << " of theta to " 
			//	<< i << "/9 of full_theta_data" << std::endl;
			++k;
		}
	}

	// compute out input_grid_data
	for(int i = 0; i < N; ++i) {
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, output_H_ * output_W_, 3, 3, (Dtype)1.,
				output_grid_data, full_theta_data + 9 * i, (Dtype)0.,
				input_grid_data + (output_H_ * output_W_ * 3) * i);
	}

	test_defined_count = test_defined_count + 1;
	////if (test_defined_count == 1000) 
	//{
	//	std::cout << "dw dw dw" << std::endl;
	//	const Dtype* full_theta_test = full_theta.cpu_data();
	//	for (int index = 0; index < full_theta.count(); ++index) {
	//		Dtype theta = full_theta_test[index];
	//		std::cout << theta << " ";
	//	}
	//	std::cout << std::endl << std::endl;
	//}

	//******be care overfitting.******** no bug
	//const int gpu_nthreads = N * output_H_ * output_W_;
	//overflow_test<Dtype> << <CAFFE_GET_BLOCKS(gpu_nthreads),
	//	CAFFE_CUDA_NUM_THREADS >> >(gpu_nthreads, N, output_H_, output_W_, input_grid_data);

	//if (test_defined_count == 1)
	//{
	//	std::cout << "dw dw dw" << std::endl;
	//	const Dtype* input_grid_data_test = input_grid.cpu_data();
	//	for (int index = 0; index < output_H_ * output_W_; ++index) {
	//		Dtype pw = input_grid_data_test[3 * index + 2];
	//		std::cout << pw << " ";
	//	}
	//	std::cout << std::endl << std::endl;

	//	std::cout << "dx dx dx" << std::endl;
	//	for (int index = 0; index < output_H_ * output_W_; ++index) {
	//		Dtype pw = input_grid_data_test[3 * index];
	//		std::cout << pw << " ";
	//	}
	//	std::cout << std::endl << std::endl;

	//	std::cout << "dy dy dy" << std::endl;
	//	for (int index = 0; index < output_H_ * output_W_; ++index) {
	//		Dtype pw = input_grid_data_test[3 * index + 1];
	//		std::cout << pw << " ";
	//	}
	//	std::cout << std::endl << std::endl;

	//	//Dtype* break_ptr = 0;
	//	//*break_ptr = 1;
	//}

	//std::cout << output_H_ << " " << output_W_ << " " << H << " " << W << " " << N << " " << C << " ";
#if 1
	const int nthreads = N * C * output_H_ * output_W_;
	SpatialTransformerPTForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
	      CAFFE_CUDA_NUM_THREADS>>>(nthreads, N, C, output_H_, output_W_, H, W, input_grid_data, U, V);
#endif

#if 0
	const Dtype* input_grid_cpu_data = input_grid.cpu_data();
	const Dtype* cpu_U = bottom[0]->cpu_data();
	Dtype* cpu_V = top[0]->mutable_cpu_data();

	for (int i = 0; i < N; ++i) {

		const Dtype* coordinates = input_grid_cpu_data + (output_H_ * output_W_ * 3) * i;

		int row_idx; Dtype px, py;

		for (int j = 0; j < C; ++j)
			for (int s = 0; s < output_H_; ++s)
				for (int t = 0; t < output_W_; ++t) {

					row_idx = output_W_ * s + t;

					px = coordinates[row_idx * 3] / coordinates[row_idx * 3 + 2];
					py = coordinates[row_idx * 3 + 1] / coordinates[row_idx * 3 + 2];

					cpu_V[top[0]->offset(i, j, s, t)] = transform_forward_cpu(
						cpu_U + bottom[0]->offset(i, j, 0, 0), px, py);
				}
	}
#endif
}

template <typename Dtype>
__global__ void SpatialTransformerPTBackwardGPU_dTheta(const int nthreads, int C,
		int output_H_, int output_W_, int H, int W,
		const Dtype* input_grid_data, const Dtype* dV_array, const Dtype* U_array,  
		Dtype* dTheta_tmp_diff) {
	
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int j = (index / (output_W_ * output_H_)) % C;
		const int i = index / (output_W_ * output_H_ * C);

		const Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 3) * i;

		const int row_idx = output_W_ * s + t;

		const Dtype pw = coordinates[row_idx * 3 + 2];
		const Dtype px = coordinates[row_idx * 3] / pw;
		const Dtype py = coordinates[row_idx * 3 + 1] / pw;
		
		Dtype delta_dpx = (Dtype)0.;
		Dtype delta_dpy = (Dtype)0.;
		Dtype delta_dpw = (Dtype)0.;

		const Dtype x = (px + 1) / 2 * H;
		const Dtype y = (py + 1) / 2 * W;
		const int dV_offset = index;
		const Dtype dV = dV_array[dV_offset];

		int m, n; 
		const Dtype* U = U_array + i * (C * H * W) + j * (H * W);

		// left-bottom neighbor
		m = floor(x); n = floor(y); 
		if(m >= 0 && m < H && n >= 0 && n < W) {
			delta_dpx -= (1 - (y - n)) * U[m * W + n] * dV * H / 2;
			delta_dpy -= (1 - (x - m)) * U[m * W + n] * dV * W / 2;
		}
		
		// left-top neighbor
		m = floor(x); n = floor(y) + 1; 
		if(m >= 0 && m < H && n >= 0 && n < W) {
			delta_dpx -= (1 - (n - y)) * U[m * W + n] * dV * H / 2;
			delta_dpy += (1 - (x - m)) * U[m * W + n] * dV * W / 2;
		}

		// right-bottom neighbor
		m = floor(x) + 1; n = floor(y); 
		if(m >= 0 && m < H && n >= 0 && n < W) {
			delta_dpx += (1 - (y - n)) * U[m * W + n] * dV * H / 2;
			delta_dpy -= (1 - (m - x)) * U[m * W + n] * dV * W / 2;
		}
		
		// right-top neighbor
		m = floor(x) + 1; n = floor(y) + 1; 
		if(m >= 0 && m < H && n >= 0 && n < W) {
			delta_dpx += (1 - (n - y)) * U[m * W + n] * dV * H / 2;
			delta_dpy += (1 - (m - x)) * U[m * W + n] * dV * W / 2;
		}
		
		delta_dpw = delta_dpx*(-coordinates[row_idx * 3])/ (pw*pw) + delta_dpy*(-coordinates[row_idx * 3+1]) / (pw*pw);
		//******be care overfitting.********
		//delta_dpw = delta_dpx*(-px)/pw + delta_dpy*(-py) / pw;
		
		int idx = j * (output_H_ * output_W_) + s * output_W_ + t;
		
		dTheta_tmp_diff[(9 * i) * (output_H_ * output_W_ * C) + idx] += delta_dpx * (s * 1.0 / output_H_ * 2 - 1) / pw;
		dTheta_tmp_diff[(9 * i + 1) * (output_H_ * output_W_ * C) + idx] += delta_dpx * (t * 1.0 / output_W_ * 2 - 1) / pw;
		dTheta_tmp_diff[(9 * i + 2) * (output_H_ * output_W_ * C) + idx] += delta_dpx / pw;
		dTheta_tmp_diff[(9 * i + 3) * (output_H_ * output_W_ * C) + idx] += delta_dpy * (s * 1.0 / output_H_ * 2 - 1) / pw;
		dTheta_tmp_diff[(9 * i + 4) * (output_H_ * output_W_ * C) + idx] += delta_dpy * (t * 1.0 / output_W_ * 2 - 1) / pw;
		dTheta_tmp_diff[(9 * i + 5) * (output_H_ * output_W_ * C) + idx] += delta_dpy / pw;
		dTheta_tmp_diff[(9 * i + 6) * (output_H_ * output_W_ * C) + idx] += delta_dpw * (s * 1.0 / output_H_ * 2 - 1);
		dTheta_tmp_diff[(9 * i + 7) * (output_H_ * output_W_ * C) + idx] += delta_dpw * (t * 1.0 / output_W_ * 2 - 1);
		dTheta_tmp_diff[(9 * i + 8) * (output_H_ * output_W_ * C) + idx] += delta_dpw;
	}
}

template <typename Dtype>
__global__ void SpatialTransformerPTBackwardGPU_dU(const int nthreads, const int C, 
	const int W,  const int H, const int output_H_, const int output_W_, 
	const Dtype* input_grid_data, const Dtype* dV, Dtype* dU) {
	
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int j = (index / (output_W_ * output_H_)) % C;
		const int i = index / (output_W_ * output_H_ * C);

		const Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 3) * i;
		const int row_idx = output_W_ * s + t;

		const Dtype pw = coordinates[row_idx * 3 + 2];
		const Dtype px = coordinates[row_idx * 3] / pw;
		const Dtype py = coordinates[row_idx * 3 + 1] / pw;

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
void SpatialTransformerPTLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	string prefix = "SpatialTransformerPTLayer::Backward_GPU::\t";

	const Dtype* dV = top[0]->gpu_diff();
	const Dtype* input_grid_data = input_grid.gpu_data();
	const Dtype* U = bottom[0]->gpu_data();

	Dtype* dFull_theta = full_theta.mutable_gpu_diff();
	Dtype* dTheta = bottom[1]->mutable_gpu_diff();
	Dtype* dTheta_tmp_diff = dTheta_tmp.mutable_gpu_diff();

	caffe_gpu_set(dTheta_tmp.count(), (Dtype)0., dTheta_tmp_diff);

	const int nthreads = N * C * output_H_ * output_W_;

	SpatialTransformerPTBackwardGPU_dTheta<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
			CAFFE_CUDA_NUM_THREADS>>>(nthreads, C, output_H_, output_W_, H, W, input_grid_data,
					dV, U, dTheta_tmp_diff);

	Dtype* all_ones_2_data = all_ones_2.mutable_gpu_data();
	caffe_gpu_set(all_ones_2.count(), (Dtype)1., all_ones_2_data);
	
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, full_theta.count(), 1, output_H_ * output_W_ * C, 
			(Dtype)1., dTheta_tmp_diff, all_ones_2_data, (Dtype)0., dFull_theta);
			
	/*const Dtype* db_dFull_theta = full_theta.cpu_diff();
	for(int i=0; i<full_theta.count(); ++i) {
		std::cout << db_dFull_theta[i] << " ";
	}
	std::cout<<std::endl;*/
			
	int k = 0;
	const int num_threads = N;
	for(int i=0; i<9; ++i) {
		if (!is_pre_defined_theta[i]) {
			copy_values<Dtype> << <CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS >> >(num_threads,
				9, i, dFull_theta, 9 - pre_defined_count, k, dTheta);
			//std::cout << "Copying " << i << "/9 of dFull_theta to " << k << "/" << 
			//	9 - pre_defined_count << " of dTheta" << std::endl;
			++k;
		}
	}
	
	/*const Dtype* db_dtheta = bottom[1]->cpu_diff();
	for(int i=0; i<bottom[1]->count(); ++i) {
		std::cout << db_dtheta[i] << " ";
	}
	std::cout<<std::endl;*/
			
	if(to_compute_dU_) {
		Dtype* dU = bottom[0]->mutable_gpu_diff();
		caffe_gpu_set(bottom[0]->count(), (Dtype)0., dU);
		const int nthreads = N * C * output_H_ * output_W_;
		SpatialTransformerPTBackwardGPU_dU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
			CAFFE_CUDA_NUM_THREADS>>>(nthreads, C, W, H, output_H_, output_W_, input_grid_data, dV, dU);
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(SpatialTransformerPTLayer);

}	// namespace caffe
