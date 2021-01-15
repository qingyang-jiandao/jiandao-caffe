#include <vector>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/layers/tps_c2t.hpp"
#include "caffe/util/benchmark.hpp"
//#include "caffe/layers/TPSInterpolate.hpp"

namespace caffe{

template <typename Dtype>
__global__ void set_value_to_constant(const int nthreads, int N, int src_size, const Dtype* src, Dtype* dst) {

	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % src_size;
		const int i = index / src_size;

		int row_index = (src_size+3) * i + t;
		dst[row_index] = src[index];
	}
}

template <typename Dtype>
void CToParaLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
	string prefix = "CToParaLayer::Forward_gpu::\t";
	//std::cout << prefix << "Starting!" << std::endl;
	//test_defined_count++;
	//Forward_cpu(bottom, top);

	
    const Dtype* ctl_points = bottom[0]->gpu_data();        
    Dtype* T = top[0]->mutable_gpu_data();
    Dtype* Y = full_y.mutable_gpu_data();

	caffe_gpu_set(top[0]->count(), (Dtype)0, T);
	caffe_gpu_set(full_y.count(), (Dtype)0, Y);

	int coff_num = top[0]->shape(1);

	const int nthreads = N * (coff_num - 6);
	set_value_to_constant<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
		CAFFE_CUDA_NUM_THREADS >> >(nthreads, N, (coff_num - 6)/2, ctl_points, Y);

    const Dtype* inv_delta = inv_delta_c.gpu_data();
    for(int i = 0; i < N; ++i) {
        Dtype* curr_t = T + coff_num * i;
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, coff_num/2, 2, coff_num/2, (Dtype)1., inv_delta, Y + coff_num * i, (Dtype)0., curr_t);
    }

	//std::cout << prefix << "ending!" << std::endl;
	
	//if (test_defined_count == 10000)
	//{
	//	std::cout << "----------------ctl_points_test gpu-------------------" << std::endl;
	//	const Dtype* ctl_points_test = bottom[0]->cpu_data();
	//	for (int i = 0; i<bottom[0]->count(); ++i) {
	//		std::cout << ctl_points_test[i] << " ";
	//	}
	//	std::cout << std::endl;

	//	std::cout << "----------------Y gpu-------------------" << std::endl;
	//	const Dtype* dY_test = full_y.cpu_data();
	//	for (int i = 0; i<full_y.count(); ++i) {
	//		std::cout << dY_test[i] << " ";
	//	}
	//	std::cout << std::endl;

	//	std::cout << "----------------dT gpu-------------------" << std::endl;
	//	const Dtype* dT_test = top[0]->cpu_data();
	//	for (int i = 0; i<top[0]->count(); ++i) {
	//		std::cout << dT_test[i] << " ";
	//	}
	//	std::cout << std::endl;

	//	Dtype* break_ptr = 0;
	//	*break_ptr = 1;
	//}
}

template <typename Dtype>
__global__ void CToParaBackwardGPU(const int nthreads, int N, int K1, int K2, const Dtype* inv_constant, const Dtype* dTop, Dtype* dBottom) {

	CUDA_KERNEL_LOOP(index, nthreads) {

		//const int dt_index = index % K1;
		const int dinv_row = index % K2;
		const int dinv_col = (index / K2) % K1;
		const int i = index / (K1 * K2);

		int t_index_x = (K2*2) * i + dinv_col;
		int inv_index_x = K2 * dinv_row + dinv_col;
		const Dtype value_x = dTop[t_index_x]* inv_constant[inv_index_x];
		dBottom[index] = value_x;

		int t_index_y = (K2 * 2) * i + dinv_col + K2;
		int inv_index_y = K2 * dinv_row + dinv_col;
		const Dtype value_y = dTop[t_index_y] * inv_constant[inv_index_y];
		dBottom[index + K1 * K2] = value_y;
	}
}
    
template <typename Dtype>
void CToParaLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom) {
	string prefix = "CToParaLayer::Backward_gpu::\t";
	//std::cout << prefix << "Starting!" << std::endl;
	//test_defined_count++;
	//Backward_cpu(top, propagate_down, bottom);
	
    const Dtype* dT = top[0]->gpu_diff();
	const Dtype* inv_c = inv_delta_c.gpu_data();
    Dtype* dC = bottom[0]->mutable_gpu_diff();
	Dtype* dC_tmp_diff = dC_tmp.mutable_gpu_diff();

	caffe_gpu_set(dC_tmp.count(), (Dtype)0., dC_tmp_diff);
	caffe_gpu_set(bottom[0]->count(), (Dtype)0, dC);

	int coff_num = top[0]->shape(1);
	int c_num = bottom[0]->shape(1);

	const int nthreads = N * (K) * (K+3);
	CToParaBackwardGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
		CAFFE_CUDA_NUM_THREADS >> >(nthreads, N, K, (K + 3), inv_c, dT, dC_tmp_diff);

	Dtype* all_ones_2_data = all_ones_2.mutable_gpu_data();
	caffe_gpu_set(all_ones_2.count(), (Dtype)1., all_ones_2_data);

	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 2*K, 1, K + 3,
		(Dtype)1., dC_tmp_diff, all_ones_2_data, (Dtype)0., dC);

	//std::cout << prefix << "ending!" << std::endl;
	


	//if (test_defined_count == 1000)
	//{
	//	std::cout << "----------------dC gpu-------------------" << std::endl;
	//	const Dtype* dC_test = bottom[0]->cpu_diff();
	//	for (int i = 0; i < bottom[0]->count(); ++i) {
	//		std::cout << dC_test[i] << " ";
	//	}
	//	std::cout << std::endl;

	//	Dtype* break_ptr = 0;
	//	*break_ptr = 1;
	//}
}

INSTANTIATE_LAYER_GPU_FUNCS(CToParaLayer);

}





