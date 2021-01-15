#include <vector>
#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <cmath>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/tps_c2t.hpp"
#include "caffe/layers/TPSInterpolate.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{

template <typename Dtype>
void CToParaLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,  const vector<Blob<Dtype>*>& top) {
	string prefix = "\t\tCToPara Layer:: LayerSetUp: \t";
	//if(global_debug_) std::cout<<prefix<<"Starting!"<<std::endl;
          
	if(this->layer_param_.st_cparam().transform_type() == "TPS") {
		transform_type_ = "TPS";
	}
	else {
		CHECK(false) << prefix << "Transformation type only supports TPS now!" << std::endl;
	}
          
          
	if(this->layer_param_.st_cparam().sampler_type() == "bilinear") {
		sampler_type_ = "bilinear";
	}
	else {
		CHECK(false) << prefix << "Sampler type only supports bilinear now!" << std::endl;
	}

	if (this->layer_param_.st_cparam().has_tps_margins_h()) {
		margin_x = this->layer_param_.st_cparam().tps_margins_h();
	}

	if (this->layer_param_.st_cparam().has_tps_margins_w()) {
		margin_y = this->layer_param_.st_cparam().tps_margins_w();
	}

	//vector<vector<Dtype> > output_ctrl_pts(K, vector<Dtype>(2));
	//int num_ctrl_pts_per_side = K / 2;

	std::vector<boost::array<Dtype, 2>> output_ctrl_pts;
	int num_ctrl_pts_per_side = K / 2;
	for (int i = 0; i < K; ++i) {
		boost::array<Dtype, 2> pos;
		
		if (i < num_ctrl_pts_per_side) {
			pos[0] = margin_x;
			pos[1] = margin_y + i * (1 - 2 * margin_y) / (num_ctrl_pts_per_side - 1);
		}
		else {
			pos[0] = 1 - margin_x;
			pos[1] = margin_y + (i - num_ctrl_pts_per_side) * (1 - 2 * margin_y) / (num_ctrl_pts_per_side - 1);
		}

		//std::cout << "K=" << K << " pos[0]=" << pos[0] << std::endl;
		output_ctrl_pts.push_back(pos);
	}

	//for (int i = 0; i < K; ++i) {
	//	boost::array<Dtype, 2> pos = output_ctrl_pts[i];
	//	std::cout << pos[0] << " ";
	//}
	//std::cout << std::endl;
	//for (int i = 0; i < K; ++i) {
	//	boost::array<Dtype, 2> pos = output_ctrl_pts[i];
	//	std::cout << pos[1] << " ";
	//}

	////test
	//{
	//	std::vector<boost::array<Dtype, 1>> value_pts;

	//	int num_ctrl_pts_per_side = K / 2;
	//	for (int i = 0; i < K; ++i) {
	//		boost::array<Dtype, 1> pos;

	//		if (i < num_ctrl_pts_per_side) {
	//			//pos[0] = margin_x;
	//			pos[0] = 0.0 + i * (1 - 2 * 0.0) / (num_ctrl_pts_per_side - 1);
	//		}
	//		else {
	//			//pos[0] = 1 - margin_x;
	//			pos[0] = 0.0 + (i - num_ctrl_pts_per_side) * (1 - 2 * 0.0) / (num_ctrl_pts_per_side - 1);
	//		}

	//		std::cout << "K=" << K << " pos[0]=" << pos[0] << std::endl;
	//		value_pts.push_back(pos);
	//	}

	//	ThinPlateSpline<2, 1, Dtype> spline_test(output_ctrl_pts, value_pts);
	//	for (int idxRow = 0; idxRow < spline_test.inv_delta_c.size1(); idxRow++)
	//	{
	//		for (int idxCol = 0; idxCol < spline_test.inv_delta_c.size2(); idxCol++)
	//		{
	//			std::cout << "idxRow=" << idxRow << " idxCol=" << idxCol << " " << spline_test.inv_delta_c(idxRow, idxCol) << std::endl;
	//		}
	//	}
	//	for (int idxRow = 0; idxRow < spline_test.Wa.size1(); idxRow++)
	//	{
	//		for (int idxCol = 0; idxCol < spline_test.Wa.size2(); idxCol++)
	//		{
	//			std::cout << "wa " << spline_test.Wa(idxRow, idxCol) << std::endl;
	//		}
	//	}

	//	Dtype* break_ptr = 0;
	//	*break_ptr = 1;
	//}

	ThinPlateSpline<2, 2, Dtype> spline(output_ctrl_pts);
        
	//Create Room for the constant matrix A (23 * 23)
	vector<int> shape_a(2);
	shape_a[0] = K + 3;
	shape_a[1] = K + 3;
	inv_delta_c.Reshape(shape_a);

	//Initialize 
	int index = 0;
	Dtype* inv_coeffi = inv_delta_c.mutable_cpu_data();  
	for (int idxRow = 0; idxRow < spline.inv_delta_c.size1(); idxRow++)
	{
		for (int idxCol = 0; idxCol < spline.inv_delta_c.size2(); idxCol++)
		{
			inv_coeffi[index++] = spline.inv_delta_c(idxRow, idxCol);
		}
	}

	CHECK(spline.inv_delta_c.size1()== K + 3) << prefix << "inv_delta_c 's size1 is not 23" << std::endl;
   
	//std::cout<<prefix<<"Finished."<<std::endl;
}
    
template <typename Dtype>
void CToParaLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
    //string prefix = "\t\tC To Para Layer:: Reshape: \t";
    //std::cout<<prefix<<"Starting!"<<std::endl;
        
    N = bottom[0]->shape(0); //N
    C = bottom[0]->shape(1); //40
    //H = bottom[0]->shape(2);
    //W = bottom[0]->shape(3);
        
    //Reshape output T
    vector<int> shape(2);
    shape[0] = N;
    shape[1] = C + 6;
    top[0]->Reshape(shape);
        
    //Reshpae Full C1(N*46*1*1)
    vector<int> shape_c(2);
    shape_c[0] = N;
    shape_c[1] = C + 6;
	full_y.Reshape(shape_c);
        
	// reshape dTheta_tmp
	vector<int> dC_tmp_shape(3);
	dC_tmp_shape[0] = N;
	dC_tmp_shape[1] = 2*K;
	dC_tmp_shape[2] = K+3;
	dC_tmp.Reshape(dC_tmp_shape);

	// init all_ones_2
	vector<int> all_ones_2_shape(1);
	all_ones_2_shape[0] = K + 3;
	all_ones_2.Reshape(all_ones_2_shape);
        
    //if(global_debug) std::cout<<prefix<<"Finished."<<std::endl;
}

template <typename Dtype>
void CToParaLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
    //string prefix = "\t\tC To Para Layer:: Forward_cpu: \t";
    //if(global_debug) std::cout<<prefix<<"Starting!"<<std::endl;
        
    const Dtype* ctl_points = bottom[0]->cpu_data();
    Dtype* T = top[0]->mutable_cpu_data();
    Dtype* Y = full_y.mutable_cpu_data();

    caffe_set(top[0]->count(), (Dtype)0, T);

	int coff_num = top[0]->shape(1);

	for (int i = 0; i < N; ++i) {
		for (int s = 0; s < 2; ++s) {
			for (int j = 0; j < K; ++j) {
				Y[full_y.offset(i, s*(K+3) + j)] = ctl_points[bottom[0]->offset(i, s*K+j)];
			}

			for (int j = K; j < K + 3; ++j) {
				Y[full_y.offset(i, s*(K + 3) + j)] = (Dtype)0;
			}
		}
	}
    
    const Dtype* inv_c = inv_delta_c.cpu_data();
    for(int i = 0; i < N; ++i) {
        Dtype* curr_t = T + coff_num * i;
        //caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 2, 23, 23, (Dtype)1., Y + 46 * i, inv_c, (Dtype)0., curr_t);
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, coff_num/2, 2, coff_num / 2, (Dtype)1., inv_c, Y + coff_num * i, (Dtype)0., curr_t);
    }

	//if (test_defined_count == 10000)
	//{
	//	std::cout << "---------------- Y cpu-------------------" << std::endl;
	//	for (int i = 0; i<full_y.count(); ++i) {
	//		std::cout << Y[i] << " ";
	//	}
	//	std::cout << std::endl;

	//	std::cout << "----------------dT cpu-------------------" << std::endl;
	//	const Dtype* dT_test = top[0]->cpu_data();
	//	for (int i = 0; i < top[0]->count(); ++i) {
	//		std::cout << dT_test[i] << " ";
	//	}
	//	std::cout << std::endl;
	//}

    //std::cout<<prefix<<"Finished."<<std::endl;                    
}
    
    
template <typename Dtype>
void CToParaLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom){
    string prefix = "\t\tC To Para Layer:: Backward_cpu: \t";
    //if(global_debug) std::cout<<prefix<<"Starting!"<<std::endl;

    const Dtype* dT = top[0]->cpu_diff();
	const Dtype* inv_c = inv_delta_c.cpu_data();
    Dtype* dC = bottom[0]->mutable_cpu_diff();

    //initialize mutable_cpu_data
    caffe_set(bottom[0]->count(), (Dtype)0, dC);

	int coff_num = top[0]->shape(1);
	int c_num = bottom[0]->shape(1);

	//Compute dC	
	for(int i = 0; i < N; ++i) {
		for(int s = 0; s < K+3; ++s) {
			for(int t = 0; t < K; ++t) {			
				dC[c_num * i + t] += dT[coff_num * i + s ] * inv_c[(K+3) * s + t];
				dC[c_num * i + t + K] += dT[coff_num * i + s + (K + 3)] * inv_c[(K + 3) * s + t];
			}
		}
	}

	//std::cout << "----------------dt-------------------" << std::endl;
	//for (int i = 0; i<top[0]->count(); ++i) {
	//	std::cout << dT[i] << " ";
	//}
	//std::cout << std::endl;

	//if (test_defined_count == 1000)
	//{
	//	std::cout << "----------------dC cpu-------------------" << std::endl;
	//	const Dtype* dC_test = bottom[0]->cpu_diff();
	//	for (int i = 0; i < bottom[0]->count(); ++i) {
	//		std::cout << dC_test[i] << " ";
	//	}
	//	std::cout << std::endl;
	//}

    //if(global_debug) std::cout<<prefix<<"Finished."<<std::endl;
}

#ifdef CPU_ONLY
STUB_GPU(CToParaLayer);
#endif
    
INSTANTIATE_CLASS(CToParaLayer);
REGISTER_LAYER_CLASS(CToPara);

}





