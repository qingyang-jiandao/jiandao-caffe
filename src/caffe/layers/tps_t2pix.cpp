#include <vector>

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/tps_t2pix.hpp"
#include "caffe/layers/TPSInterpolate.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
        
template <typename Dtype>
void TPStransformerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top){
    //setup for prototxt
    string prefix = "\t\tTPStransformer Layer:: LayerSetUp: \t";

    //setup dU
    if(this->layer_param_.st_tparam().to_compute_du()) {
        to_compute_dU_ = true;
    }

    //get image size H and W
    output_H_ = bottom[0]->shape(2);

    if(this->layer_param_.st_tparam().has_output_h()) {
        output_H_ = this->layer_param_.st_tparam().output_h();
    }
    output_W_ = bottom[0]->shape(3);

    if(this->layer_param_.st_tparam().has_output_w()) {
        output_W_ = this->layer_param_.st_tparam().output_w();
    }
    std::cout<<prefix<<"output_H_ = "<<output_H_<<", output_W_ = "<<output_W_<<std::endl;

	if (this->layer_param_.st_tparam().has_tps_margins_h()) {
		margin_x = this->layer_param_.st_tparam().tps_margins_h();
	}

	if (this->layer_param_.st_tparam().has_tps_margins_w()) {
		margin_y = this->layer_param_.st_tparam().tps_margins_w();
	}
	std::cout << prefix << "margin_x = " << margin_x << ", margin_y = " << margin_y << std::endl;

	//the same matrix(normalize)[0,1] by [0,1] fro all batch elements
    //Create Room
    std::cout<<prefix<<"Initializing the matrix for output grid"<<std::endl;
	
	mat_dim = K + 3;

    vector<int> shape_output(2);
    shape_output[0] = output_H_ * output_W_;
    shape_output[1] = mat_dim;
    output_grid.Reshape(shape_output);

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
			pos[1] = margin_y + (i- num_ctrl_pts_per_side) * (1 - 2 * margin_y) / (num_ctrl_pts_per_side - 1);
		}
		//std::cout << "K=" << K << " pos[0]=" << pos[0] << " pos[1]=" << pos[1] << std::endl;
		output_ctrl_pts.push_back(pos);
	}

    //Initialize 
  //  Dtype* data = output_grid.mutable_cpu_data();
  //  for(int i=0; i < output_H_ * output_W_; ++i) {
  //      Dtype rx, ry, dis;
		//rx = (i / output_W_) * 1.0 / output_H_;
		//ry = (i % output_W_) * 1.0 / output_W_;
  //      data[mat_dim * i] = 1;
  //      data[mat_dim * i + 1] = rx;
  //      data[mat_dim * i + 2] = ry;
  //      for(int j = 0; j<K; j++) {
  //          dis = (pow((rx - output_ctrl_pts[j][0]), 2) + pow((ry - output_ctrl_pts[j][1]), 2));
  //          if(dis==0) {
		//		data[mat_dim * i + j + 3] = (Dtype)0.;
  //          }
  //          else {
		//		data[mat_dim * i + j + 3] = (dis) * log (dis);
  //          }
  //         //std::cout << "i=" << i << ",j=" << j << ",dis=" << dis << std::endl;
  //      }
  //  }

	Dtype* data = output_grid.mutable_cpu_data();
	for (int i = 0; i < output_H_ * output_W_; ++i) {
		Dtype rx, ry, dis;
		rx = (i / output_W_) * 1.0 / output_H_;
		ry = (i % output_W_) * 1.0 / output_W_;
		int j = 0;
		for (; j<K; j++) {
			dis = (pow((rx - output_ctrl_pts[j][0]), 2) + pow((ry - output_ctrl_pts[j][1]), 2));
			if (dis == 0) {
				data[mat_dim * i + j] = (Dtype)0.;
			}
			else {
				data[mat_dim * i + j] = (dis)* log(dis);
			}
			//std::cout << "i=" << i << " j=" << j << " rx=" << rx << " ry=" << ry << " dis=" << dis << std::endl;
		}
		data[mat_dim * i + j] = 1; j++;
		data[mat_dim * i + j] = rx; j++;
		data[mat_dim * i + j] = ry;
	}

	//initialize the matrix for input grid 
	//the input grid gives, for(i,j) in the output, the corresponding position(projection) on the input
    std::cout<<prefix<<"Initializing the matrix for input grid"<<std::endl;
        
    vector<int> shape_input(3);
    shape_input[0] = bottom[1]->shape(0);
    shape_input[1] = output_H_ * output_W_;
    shape_input[2] = 2;
    input_grid.Reshape(shape_input);
        
    std::cout<<prefix<<"Initialization done."<<std::endl;
}

template <typename Dtype>
void TPStransformerLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top) {
    string prefix = "\t\tTPStransformer Layer:: Reshape: \t";

    //if(global_debug) std::cout<<prefix<<"Starting!"<<std::endl;
        
    N = bottom[0]->shape(0);
    C = bottom[0]->shape(1);
    H = bottom[0]->shape(2);
    W = bottom[0]->shape(3);

    //Reshape V
    vector<int> shape(4);
    shape[0] = N;
    shape[1] = C;
    shape[2] = output_H_;
    shape[3] = output_W_;
    top[0]->Reshape(shape);

    // reshape dT_tmp for GPU
    vector<int> dT_tmp_shape(4);
    dT_tmp_shape[0] = N;
    dT_tmp_shape[1] = 2;
    dT_tmp_shape[2] = mat_dim;
    dT_tmp_shape[3] = output_H_ * output_W_ * C;
    dT_tmp.Reshape(dT_tmp_shape);

	// init all_ones_2 for GPU
    vector<int> all_ones_2_shape(1);
    all_ones_2_shape[0] = output_H_ * output_W_ * C;
    all_ones_2.Reshape(all_ones_2_shape);

    //if(global_debug) std::cout<<prefix<<"Finished."<<std::endl;
}

//Bilinear interpolate
template <typename Dtype>
Dtype TPStransformerLayer<Dtype>::transform_forward_cpu(const Dtype* pic, Dtype px, Dtype py) {
    string prefix = "\t\tTPStransformer Layer:: transform_forward_cpu:: \t";

    Dtype res = (Dtype)0.;

    //integral coordinates
    Dtype x = px * H;
    Dtype y = py * W;

    //for(int m = floor(x); m <= ceil(x); ++m) {
    //    for(int n = floor(y); n <= ceil(y); ++n){
    //        if(m >= 0 && m < H && n >= 0 && n < W){
    //            res += (1 - abs(x - m)) * (1 - abs(y - n)) * pic[m * W + n];
    //        }
    //    }	
    //}

	int m, n; Dtype w;

	m = floor(x); n = floor(y); w = 0;

	if (m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));
		res += w * pic[m * W + n];
	}

	m = floor(x) + 1; n = floor(y); w = 0;

	if (m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));
		res += w * pic[m * W + n];
	}

	m = floor(x); n = floor(y) + 1; w = 0;

	if (m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));
		res += w * pic[m * W + n];
	}

	m = floor(x) + 1; n = floor(y) + 1; w = 0;

	if (m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));
		res += w * pic[m * W + n];
	}

    return res;
}

template <typename Dtype>
void TPStransformerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top) {
    string prefix = "\t\tTPStransformer Layer:: Forward_cpu: \t ";

   // if(global_debug) std::cout<<prefix<<"Starting"<<std::endl;

	//declare the variables
    const Dtype* U = bottom[0]->cpu_data();
    const Dtype* T = bottom[1]->cpu_data();

    //Dtype* full_T_data = full_T.mutable_cpu_data();
    Dtype* input_grid_data = input_grid.mutable_cpu_data();
    const Dtype* output_grid_data = output_grid.cpu_data();
    Dtype* V = top[0]->mutable_cpu_data();

    //initialize mutable_cpu_data arrays
    caffe_set(input_grid.count(), (Dtype)0, input_grid_data);
    caffe_set(top[0]->count(), (Dtype)0, V);
    
	//int coff_num = bottom[1]->shape(1);
	//for (int i = 0; i < N; ++i) {
	//	for (int j = 0; j < coff_num; j++) {
	//		full_T_data[full_T.offset(i, j)] = T[bottom[1]->offset(i, j)];
	//	}
	//}

	int coff_num = bottom[1]->shape(1);

	//For each input in the batch, compute P_input=T*P_output in [-1, 1]
    for(int i = 0; i < N; ++i) {
        Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 2) * i;
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, output_H_ * output_W_, 2, mat_dim, (Dtype)1., output_grid_data, T + coff_num * i, (Dtype)0., coordinates);

        int row_idx;
        Dtype px, py;  //x,y in [0,1]
            
        for(int j = 0; j < C; ++j)
            for(int s = 0; s < output_H_; ++s)
                for(int t = 0; t < output_W_; ++t) {
                    row_idx = output_W_ * s + t;  
                    px = coordinates[row_idx * 2];
                    py = coordinates[row_idx * 2 + 1];
				
                    if(global_debug && i==0)std::cout<<"j="<<j<<", s="<<s<<", t="<<t<<", px="<<px<<", py="<<py<<std::endl;	
                    V[top[0]->offset(i, j, s, t)] = transform_forward_cpu(U + bottom[0]->offset(i,j,0,0), px, py);    
                }
    }

	//if (test_defined_count == 100)
	//{
	//	std::cout << "cpu input_grid" << std::endl;
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
	//}

    //if(global_debug) std::cout<<prefix<<"Finished."<<std::endl;
}

template <typename Dtype>
void TPStransformerLayer<Dtype>::transform_backward_cpu(Dtype dV, const Dtype* U, const Dtype px,const Dtype py, Dtype* dU, Dtype& dpx, Dtype& dpy) {
    string prefix = "\t\tTPStransformer Layer:: transform_backward_cpu:\t";

    //Get integral coordinates (x,y)
	Dtype x = px * H;
	Dtype y = py * W;

    //for(int m = floor(x); m <= ceil(x); ++m) {
    //    for(int n = floor(y); n <= ceil(y); ++n) {
    //        if(m >= 0 && m < H && n >= 0 && n < W) {
    //            dU[m * W +n] += dV * (1 - abs(x-m)) * (1 - abs(y - n));
    //            dpx += caffe_sign<Dtype>(m - x) * (1 - abs(y - n)) * U[m * W + n] * dV * H;
    //            dpy += caffe_sign<Dtype>(n - y) * (1 - abs(x - m)) * U[m * W + n] * dV * W;
    //        } 
    //    }
    //}

	int m, n; Dtype w;

	// left-bottom neighbor
	m = floor(x); n = floor(y);
	if (m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));
		dU[m * W + n] += w * dV;

		dpx -= (1 - (y - n)) * U[m * W + n] * dV * H;
		dpy -= (1 - (x - m)) * U[m * W + n] * dV * W;
	}

	// left-top neighbor
	m = floor(x); n = floor(y) + 1;
	if (m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));
		dU[m * W + n] += w * dV;

		dpx -= (1 - (n - y)) * U[m * W + n] * dV * H;
		dpy += (1 - (x - m)) * U[m * W + n] * dV * W;
	}

	// right-bottom neighbor
	m = floor(x) + 1; n = floor(y);
	if (m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));
		dU[m * W + n] += w * dV;

		dpx += (1 - (y - n)) * U[m * W + n] * dV * H;
		dpy -= (1 - (m - x)) * U[m * W + n] * dV * W;
	}

	// right-top neighbor
	m = floor(x) + 1; n = floor(y) + 1;
	if (m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));
		dU[m * W + n] += w * dV;

		dpx += (1 - (n - y)) * U[m * W + n] * dV * H;
		dpy += (1 - (m - x)) * U[m * W + n] * dV * W;
	}
}

//Compute dV/dT and dV/dU
template <typename Dtype>
void TPStransformerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom) {
    string prefix ="\t\tTPStransformer Layer:: Backward_cpu: \t";

    //std::cout<<prefix<<"Starting!"<<std::endl;

    const Dtype* dV = top[0]->cpu_diff();
    const Dtype* U = bottom[0]->cpu_data();
	const Dtype* input_grid_data = input_grid.cpu_data();
    const Dtype* output_grid_data = output_grid.cpu_data();

    Dtype* dU = bottom[0]->mutable_cpu_diff();
    Dtype* dT = bottom[1]->mutable_cpu_diff();
    Dtype* input_grid_diff = input_grid.mutable_cpu_diff();
        
    caffe_set(bottom[0]->count(), (Dtype)0, dU);
    caffe_set(bottom[1]->count(), (Dtype)0, dT);
    caffe_set(input_grid.count(), (Dtype)0, input_grid_diff);

	int coff_num = bottom[1]->shape(1);

    for(int i = 0; i < N; ++i){
        const Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 2) * i;
        Dtype* coordinates_diff = input_grid_diff + (output_H_ * output_W_ * 2) * i;
            
        int row_idx;
        Dtype px, py, dpx, dpy, delta_dpx, delta_dpy;
            
        for(int s = 0; s < output_H_; ++s){
            for(int t = 0; t < output_W_; ++t){
                    
                row_idx = output_W_ * s + t;
                    
                px = coordinates[row_idx * 2];
                py = coordinates[row_idx * 2 + 1];
                    
                for(int j = 0; j < C; ++j) {
                    delta_dpx = delta_dpy = (Dtype)0.;

                    transform_backward_cpu(dV[top[0]->offset(i, j, s, t)], U + bottom[0]->offset(i, j, 0, 0), px, py, 
                                dU + bottom[0]->offset(i, j, 0, 0), delta_dpx, delta_dpy);
                    coordinates_diff[row_idx * 2] += delta_dpx;
                    coordinates_diff[row_idx * 2 + 1] += delta_dpy;					
                        
                }

                dpx = coordinates_diff[row_idx * 2];
                dpy = coordinates_diff[row_idx * 2 + 1];

                for(int j = 0; j < mat_dim; ++j) {
                    dT[coff_num * i + j] += dpx * output_grid_data[row_idx * mat_dim + j];
                }

                for(int j = mat_dim; j < coff_num; ++j){
                    dT[coff_num * i + j] += dpy * output_grid_data[row_idx * mat_dim + j - mat_dim];
                }
            }
        }
    }

	//std::cout << prefix << "end!" << std::endl;

	//if (test_defined_count == 1000)
	//{
	//	std::cout << "----------------dT cpu-------------------" << std::endl;
	//	for (int i = 0; i < bottom[1]->count(); ++i) {
	//		std::cout << dT[i] << " ";
	//	}
	//	std::cout << std::endl;
	//}
}	

#ifdef CPU_ONLY
    STUB_GPU(TPStransformerLayer);
    #endif

    INSTANTIATE_CLASS(TPStransformerLayer);
    REGISTER_LAYER_CLASS(TPStransformer);
        
}//namespace caffe

