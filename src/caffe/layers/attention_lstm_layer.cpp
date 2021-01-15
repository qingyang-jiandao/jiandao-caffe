#include <string>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/attention_lstm_layer.hpp"
#include "caffe/layers/lstm_layer.hpp"
#include "caffe/util/math_functions.hpp"

#if 1
//Todo: parameters sharing among different ALSTMs  //by lstm_node
namespace caffe {
		
	template <typename Dtype>
	void AttLSTMLayer<Dtype>::RecurrentInputBlobNames(vector<string>* names) const {
		names->resize(2);
		(*names)[0] = "h_0";
		(*names)[1] = "c_0";
	}

	template <typename Dtype>
	void AttLSTMLayer<Dtype>::RecurrentOutputBlobNames(vector<string>* names) const {
		names->resize(2);
		(*names)[0] = "h_" + format_int(this->T_);
		(*names)[1] = "c_T";
	}

	template <typename Dtype>
	void AttLSTMLayer<Dtype>::RecurrentInputShapes(vector<BlobShape>* shapes) const {
		const int num_cells = this->layer_param_.recurrent_param().num_cells();
		const int num_blobs = 2;
		shapes->resize(num_blobs);
		int i = 0;
		for (i = 0; i < num_blobs; ++i) {
			(*shapes)[i].Clear();
			//(*shapes)[i].add_dim(1);  // a single timestep
			(*shapes)[i].add_dim(this->N_);
			(*shapes)[i].add_dim(num_cells);
		}
	}

	template <typename Dtype>
	void AttLSTMLayer<Dtype>::OutputBlobNames(vector<string>* names) const {
		names->resize(1);
		/*(*names)[0] = "h";*/
		(*names)[0] = "predict_concat";
	}

	template <typename Dtype>
	void AttLSTMLayer<Dtype>::FillUnrolledNet(NetParameter* net_param) const {
		const int num_cells = this->layer_param_.recurrent_param().num_cells();
		const int num_output = this->layer_param_.recurrent_param().num_output();
		CHECK_GT(num_cells, 0) << "num_cells must be positive";
		const FillerParameter& weight_filler =
			this->layer_param_.recurrent_param().weight_filler();
		const FillerParameter& bias_filler =
			this->layer_param_.recurrent_param().bias_filler();

		// Take bottom[0].shape=(40,48,512) bottom[1].shape=(40,16) and recurrent_param().num_cells()=256 for illustrating, 
		// where T=16, N=40, C=512, seq_num=48

		// Add generic LayerParameter's (without bottoms/tops) of layer types we'll
		// use to save redundant code.
		LayerParameter hidden_param;
		hidden_param.set_type("InnerProduct");
		hidden_param.mutable_inner_product_param()->set_num_output(num_cells * 4);
		hidden_param.mutable_inner_product_param()->set_bias_term(false);
		hidden_param.mutable_inner_product_param()->set_axis(2);
		hidden_param.mutable_inner_product_param()->
			mutable_weight_filler()->CopyFrom(weight_filler);

		LayerParameter biased_hidden_param(hidden_param);
		biased_hidden_param.mutable_inner_product_param()->set_bias_term(true);
		biased_hidden_param.mutable_inner_product_param()->
			mutable_bias_filler()->CopyFrom(bias_filler);

		LayerParameter attention_param(hidden_param);
		attention_param.mutable_inner_product_param()->set_num_output(num_cells);

		LayerParameter biased_attention_param(attention_param);
		biased_attention_param.mutable_inner_product_param()->set_bias_term(true);
		biased_attention_param.mutable_inner_product_param()->
			mutable_bias_filler()->CopyFrom(bias_filler); // 

		LayerParameter embed_param;
		embed_param.set_type("Embed");
		embed_param.add_propagate_down(false);
		embed_param.mutable_embed_param()->set_num_output(num_cells);
		embed_param.mutable_embed_param()->set_input_dim(num_output);
		embed_param.mutable_embed_param()->set_bias_term(false);
		embed_param.mutable_embed_param()->
			mutable_weight_filler()->CopyFrom(weight_filler);

		LayerParameter concat_param;
		concat_param.set_type("Concat");
		concat_param.mutable_concat_param()->set_axis(2);

		LayerParameter sum_param;
		sum_param.set_type("Eltwise");
		sum_param.mutable_eltwise_param()->set_operation(
			EltwiseParameter_EltwiseOp_SUM);

		LayerParameter scale_param;
		scale_param.set_type("Scale");
		scale_param.mutable_scale_param()->set_axis(0);

		LayerParameter slice_param;
		slice_param.set_type("Slice");
		slice_param.mutable_slice_param()->set_axis(0);

		LayerParameter softmax_param;
		softmax_param.set_type("Softmax");
		softmax_param.mutable_softmax_param()->set_axis(1);

		LayerParameter split_param;
		split_param.set_type("Split");

		LayerParameter permute_param;
		permute_param.set_type("Permute");

		LayerParameter reshape_param;
		reshape_param.set_type("Reshape");

		LayerParameter bias_param;
		bias_param.set_type("Bias");

		LayerParameter pool_param;
		pool_param.set_type("Pooling");

		vector<BlobShape> input_shapes;
		RecurrentInputShapes(&input_shapes);

		LayerParameter* input_layer_param = net_param->add_layer();
		input_layer_param->set_type("Input");
		input_layer_param->set_name("hidden_and_mem_prev");
		InputParameter* input_param = input_layer_param->mutable_input_param();
		input_layer_param->add_top("c_0");
		input_param->add_shape()->CopyFrom(input_shapes[0]);
		input_layer_param->add_top("h_0");
		input_param->add_shape()->CopyFrom(input_shapes[1]);


		//LayerParameter* dummy_layer_param_c_0 = net_param->add_layer();
		//dummy_layer_param_c_0->set_type("DummyData");
		//dummy_layer_param_c_0->set_name("lstm1_mem_cell_prev");
		//dummy_layer_param_c_0->add_top("c_0");
		//DummyDataParameter* dummy_data_param = dummy_layer_param_c_0->mutable_dummy_data_param();
		//dummy_data_param->add_shape()->CopyFrom(input_shapes[0]);

		//LayerParameter* dummy_layer_param_h_0 = net_param->add_layer();
		//dummy_layer_param_h_0->set_type("DummyData");
		//dummy_layer_param_h_0->set_name("lstm1_hidden_prev");
		//dummy_layer_param_h_0->add_top("h_0");
		//dummy_data_param = dummy_layer_param_h_0->mutable_dummy_data_param();
		//dummy_data_param->add_shape()->CopyFrom(input_shapes[1]);

		LayerParameter* y_slice_param = NULL;
		if (this->phase_ != TEST) {
			y_slice_param = net_param->add_layer();
			y_slice_param->CopyFrom(slice_param);
			y_slice_param->set_name("y_slice");
			y_slice_param->add_bottom("y");
			y_slice_param->mutable_slice_param()->set_axis(1);
		}

		LayerParameter* att_x_param = net_param->add_layer();
		att_x_param->CopyFrom(attention_param);
		att_x_param->set_name("xProj");
		att_x_param->mutable_inner_product_param()->set_axis(2);
		att_x_param->add_bottom("x");  //(40,48,512)
		att_x_param->add_top("xProj");  //(40,48,256)
		//att_x_param->add_propagate_down(true);

		LayerParameter output_concat_layer;
		output_concat_layer.set_name("predict_concat");
		output_concat_layer.set_type("Concat");
		/*output_concat_layer.add_top("h");*/
		output_concat_layer.add_top("predict_concat");
		output_concat_layer.mutable_concat_param()->set_axis(1);

		for (int t = 1; t <= this->T_; ++t) {
			string tm1s = format_int(t - 1);
			string ts = format_int(t);

			if (this->phase_ != TEST && y_slice_param) {
				y_slice_param->add_top("y_" + ts);
			}

			{  
				if (this->phase_ != TEST) {
					//emed
					LayerParameter* embedding_param = net_param->add_layer();
					embedding_param->CopyFrom(embed_param);
					embedding_param->set_name("y_embedding_" + ts);
					ParamSpec* param0 = embedding_param->add_param();
					param0->set_name("embed_param");
					embedding_param->add_bottom("y_" + ts);
					embedding_param->add_top("y_embedding_" + ts); //(40,256)  
				}
				else {
					if (t == 1) {
						LayerParameter* embedding_param = net_param->add_layer();
						embedding_param->CopyFrom(embed_param);
						embedding_param->set_name("y_embedding_" + ts);
						ParamSpec* param0 = embedding_param->add_param();
						param0->set_name("embed_param");
						embedding_param->add_bottom("y");
						embedding_param->add_top("y_embedding_" + ts); //(1,256)  
					}
					else {

						LayerParameter* argmax_prob_param = net_param->add_layer();
						argmax_prob_param->set_type("ArgMax");
						argmax_prob_param->set_name("prob_" + ts);
						argmax_prob_param->mutable_argmax_param()->set_axis(1);
						argmax_prob_param->add_bottom("prob_" + tm1s);
						argmax_prob_param->add_top("y_" + ts); //(1,1)


						LayerParameter* embedding_param = net_param->add_layer();
						embedding_param->CopyFrom(embed_param);
						embedding_param->set_name("y_embedding_" + ts);
						ParamSpec* param0 = embedding_param->add_param();
						param0->set_name("embed_param");
						embedding_param->add_bottom("y_" + ts);
						embedding_param->add_top("y_embedding_" + ts); //(1,256)  
					}
				}
			}

			{
				LayerParameter* att_s_param = net_param->add_layer();
				att_s_param->CopyFrom(attention_param);
				att_s_param->mutable_inner_product_param()->set_axis(1);
				ParamSpec* param0 = att_s_param->add_param();
				param0->set_name("hidden_att_param_0");
				att_s_param->set_name("sProj_" + ts);
				att_s_param->add_bottom("h_" + tm1s); //(40,256) 
				att_s_param->add_top("sProj_" + ts);     //(40,256)
			}

			if (this->phase_ != TEST) {
				{
					LayerParameter* tile_s_param = net_param->add_layer();
					tile_s_param->set_type("Tile");
					tile_s_param->set_name("tile_sProj_" + ts);
					tile_s_param->mutable_tile_param()->set_axis(1);
					tile_s_param->mutable_tile_param()->set_tiles(this->SEQ_NUM_);
					tile_s_param->add_bottom("sProj_" + ts);
					tile_s_param->add_top("tile_sProj_" + ts);  ////(40,48*256)
				}

				{
					LayerParameter* tile_s_reshape_param = net_param->add_layer();
					tile_s_reshape_param->set_type("Reshape");
					tile_s_reshape_param->set_name("tile_sProj_reshape_" + ts);
					BlobShape* new_shape = tile_s_reshape_param->mutable_reshape_param()->mutable_shape();
					new_shape->Clear();
					new_shape->add_dim(0);
					new_shape->add_dim(-1);
					new_shape->add_dim(num_cells);
					tile_s_reshape_param->add_bottom("tile_sProj_" + ts);
					tile_s_reshape_param->add_top("tile_sProj_reshape_" + ts);  ////(40,48,256)
				}

				// m_input := 
				//         := m_x_a_{t-1} + m_{t-1}
				{
					LayerParameter* m_sum_layer = net_param->add_layer();
					m_sum_layer->CopyFrom(sum_param);
					m_sum_layer->set_name("sum_proj_" + ts);
					m_sum_layer->add_bottom("xProj");
					m_sum_layer->add_bottom("tile_sProj_reshape_" + ts);
					m_sum_layer->add_top("sum_proj_" + ts);  //(40,48,256)
				}
			}
			else {
				{
					LayerParameter* s_reshape_param = net_param->add_layer();
					s_reshape_param->set_type("Reshape");
					s_reshape_param->set_name("sProj_reshape_" + ts);
					BlobShape* new_shape = s_reshape_param->mutable_reshape_param()->mutable_shape();
					new_shape->Clear();
					new_shape->add_dim(num_cells);
					s_reshape_param->add_bottom("sProj_" + ts);
					s_reshape_param->add_top("sProj_reshape_" + ts);  ////(256)
				}
				// m_input := 
				//         := m_x_a_{t-1} + m_{t-1}
				{
					LayerParameter* m_sum_layer = net_param->add_layer();
					m_sum_layer->CopyFrom(bias_param);
					m_sum_layer->mutable_bias_param()->set_axis(2);
					m_sum_layer->set_name("sum_proj_" + ts);
					m_sum_layer->add_bottom("xProj");
					m_sum_layer->add_bottom("sProj_reshape_" + ts);
					m_sum_layer->add_top("sum_proj_" + ts);  //(1,48,256)
				}
			}

			{
				LayerParameter* m_tanh_layer = net_param->add_layer();
				m_tanh_layer->set_type("TanH");
				m_tanh_layer->set_name("tanh_proj_" + ts);
				m_tanh_layer->add_bottom("sum_proj_" + ts);
				m_tanh_layer->add_top("sum_proj_" + ts);  //(40,48,256)
			}

			{
				LayerParameter* att_x_ap_param = net_param->add_layer();
				att_x_ap_param->CopyFrom(attention_param);
				att_x_ap_param->set_name("predict_att_" + ts);
				att_x_ap_param->mutable_inner_product_param()->set_axis(2);
				att_x_ap_param->mutable_inner_product_param()->set_num_output(1);
				ParamSpec* param0 = att_x_ap_param->add_param();
				param0->set_name("predict_att_param_0");
				att_x_ap_param->add_bottom("sum_proj_" + ts);
				att_x_ap_param->add_top("predict_att_" + ts);  //(40,48,1)
			}

			{
				LayerParameter* reshape_m_param = net_param->add_layer();
				reshape_m_param->CopyFrom(reshape_param);
				BlobShape* new_shape = reshape_m_param->mutable_reshape_param()->mutable_shape();
				new_shape->Clear();
				new_shape->add_dim(0);
				new_shape->add_dim(-1);
				reshape_m_param->set_name("reshape_predict_att_" + ts);
				reshape_m_param->add_bottom("predict_att_" + ts);
				reshape_m_param->add_top("reshape_predict_att_" + ts); //(40,48) 
			}

			// Add a softmax layers to generate attention masks  
			{
				LayerParameter* softmax_m_param = net_param->add_layer();
				softmax_m_param->CopyFrom(softmax_param);
				softmax_m_param->mutable_softmax_param()->set_axis(1);
				softmax_m_param->mutable_softmax_param()->set_engine(SoftmaxParameter_Engine_CAFFE);
				softmax_m_param->set_name("att_weight_" + ts);
				softmax_m_param->add_bottom("reshape_predict_att_" + ts);
				softmax_m_param->add_top("att_weight_" + ts); //(40,48)
			}

			// Conbine mask with input features  
			{
				LayerParameter* scale_x_param = net_param->add_layer();
				scale_x_param->CopyFrom(scale_param);
				scale_x_param->mutable_scale_param()->set_axis(0);
				scale_x_param->set_name("att_product_" + ts);
				scale_x_param->add_bottom("x");
				scale_x_param->add_bottom("att_weight_" + ts);
				scale_x_param->add_top("att_product_" + ts);  //(40,48,512)
			}

			{
				LayerParameter* permute_att_param = net_param->add_layer();
				permute_att_param->CopyFrom(permute_param);
				permute_att_param->set_name("permute_att_product_" + ts);
				permute_att_param->mutable_permute_param()->add_order(0);
				permute_att_param->mutable_permute_param()->add_order(2);
				permute_att_param->mutable_permute_param()->add_order(1);
				permute_att_param->add_bottom("att_product_" + ts);
				permute_att_param->add_top("permute_att_product_" + ts);  //(40,512,48)
			}

			{
				LayerParameter* context_x_param = net_param->add_layer();
				context_x_param->set_type("Reduction");
				context_x_param->set_name("x_context_" + ts);
				context_x_param->mutable_reduction_param()->set_axis(2);
				context_x_param->add_bottom("permute_att_product_" + ts);
				context_x_param->add_top("x_context_" + ts);  ////(40,512)
			}

			{
				LayerParameter* decode_concat_param = net_param->add_layer();
				decode_concat_param->CopyFrom(concat_param);
				decode_concat_param->set_name("concat_" + ts);
				decode_concat_param->mutable_concat_param()->set_axis(1);
				decode_concat_param->add_bottom("x_context_" + ts);
				decode_concat_param->add_bottom("y_embedding_" + ts);
				decode_concat_param->add_top("concat_x_y_" + ts);   //(40,768)
			}

			{
				LayerParameter* lstm_unit_param = net_param->add_layer();
				lstm_unit_param->set_type("LSTMNode");
				lstm_unit_param->mutable_lstmnode_param()->set_num_cells(num_cells);
				lstm_unit_param->mutable_lstmnode_param()->mutable_input_weight_filler()->CopyFrom(weight_filler);
				lstm_unit_param->mutable_lstmnode_param()->mutable_input_gate_weight_filler()->CopyFrom(weight_filler);
				lstm_unit_param->mutable_lstmnode_param()->mutable_forget_gate_weight_filler()->CopyFrom(weight_filler);
				lstm_unit_param->mutable_lstmnode_param()->mutable_output_gate_weight_filler()->CopyFrom(weight_filler);
				lstm_unit_param->mutable_lstmnode_param()->mutable_input_bias_filler()->CopyFrom(bias_filler);
				lstm_unit_param->mutable_lstmnode_param()->mutable_input_gate_bias_filler()->CopyFrom(bias_filler);
				lstm_unit_param->mutable_lstmnode_param()->mutable_forget_gate_bias_filler()->CopyFrom(bias_filler);
				lstm_unit_param->mutable_lstmnode_param()->mutable_forget_gate_bias_filler()->set_value(1.0);
				lstm_unit_param->mutable_lstmnode_param()->mutable_output_gate_bias_filler()->CopyFrom(bias_filler);
				ParamSpec* param0 = lstm_unit_param->add_param();
				param0->set_name("lstm_node_param_0");
				ParamSpec* param1 = lstm_unit_param->add_param();
				param1->set_name("lstm_node_param_1");
				lstm_unit_param->add_bottom("concat_x_y_" + ts);//(40,768)
				lstm_unit_param->add_bottom("c_" + tm1s); //(40,256)
				lstm_unit_param->add_top("h_" + ts);//(40,256)
				lstm_unit_param->add_top("c_" + ts);//(40,256)
				lstm_unit_param->set_name("node_" + ts);
				if (t == 1) {
					lstm_unit_param->add_propagate_down(true);
					lstm_unit_param->add_propagate_down(false);
				}
				else {
					lstm_unit_param->add_propagate_down(true);
					lstm_unit_param->add_propagate_down(true);
				}
			}

			{
				LayerParameter* predict_param = net_param->add_layer();
				predict_param->set_type("InnerProduct");
				predict_param->mutable_inner_product_param()->set_num_output(num_output);
				predict_param->mutable_inner_product_param()->set_bias_term(true);
				predict_param->mutable_inner_product_param()->set_axis(1);
				predict_param->mutable_inner_product_param()->
					mutable_weight_filler()->CopyFrom(weight_filler);
				predict_param->mutable_inner_product_param()->
					mutable_bias_filler()->CopyFrom(bias_filler);
				ParamSpec* param0 = predict_param->add_param();
				param0->set_name("predict_param_0");
				param0->set_lr_mult(1.0);
				param0->set_decay_mult(1.0);
				ParamSpec* param1 = predict_param->add_param();
				param1->set_name("predict_param_1");
				param1->set_lr_mult(2.0);
				param1->set_decay_mult(0.0);
				predict_param->set_name("predict_" + ts);
				predict_param->add_bottom("h_" + ts); //(40,256) 
				predict_param->add_top("predict_" + ts);     //(40,6863)
			}

			if (this->phase_ != TEST) {
				output_concat_layer.add_bottom("predict_" + ts);
			}
			else {
				LayerParameter* softmax_prob_param = net_param->add_layer();
				softmax_prob_param->set_type("Softmax");
				softmax_prob_param->mutable_softmax_param()->set_axis(1);
				softmax_prob_param->set_name("prob_" + ts);
				softmax_prob_param->add_bottom("predict_" + ts);
				softmax_prob_param->add_top("prob_" + ts); //(1,6863)

				output_concat_layer.add_bottom("prob_" + ts);
			}
		}  // for (int t = 1; t <= this->T_; ++t)

		{
			LayerParameter* c_T_copy_param = net_param->add_layer();
			c_T_copy_param->CopyFrom(split_param);
			c_T_copy_param->add_bottom("c_" + format_int(this->T_));
			c_T_copy_param->add_top("c_T");
		}
		net_param->add_layer()->CopyFrom(output_concat_layer);
	}

	INSTANTIATE_CLASS(AttLSTMLayer);
	REGISTER_LAYER_CLASS(AttLSTM);

}  // namespace caffe

#else
//Todo: parameters sharing among different ALSTMs  //by lstm_unit
namespace caffe {

	template <typename Dtype>
	void AttLSTMLayer<Dtype>::RecurrentInputBlobNames(vector<string>* names) const {
		names->resize(3);
		(*names)[0] = "h_0";
		(*names)[1] = "c_0";
		(*names)[2] = "cont_0";
	}

	template <typename Dtype>
	void AttLSTMLayer<Dtype>::RecurrentOutputBlobNames(vector<string>* names) const {
		names->resize(2);
		(*names)[0] = "h_" + format_int(this->T_);
		(*names)[1] = "c_T";
	}

	//template <typename Dtype>
	//void AttLSTMLayer<Dtype>::RecurrentInputShapes(vector<BlobShape>* shapes) const {
	//	const int num_cells = this->layer_param_.recurrent_param().num_cells();
	//	const int num_blobs = 2;
	//	shapes->resize(num_blobs);
	//	int i = 0;
	//	for (i = 0; i < num_blobs; ++i) {
	//		(*shapes)[i].Clear();
	//		(*shapes)[i].add_dim(1);  // a single timestep
	//		(*shapes)[i].add_dim(this->N_);
	//		(*shapes)[i].add_dim(num_cells);
	//	}
	//}

	template <typename Dtype>
	void AttLSTMLayer<Dtype>::RecurrentInputShapes(vector<BlobShape>* shapes) const {
		const int num_cells = this->layer_param_.recurrent_param().num_cells();
		const int num_blobs = 3;
		shapes->resize(num_blobs);
		int i = 0;
		for (i = 0; i < num_blobs-1; ++i) {
			(*shapes)[i].Clear();
			(*shapes)[i].add_dim(1);  // a single timestep
			(*shapes)[i].add_dim(this->N_);
			(*shapes)[i].add_dim(num_cells);
		}

		(*shapes)[i].Clear();
		(*shapes)[i].add_dim(1);  // a single timestep
		(*shapes)[i].add_dim(this->N_);
	}

	template <typename Dtype>
	void AttLSTMLayer<Dtype>::OutputBlobNames(vector<string>* names) const {
		names->resize(2);
		(*names)[0] = "h";
		(*names)[1] = "mask";
	}

	template <typename Dtype>
	void AttLSTMLayer<Dtype>::FillUnrolledNet(NetParameter* net_param) const {
		const int num_cells = this->layer_param_.recurrent_param().num_cells();
		const int num_output = this->layer_param_.recurrent_param().num_output();
		CHECK_GT(num_cells, 0) << "num_cells must be positive";
		const FillerParameter& weight_filler =
			this->layer_param_.recurrent_param().weight_filler();
		const FillerParameter& bias_filler =
			this->layer_param_.recurrent_param().bias_filler();

		// Take bottom[0].shape=(40,48,512) bottom[1].shape=(40,16) and recurrent_param().num_cells()=256 for illustrating, 
		// where T=16, N=40, C=512, seq_num=48

		// Add generic LayerParameter's (without bottoms/tops) of layer types we'll
		// use to save redundant code.
		LayerParameter hidden_param;
		hidden_param.set_type("InnerProduct");
		hidden_param.mutable_inner_product_param()->set_num_output(num_cells * 4);
		hidden_param.mutable_inner_product_param()->set_bias_term(false);
		hidden_param.mutable_inner_product_param()->set_axis(2);
		hidden_param.mutable_inner_product_param()->
			mutable_weight_filler()->CopyFrom(weight_filler);

		LayerParameter biased_hidden_param(hidden_param);
		biased_hidden_param.mutable_inner_product_param()->set_bias_term(true);
		biased_hidden_param.mutable_inner_product_param()->
			mutable_bias_filler()->CopyFrom(bias_filler);

		LayerParameter attention_param(hidden_param);
		attention_param.mutable_inner_product_param()->set_num_output(num_cells);

		LayerParameter biased_attention_param(attention_param);
		biased_attention_param.mutable_inner_product_param()->set_bias_term(true);
		biased_attention_param.mutable_inner_product_param()->
			mutable_bias_filler()->CopyFrom(bias_filler); // 

		LayerParameter embed_param;
		embed_param.set_type("Embed");
		embed_param.add_propagate_down(false);
		embed_param.mutable_embed_param()->set_num_output(num_cells);
		embed_param.mutable_embed_param()->set_input_dim(num_output);
		embed_param.mutable_embed_param()->set_bias_term(false);
		embed_param.mutable_embed_param()->
			mutable_weight_filler()->CopyFrom(weight_filler);

		LayerParameter concat_param;
		concat_param.set_type("Concat");
		concat_param.mutable_concat_param()->set_axis(2);

		LayerParameter sum_param;
		sum_param.set_type("Eltwise");
		sum_param.mutable_eltwise_param()->set_operation(
			EltwiseParameter_EltwiseOp_SUM);

		LayerParameter scale_param;
		scale_param.set_type("Scale");
		scale_param.mutable_scale_param()->set_axis(0);

		LayerParameter slice_param;
		slice_param.set_type("Slice");
		slice_param.mutable_slice_param()->set_axis(0);

		LayerParameter softmax_param;
		softmax_param.set_type("Softmax");
		softmax_param.mutable_softmax_param()->set_axis(1);

		LayerParameter split_param;
		split_param.set_type("Split");

		LayerParameter permute_param;
		permute_param.set_type("Permute");

		LayerParameter reshape_param;
		reshape_param.set_type("Reshape");

		LayerParameter bias_param;
		bias_param.set_type("Bias");

		LayerParameter pool_param;
		pool_param.set_type("Pooling");

		vector<BlobShape> input_shapes;
		RecurrentInputShapes(&input_shapes);

		LayerParameter* input_layer_param = net_param->add_layer();
		input_layer_param->set_type("Input");
		InputParameter* input_param = input_layer_param->mutable_input_param();

		input_layer_param->add_top("c_0");
		input_param->add_shape()->CopyFrom(input_shapes[0]);

		input_layer_param->add_top("h_0");
		input_param->add_shape()->CopyFrom(input_shapes[1]);

		input_layer_param->add_top("cont_0");
		input_param->add_shape()->CopyFrom(input_shapes[2]);

		//LayerParameter* cont_layer_param = net_param->add_layer();
		//cont_layer_param->set_type("ContinuationIndicator");
		//ContinuationIndicatorParameter* continuation_indicator_param = cont_layer_param->mutable_continuation_indicator_param();
		//continuation_indicator_param->set_time_step(1);
		//continuation_indicator_param->set_batch_size(this->N_);
		//cont_layer_param->set_name("indicator");
		//cont_layer_param->add_top("cont_0");

		//LayerParameter* cont_slice_param = net_param->add_layer();
		//cont_slice_param->set_type("Slice");
		//cont_slice_param->mutable_slice_param()->set_axis(0);
		//cont_slice_param->set_name("cont_slice");
		//cont_slice_param->add_bottom("cont");
		//cont_slice_param->add_top("cont_0");

		LayerParameter* y_slice_param = NULL;
		if (this->phase_ != TEST) {
			y_slice_param = net_param->add_layer();
			y_slice_param->CopyFrom(slice_param);
			y_slice_param->set_name("y_slice");
			y_slice_param->add_bottom("y");
			y_slice_param->mutable_slice_param()->set_axis(1);
		}

		LayerParameter* att_x_param = net_param->add_layer();
		att_x_param->CopyFrom(attention_param);
		att_x_param->set_name("xProj");
		att_x_param->mutable_inner_product_param()->set_axis(2);
		att_x_param->add_bottom("x");  //(40,48,512)
		att_x_param->add_top("xProj");  //(40,48,256)

		LayerParameter* permute_x_a_param = net_param->add_layer();
		permute_x_a_param->CopyFrom(permute_param);
		permute_x_a_param->set_name("permute_xProj");
		permute_x_a_param->mutable_permute_param()->add_order(1);
		permute_x_a_param->mutable_permute_param()->add_order(0);
		permute_x_a_param->mutable_permute_param()->add_order(2);
		permute_x_a_param->add_bottom("xProj");
		permute_x_a_param->add_top("permute_xProj");  //(48,40,256)

		LayerParameter output_concat_layer;
		output_concat_layer.set_name("h_concat");
		output_concat_layer.set_type("Concat");
		output_concat_layer.add_top("h");
		output_concat_layer.mutable_concat_param()->set_axis(2);
		//if (this->phase_ != TEST) {
		//	output_concat_layer.mutable_concat_param()->set_axis(2);
		//}
		//else {
		//	output_concat_layer.mutable_concat_param()->set_axis(1);
		//}

		LayerParameter output_m_layer;
		output_m_layer.set_name("m_concat");
		output_m_layer.set_type("Concat");
		output_m_layer.add_top("mask");
		output_m_layer.mutable_concat_param()->set_axis(0); // output attention mask

		for (int t = 1; t <= this->T_; ++t) {
			string tm1s = format_int(t - 1);
			string ts = format_int(t);

			if (this->phase_ != TEST && y_slice_param) {
				y_slice_param->add_top("y_" + ts);
			}

			{
				if (this->phase_ != TEST) {
					//emed
					LayerParameter* embedding_param = net_param->add_layer();
					embedding_param->CopyFrom(embed_param);
					embedding_param->set_name("y_embedding_" + ts);
					ParamSpec* param0 = embedding_param->add_param();
					param0->set_name("embed_param");
					embedding_param->add_bottom("y_" + ts);
					embedding_param->add_top("y_embedding_" + ts); //(40,256)  
				}
				else {
					if (t == 1) {
						LayerParameter* embedding_param = net_param->add_layer();
						embedding_param->CopyFrom(embed_param);
						embedding_param->set_name("y_embedding_" + ts);
						ParamSpec* param0 = embedding_param->add_param();
						param0->set_name("embed_param");
						embedding_param->add_bottom("y");
						embedding_param->add_top("y_embedding_" + ts); //(1,256)  
					}
					else {

						LayerParameter* predict_reshape_param = net_param->add_layer();
						predict_reshape_param->set_type("Reshape");
						predict_reshape_param->set_name("prob_reshape_" + ts);
						BlobShape* new_shape = predict_reshape_param->mutable_reshape_param()->mutable_shape();
						new_shape->Clear();
						new_shape->add_dim(-1);
						new_shape->add_dim(num_output);
						predict_reshape_param->add_bottom("prob_" + tm1s);
						predict_reshape_param->add_top("prob_reshape_" + tm1s);  ////(1,6863)

						LayerParameter* argmax_prob_param = net_param->add_layer();
						argmax_prob_param->set_type("ArgMax");
						argmax_prob_param->set_name("prob_" + ts);
						argmax_prob_param->mutable_argmax_param()->set_axis(1);
						argmax_prob_param->add_bottom("prob_reshape_" + tm1s);
						argmax_prob_param->add_top("y_" + ts); //(1,1)


						LayerParameter* embedding_param = net_param->add_layer();
						embedding_param->CopyFrom(embed_param);
						embedding_param->set_name("y_embedding_" + ts);
						ParamSpec* param0 = embedding_param->add_param();
						param0->set_name("embed_param");
						embedding_param->add_bottom("y_" + ts);
						embedding_param->add_top("y_embedding_" + ts); //(1,256)  
					}
				}
			}

			// Add a layer to generate attention weights
			{
				LayerParameter* reshape_h_param = net_param->add_layer();
				reshape_h_param->CopyFrom(reshape_param);
				BlobShape* new_shape = reshape_h_param->mutable_reshape_param()->mutable_shape();
				new_shape->Clear();
				new_shape->add_dim(-1);
				new_shape->add_dim(num_cells);
				reshape_h_param->set_name("reshape_h_" + tm1s);
				reshape_h_param->add_bottom("h_" + tm1s);
				reshape_h_param->add_top("reshape_h_" + tm1s); //(40,256)
			}

			{
				LayerParameter* att_m_param = net_param->add_layer();
				att_m_param->CopyFrom(attention_param);
				att_m_param->mutable_inner_product_param()->set_axis(1);
				att_m_param->add_param()->set_name("hidden_att_param_0");
				att_m_param->set_name("sProj_" + ts);
				att_m_param->add_bottom("reshape_h_" + tm1s); //(40,256) 
				att_m_param->add_top("sProj_" + ts);     //(40,256)
			}

			// m_input := 
			//         := m_x_a_{t-1} + m_{t-1}
			{
				LayerParameter* m_sum_layer = net_param->add_layer();
				m_sum_layer->CopyFrom(bias_param);
				m_sum_layer->set_name("sum_proj_" + ts);
				m_sum_layer->add_bottom("permute_xProj");
				m_sum_layer->add_bottom("sProj_" + ts);
				m_sum_layer->add_top("sum_proj_" + ts);  //(48,40,256)
			}

			{
				LayerParameter* m_tanh_layer = net_param->add_layer();
				m_tanh_layer->set_type("TanH");
				m_tanh_layer->set_name("tanh_proj_" + ts);
				m_tanh_layer->add_bottom("sum_proj_" + ts);
				m_tanh_layer->add_top("sum_proj_" + ts);  //(48,40,256)
			}

			{
				LayerParameter* att_x_ap_param = net_param->add_layer();
				att_x_ap_param->CopyFrom(attention_param);
				att_x_ap_param->set_name("predict_att_" + ts);
				att_x_ap_param->mutable_inner_product_param()->set_axis(2);
				att_x_ap_param->mutable_inner_product_param()->set_num_output(1);
				att_x_ap_param->add_param()->set_name("predict_att_param_0");
				att_x_ap_param->add_bottom("sum_proj_" + ts);
				att_x_ap_param->add_top("predict_att_" + ts);  //(48,40,1)
			}

			{
				LayerParameter* permute_m_param = net_param->add_layer();
				permute_m_param->CopyFrom(permute_param);
				permute_m_param->set_name("permute_predict_att_" + ts);
				permute_m_param->mutable_permute_param()->add_order(1);
				permute_m_param->mutable_permute_param()->add_order(0);
				permute_m_param->mutable_permute_param()->add_order(2);
				permute_m_param->add_bottom("predict_att_" + ts);
				permute_m_param->add_top("permute_predict_att_" + ts);  //(40,48,1)
			}

			{
				LayerParameter* reshape_m_param = net_param->add_layer();
				reshape_m_param->CopyFrom(reshape_param);
				BlobShape* new_shape = reshape_m_param->mutable_reshape_param()->mutable_shape();
				new_shape->Clear();
				new_shape->add_dim(0);
				new_shape->add_dim(0);
				reshape_m_param->set_name("reshape_predict_att_" + ts);
				reshape_m_param->add_bottom("permute_predict_att_" + ts);
				reshape_m_param->add_top("reshape_predict_att_" + ts); //(40,48) 
			}

			// Add a softmax layers to generate attention masks  
			{
				LayerParameter* softmax_m_param = net_param->add_layer();
				softmax_m_param->CopyFrom(softmax_param);
				softmax_m_param->mutable_softmax_param()->set_axis(1);
				softmax_m_param->set_name("att_weight_" + ts);
				softmax_m_param->add_bottom("reshape_predict_att_" + ts);
				softmax_m_param->add_top("att_weight_" + ts); //(40,48)
			}

			// Conbine mask with input features  
			{
				LayerParameter* scale_x_param = net_param->add_layer();
				scale_x_param->CopyFrom(scale_param);
				scale_x_param->mutable_scale_param()->set_axis(0);
				scale_x_param->set_name("att_product_" + ts);
				scale_x_param->add_bottom("x");
				scale_x_param->add_bottom("att_weight_" + ts);
				scale_x_param->add_top("att_product_" + ts);  //(40,48,512)
			}

			{
				LayerParameter* permute_att_param = net_param->add_layer();
				permute_att_param->CopyFrom(permute_param);
				permute_att_param->set_name("permute_att_product_" + ts);
				permute_att_param->mutable_permute_param()->add_order(0);
				permute_att_param->mutable_permute_param()->add_order(2);
				permute_att_param->mutable_permute_param()->add_order(1);
				permute_att_param->add_bottom("att_product_" + ts);
				permute_att_param->add_top("permute_att_product_" + ts);  //(40,512,48)
			}

			{
				LayerParameter* context_x_param = net_param->add_layer();
				context_x_param->set_type("Reduction");
				context_x_param->set_name("x_context_" + ts);
				context_x_param->mutable_reduction_param()->set_axis(2);
				context_x_param->add_bottom("permute_att_product_" + ts);
				context_x_param->add_top("x_context_" + ts);  ////(40,512)
			}

			{
				LayerParameter* reshape_y_embedding_param = net_param->add_layer();
				reshape_y_embedding_param->CopyFrom(reshape_param);
				BlobShape* new_shape = reshape_y_embedding_param->mutable_reshape_param()->mutable_shape();
				new_shape->Clear();
				new_shape->add_dim(1);
				reshape_y_embedding_param->mutable_reshape_param()->set_num_axes(0);
				reshape_y_embedding_param->set_name("reshape_y_embedding_" + ts);
				reshape_y_embedding_param->add_bottom("y_embedding_" + ts);
				reshape_y_embedding_param->add_top("reshape_y_embedding_" + ts);  //(1,40,256)  
			}

			{
				LayerParameter* reshape_x_context_param = net_param->add_layer();
				reshape_x_context_param->CopyFrom(reshape_param);
				BlobShape* new_shape = reshape_x_context_param->mutable_reshape_param()->mutable_shape();
				new_shape->Clear();
				new_shape->add_dim(1);
				reshape_x_context_param->mutable_reshape_param()->set_num_axes(0);
				reshape_x_context_param->set_name("reshape_x_context_" + ts);
				reshape_x_context_param->add_bottom("x_context_" + ts);
				reshape_x_context_param->add_top("reshape_x_context_" + ts); //(1,40,512)  
			}

			{
				LayerParameter* decode_concat_param = net_param->add_layer();
				decode_concat_param->CopyFrom(concat_param);
				decode_concat_param->set_name("concat_" + ts);
				decode_concat_param->add_bottom("reshape_x_context_" + ts);
				decode_concat_param->add_bottom("reshape_y_embedding_" + ts);
				decode_concat_param->add_top("concat_x_y_" + ts);   //(1,40,768)
			}

			// Add layer to transform a timestep of x_pool_permute to the hidden state dimension.
			//     W_xc_x_ = W_xc * x_pool_permute + b_c
			{
				LayerParameter* x_transform_param = net_param->add_layer();
				x_transform_param->CopyFrom(biased_hidden_param);
				x_transform_param->mutable_inner_product_param()->set_axis(2);
				x_transform_param->set_name("gate_input_" + ts);
				x_transform_param->add_param()->set_name("W_xc");
				x_transform_param->add_param()->set_name("b_c");
				x_transform_param->add_bottom("concat_x_y_" + ts);
				x_transform_param->add_top("gate_input_" + ts);  //(1,40,256*4)
			}

			/* ========================================================
			// Add layers to flush the hidden state when beginning a new
			// sequence, as indicated by cont_t.
			//     h_conted_{t-1} := cont_t * h_{t-1}
			//
			// Normally, cont_t is binary (i.e., 0 or 1), so:
			//     h_conted_{t-1} := h_{t-1} if cont_t == 1
			//                       0   otherwise

			{
				LayerParameter* cont_h_param = net_param->add_layer();
				cont_h_param->CopyFrom(scale_param);
				cont_h_param->set_name("h_conted_" + tm1s);
				cont_h_param->add_bottom("h_" + tm1s);
				cont_h_param->add_bottom("cont_0");
				cont_h_param->add_top("h_conted_" + tm1s); //(1,40,256)
			}

		    // Add layer to compute
		    //     W_hc_h_{t-1} := W_hc * h_conted_{t-1}
		    {
		      LayerParameter* w_param = net_param->add_layer();
		      w_param->CopyFrom(hidden_param);
			  w_param->mutable_inner_product_param()->set_axis(2);
		      w_param->set_name("transform_" + ts);
		      w_param->add_param()->set_name("W_hc");
		      w_param->add_bottom("h_conted_" + tm1s);
		      w_param->add_top("W_hc_h_" + tm1s); //(1,40,1024)
		    }

		    // Add the outputs of the linear transformations to compute the gate input.
		    //     gate_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c
		    //                   = W_hc_h_{t-1} + W_xc_x_t + b_c
		    {
		      LayerParameter* input_sum_layer = net_param->add_layer();
		      input_sum_layer->CopyFrom(sum_param);
		      input_sum_layer->set_name("gate_input_" + ts);
		      input_sum_layer->add_bottom("W_hc_h_" + tm1s);
		      input_sum_layer->add_bottom("W_xc_x_" + ts);

		      input_sum_layer->add_top("gate_input_" + ts); //(1,40,1024)
		    }
			========================================================*/

			// Add LSTMUnit layer to compute the cell & hidden vectors c_t and h_t.
			// Inputs: c_{t-1}, gate_input_t = (i_t, f_t, o_t, g_t), cont_t
			// Outputs: c_t, h_t
			//     [ i_t' ]
			//     [ f_t' ] := gate_input_t
			//     [ o_t' ]
			//     [ g_t' ]
			//         i_t := \sigmoid[i_t']
			//         f_t := \sigmoid[f_t']
			//         o_t := \sigmoid[o_t']
			//         g_t := \tanh[g_t']
			//         c_t := cont_t * (f_t .* c_{t-1}) + (i_t .* g_t)
			//         h_t := o_t .* \tanh[c_t]

			{
				LayerParameter* lstm_unit_param = net_param->add_layer();
				lstm_unit_param->set_type("LSTMUnit");
				lstm_unit_param->add_bottom("c_" + tm1s); //(1,40,256)
				lstm_unit_param->add_bottom("gate_input_" + ts);//(1,40,1024)
				lstm_unit_param->add_bottom("cont_0");//(1,40)
				lstm_unit_param->add_top("c_" + ts);//(1,40,256)
				lstm_unit_param->add_top("h_" + ts);//(1,40,256)
				lstm_unit_param->set_name("unit_" + ts);
			}

			{
				LayerParameter* predict_param = net_param->add_layer();
				predict_param->set_type("InnerProduct");
				predict_param->mutable_inner_product_param()->set_num_output(num_output);
				predict_param->mutable_inner_product_param()->set_bias_term(true);
				predict_param->mutable_inner_product_param()->set_axis(2);
				predict_param->mutable_inner_product_param()->
					mutable_weight_filler()->CopyFrom(weight_filler);
				predict_param->mutable_inner_product_param()->
					mutable_bias_filler()->CopyFrom(bias_filler);
				ParamSpec* param0 = predict_param->add_param();
				param0->set_name("predict_param_0");
				param0->set_lr_mult(1.0);
				param0->set_decay_mult(1.0);
				ParamSpec* param1 = predict_param->add_param();
				param1->set_name("predict_param_1");
				param1->set_lr_mult(2.0);
				param1->set_decay_mult(0.0);
				predict_param->set_name("predict_" + ts);
				predict_param->add_bottom("h_" + ts); //(1,40,256) 
				predict_param->add_top("predict_" + ts);     //(1,40,6863)
			}

			if (this->phase_ != TEST) {
				output_concat_layer.add_bottom("predict_" + ts);
			}
			else {
				//LayerParameter* predict_reshape_param = net_param->add_layer();
				//predict_reshape_param->set_type("Reshape");
				//predict_reshape_param->set_name("predict_reshape_" + ts);
				//BlobShape* new_shape = predict_reshape_param->mutable_reshape_param()->mutable_shape();
				//new_shape->Clear();
				//new_shape->add_dim(-1);
				//new_shape->add_dim(num_output);
				//predict_reshape_param->add_bottom("predict_" + ts);
				//predict_reshape_param->add_top("predict_reshape_" + ts);  ////(1,6863)

				LayerParameter* softmax_prob_param = net_param->add_layer();
				softmax_prob_param->set_type("Softmax");
				softmax_prob_param->mutable_softmax_param()->set_axis(2);
				softmax_prob_param->set_name("prob_" + ts);
				softmax_prob_param->add_bottom("predict_" + ts);
				softmax_prob_param->add_top("prob_" + ts); //(1,1,6863)

				output_concat_layer.add_bottom("prob_" + ts);
			}

			output_m_layer.add_bottom("att_weight_" + ts);
		}  // for (int t = 1; t <= this->T_; ++t)

		{
			LayerParameter* c_T_copy_param = net_param->add_layer();
			c_T_copy_param->CopyFrom(split_param);
			c_T_copy_param->add_bottom("c_" + format_int(this->T_));
			c_T_copy_param->add_top("c_T");
		}
		net_param->add_layer()->CopyFrom(output_concat_layer);
		net_param->add_layer()->CopyFrom(output_m_layer);
	}

	INSTANTIATE_CLASS(AttLSTMLayer);
	REGISTER_LAYER_CLASS(AttLSTM);

}  // namespace caffe

#endif