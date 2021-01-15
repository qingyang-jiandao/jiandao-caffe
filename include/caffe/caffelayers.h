#ifndef CAFFE_REG_H
#define CAFFE_REG_H
#include "caffe/common.hpp"  
#include "caffe/layers/input_layer.hpp"  
#include "caffe/layers/inner_product_layer.hpp"  
#include "caffe/layers/dropout_layer.hpp"  
#include "caffe/layers/conv_layer.hpp"  
#include "caffe/layers/relu_layer.hpp"  
#include "caffe/layers/pooling_layer.hpp"  
#include "caffe/layers/lrn_layer.hpp"  
#include "caffe/layers/softmax_layer.hpp"  
//#include "caffe/layers/normalize_layer.hpp"
#include "caffe/layers/permute_layer.hpp"
#include "caffe/layers/flatten_layer.hpp"
//#include "caffe/layers/prior_box_layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/reshape_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/argmax_layer.hpp"
//#include "caffe/layers/detection_output_layer.hpp"
#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/layers/scale_layer.hpp"
#include "caffe/layers/bias_layer.hpp"
#include "caffe/layers/interp_layer.hpp"
#include "caffe/layers/im2col_layer.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/layers/lstm_layer.hpp"
#include "caffe/layers/text_proposal_layer.hpp"
#include "caffe/st_layer.hpp"
#include "caffe/st_pt_layer.hpp"
#include "caffe/st_pt_op_layer.hpp"
#include "caffe/layers/deconv_layer.hpp"
#include "caffe/layers/upsample_layer.hpp"
#include "caffe/layers/yolov3_layer.hpp"
#include "caffe/layers/yolov3_detection_output_layer.hpp"
#include "caffe/layers/detection_evaluate_layer.hpp"
#include "caffe/layers/yolov3_proposaltarget_layer.hpp"
#include "caffe/layers/proposal_layer.hpp"
#include "caffe/layers/roi_pooling_layer.hpp"
#include "caffe/layers/slice_layer.hpp"
#include "caffe/layers/relu6_layer.hpp"

namespace caffe
{
	extern INSTANTIATE_CLASS(InputLayer);
	extern INSTANTIATE_CLASS(ConvolutionLayer);
	//REGISTER_LAYER_CLASS(Convolution);
	extern INSTANTIATE_CLASS(InnerProductLayer);
	extern INSTANTIATE_CLASS(DropoutLayer);
	extern INSTANTIATE_CLASS(ReLULayer);
	//REGISTER_LAYER_CLASS(ReLU);
	extern INSTANTIATE_CLASS(PoolingLayer);
	//REGISTER_LAYER_CLASS(Pooling);
	extern INSTANTIATE_CLASS(LRNLayer);//REGISTER_LAYER_CLASS(LRN);
	//extern INSTANTIATE_CLASS(NormalizeLayer);//REGISTER_LAYER_CLASS(Normalize);
	extern INSTANTIATE_CLASS(PermuteLayer);//REGISTER_LAYER_CLASS(Permute);
	extern INSTANTIATE_CLASS(FlattenLayer);
	//extern INSTANTIATE_CLASS(PriorBoxLayer);
	extern INSTANTIATE_CLASS(ConcatLayer);
	extern INSTANTIATE_CLASS(ReshapeLayer);
	extern INSTANTIATE_CLASS(SoftmaxLayer);
	//REGISTER_LAYER_CLASS(Softmax);
	extern INSTANTIATE_CLASS(ArgMaxLayer);
	//extern INSTANTIATE_CLASS(DetectionOutputLayer);
	extern INSTANTIATE_CLASS(BatchNormLayer);
	extern INSTANTIATE_CLASS(ScaleLayer);
	extern INSTANTIATE_CLASS(BiasLayer);
	extern INSTANTIATE_CLASS(InterpLayer);
	extern INSTANTIATE_CLASS(Im2colLayer);
	extern INSTANTIATE_CLASS(TransposeLayer);
	extern INSTANTIATE_CLASS(LstmLayer);
	extern INSTANTIATE_CLASS(ReverseLayer);
	extern INSTANTIATE_CLASS(TextProposalLayer);
	extern INSTANTIATE_CLASS(SpatialTransformerLayer);
	extern INSTANTIATE_CLASS(DeconvolutionLayer);
	extern INSTANTIATE_CLASS(SpatialTransformerPTLayer);
	extern INSTANTIATE_CLASS(SpatialTransformerPTOPLayer);
	extern INSTANTIATE_CLASS(UpsampleLayer);
	extern INSTANTIATE_CLASS(ProposalLayer);
	extern INSTANTIATE_CLASS(ROIPoolingLayer);
	extern INSTANTIATE_CLASS(SliceLayer);
	extern INSTANTIATE_CLASS(Yolov3Layer);
	extern INSTANTIATE_CLASS(Yolov3DetectionOutputLayer);
	extern INSTANTIATE_CLASS(DetectionEvaluateLayer);
	extern INSTANTIATE_CLASS(Yolov3ProposalTargetLayer);
	extern INSTANTIATE_CLASS(ReLU6Layer);
}

#endif