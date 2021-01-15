//
// This script converts the CIFAR dataset to the leveldb format used
// by caffe to perform classification.
// Usage:
//    convert_cifar_data input_folder output_db_file
// The CIFAR dataset could be downloaded at
//    http://www.cs.toronto.edu/~kriz/cifar.html

#include <fcntl.h>
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#if defined(_MSC_VER)
#include <io.h>
#endif

#include "boost/scoped_ptr.hpp"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "stdint.h"

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"

#define MAX_IMAGE_COUNT 10

using caffe::Datum;
using boost::scoped_ptr;
using std::string;
namespace db = caffe::db;

bool WriteDatumToFile(const string& filename,
	Datum* datum) {
	std::streampos size;
	const string& data = datum->data();
	std::fstream file(filename.c_str(), std::ios::out | std::ios::binary | std::ios::ate);
	if (file.is_open()) {
		file.write(data.c_str(),
			data.size());
		return true;
	}
	else {
		return false;
	}
}

//不同的通道对应不同的featuremap
cv::Mat DatumToCVMat(const Datum* datum) {
	cv::Mat cv_img;
	int datum_channels = datum->channels();
	int datum_height = datum->height();
	int datum_width = datum->width();
	int datum_size = datum_channels * datum_height * datum_width;
	const string& data = datum->data();
	std::vector<uchar> vec_data;
	for (int h = 0; h < datum_height; ++h) {
		for (int w = 0; w < datum_width; ++w) {
			for (int c = 0; c < datum_channels; ++c) {
				int datum_index = (c * datum_height + h) * datum_width + w;
				vec_data.push_back(data[datum_index]);
			}
		}
	}

	switch (datum_channels)
	{
	case 1:
		cv_img = cv::Mat(datum_height, datum_width, CV_8UC1, (void*)&vec_data[0]);
		break;
	case 2:
		cv_img = cv::Mat(datum_height, datum_width, CV_8UC2, (void*)&vec_data[0]);
		break;
	case 3:
		cv_img = cv::Mat(datum_height, datum_width, CV_8UC3, (void*)&vec_data[0]);
		break;
	case 4:
		cv_img = cv::Mat(datum->height(), datum_width, CV_8UC4, (void*)&vec_data[0]);
		break;
	default:
		printf("datum_channels:%d.\n", datum_channels);
		break;
	}

	return cv_img;
}

static void StringReplace(const string& s, const string& oldsub,
	const string& newsub, bool replace_all,
	string* res) {
	if (oldsub.empty()) {
		res->append(s);  // if empty, append the given string.
		return;
	}

	string::size_type start_pos = 0;
	string::size_type pos;
	do {
		pos = s.find(oldsub, start_pos);
		if (pos == string::npos) {
			break;
		}
		res->append(s, start_pos, pos - start_pos);
		res->append(newsub);
		start_pos = pos + oldsub.size();  // start searching again after the "old"
	} while (replace_all);
	res->append(s, start_pos, s.length() - start_pos);
}

void convert_db_data_to_imageset(const string& input_folder, const string& output_folder,
    const string& db_type, const string& filename) {

  printf("convert_db_data_to_imageset start.\n");

  scoped_ptr<db::DB> proccessing_db(db::GetDB(db_type));
  proccessing_db->Open(input_folder, db::READ);
  scoped_ptr<db::Transaction> txn(proccessing_db->NewTransaction());
  scoped_ptr<db::Cursor> cursor(proccessing_db->NewCursor());

  const string extend_name = ".jpg";
  char label_buf[10];
  std::fstream file(filename.c_str(), std::ios::out | std::ios::binary | std::ios::ate);
  int count = 0;
  while (cursor->valid()/* && count<MAX_IMAGE_COUNT*/)
  {
	  string key = cursor->key();
	  Datum datum;

	  string basename = key;
	  string::size_type start_pos = 0;
	  string::size_type pos;
	  pos = key.rfind("/");
	  if (pos!= string::npos) {
		  basename = key.substr(pos+1);
	  }

	  string saveName = output_folder + basename;
	  //datum.ParseFromString(cursor->value());
	  //WriteDatumToFile(saveName, &datum);
	 
	  datum.ParseFromString(cursor->value());
	  DecodeDatumNative(&datum);
	  printf("imagename:%s channels:%d height:%d width:%d\n", key.c_str(), datum.channels(), datum.height(), datum.width());
	  cv::Mat cv_img = DatumToCVMat(&datum);

      cv::imwrite(saveName, cv_img);
	  if (file.is_open()) {
		  int label = datum.label();
		  snprintf(label_buf, 10, "%d\n", label);
		  std::string line = saveName + std::string(" ") + std::string(label_buf);

		  file.write(line.c_str(),
			  line.size());
	  }
	  cursor->Next();
	  ++count;
  }

  //while (cursor->valid()) {
	 // Datum datum;
	 // datum.ParseFromString(cursor->value());
	 // DecodeDatumNative(&datum);
	 // const int data_size = datum.channels() * datum.height() * datum.width();
	 // const std::string& data = datum.data();
	 // size_in_datum = std::max<int>(datum.data().size(), datum.float_data_size());
	 // CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " << size_in_datum;
	 // if (data.size() != 0) {
		//  CHECK_EQ(data.size(), size_in_datum);
		//  for (int i = 0; i < size_in_datum; ++i) {
		//	  sum_blob.set_data(i, sum_blob.data(i) + (uint8_t)data[i]);
		//  }
	 // }
	 // else {
		//  CHECK_EQ(datum.float_data_size(), size_in_datum);
		//  for (int i = 0; i < size_in_datum; ++i) {
		//	  sum_blob.set_data(i, sum_blob.data(i) +
		//		  static_cast<float>(datum.float_data(i)));
		//  }
	 // }
	 // ++count;
	 // if (count % 10000 == 0) {
		//  LOG(INFO) << "Processed " << count << " files.";
	 // }

	 // break;
	 // cursor->Next();
  //}

  printf("convert_db_data_to_imageset done.\n");
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = 1;

  if (argc != 5) {
    printf("This script converts the CIFAR dataset to the leveldb format used\n"
           "by caffe to perform classification.\n"
           "Usage:\n"
           "    convert_cifar_data input_folder output_folder db_type\n"
           "Where the input folder should contain the binary batch files.\n"
           "The CIFAR dataset could be downloaded at\n"
           "    http://www.cs.toronto.edu/~kriz/cifar.html\n"
           "You should gunzip them after downloading.\n");
  } else {
    google::InitGoogleLogging(argv[0]);
	convert_db_data_to_imageset(string(argv[1]), string(argv[2]), string(argv[3]), string(argv[4]));
  }
  return 0;
}
