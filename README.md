# jiandao-caffe
Jiandao Caffe is a modified version of the popular Caffe Deep Learning framework adapted for use with DesignWare EV6x Processors. It combines multiple customized branches and includes a large range of patches to support diverse models. See FEATURES.md for a short overview.


### Installation
Please check out the prerequisites and read the detailed notes at the [BVLC Caffe Installation](http://caffe.berkeleyvision.org/installation.html) if this is your first time to install Caffe.

#### Linux
A simple guide:
1. Ensure that you have all the dependencies mentioned at the [BVLC Caffe Installation](http://caffe.berkeleyvision.org/installation.html) for your OS installed (protobuf, leveldb, snappy, opencv, hdf5-serial, protobuf-compiler, BLAS, Python, CUDA etc.)
2. Checkout the Jiandao Caffe **main** branch. Configure the build by copying and modifying the example Makefile.config for your setup.
```Shell
git clone https://github.com/qingyang-jiandao/jiandao-caffe.git
cd jiandao-caffe
cp Makefile.config.example Makefile.config
# Modify Makefile.config to suit your needs, e.g. enable/disable the CPU-ONLY, CUDNN, NCCL and set the path for CUDA, Python and BLAS.
# If needed, add [your installed matio path]/include to INCLUDE_DIRS and [your installed matio path]/lib to LIBRARY_DIRS.
```
3. Build Caffe and run the tests.
```Shell
make -j4 && make pycaffe
```

#### Windows
A simple guide:
1. Download the **Visual Studio 2015** (or VS 2017). Choose to install the support for visual C++ instead of applying the default settings.
2. Install the CMake 3.4 or higher. Install Python 2.7 or 3.5/3.6. Add cmake.exe and python.exe to your PATH.
3. After installing the Python, please open a `cmd` prompt and use `pip install numpy` to install the **numpy** package.
4. Checkout the Jiandao Caffe **master** branch for build. The windows branch is deprecated, please do not use it. We use `C:\Projects` as the current folder for the following instructions.
5. Edit any of the options inside **jiandao-caffe\scripts\build_win.cmd** to suit your needs, such as settings for Python version, CUDA/CuDNN enabling etc.   
```cmd
C:\Projects> git clone https://github.com/qingyang-jiandao/jiandao-caffe.git
C:\Projects> cd jiandao-caffe
C:\Projects\jiandao-caffe> build_win.cmd or build_win_vs2015.cmd
:: If no error occurs, the caffe.exe will be created at C:\Projects\jiandao-caffe\build\tools\Release after a successful build.
```
Other detailed installation instructions can be found [here](https://github.com/BVLC/caffe/blob/windows/README.md).

