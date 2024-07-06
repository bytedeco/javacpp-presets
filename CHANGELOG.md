
 * Enable `opencv_python3` module for `macosx-arm64` as well ([pull #1517](https://github.com/bytedeco/javacpp-presets/pull/1517))
 * Introduce `macosx-arm64` builds for CPython ([pull #1511](https://github.com/bytedeco/javacpp-presets/pull/1511)), NumPy ([pull #1515](https://github.com/bytedeco/javacpp-presets/pull/1515)), SciPy ([pull #1516](https://github.com/bytedeco/javacpp-presets/pull/1516))
 * Update and fix the sample code of the presets for LLVM ([pull #1501](https://github.com/bytedeco/javacpp-presets/pull/1501))
 * Fix Vulkan GPU acceleration for FFmpeg ([pull #1497](https://github.com/bytedeco/javacpp-presets/pull/1497))
 * Build FFmpeg with zimg to enable zscale filter ([pull #1481](https://github.com/bytedeco/javacpp-presets/pull/1481))
 * Enable PulseAudio support for FFmpeg on Linux ([pull #1472](https://github.com/bytedeco/javacpp-presets/pull/1472))
 * Virtualize `btCollisionWorld`, `btOverlapFilterCallback`, `btOverlapCallback` from Bullet Physics SDK ([pull #1475](https://github.com/bytedeco/javacpp-presets/pull/1475))
 * Upgrade presets for OpenCV 4.10.0, FFmpeg 7.0, DNNL 3.4.1, OpenBLAS 0.3.27, CMINPACK 1.3.9, GSL 2.8, CPython 3.12.4, NumPy 2.0.0, SciPy 1.14.0, LLVM 18.1.4, Tesseract 5.4.1, libffi 3.4.6, PyTorch 2.3.0 ([pull #1466](https://github.com/bytedeco/javacpp-presets/pull/1466)), SentencePiece 0.2.0, TensorFlow Lite 2.16.1, TensorRT 10.0.1.6, Triton Inference Server 2.44.0, ONNX 1.16.1, ONNX Runtime 1.18.0, TVM 0.16.0, and their dependencies

### January 29, 2024 version 1.5.10
 * Introduce `macosx-arm64` builds for PyTorch ([pull #1463](https://github.com/bytedeco/javacpp-presets/pull/1463))
 * Reenable `linux-arm64` builds for CPython and NumPy ([pull #1386](https://github.com/bytedeco/javacpp-presets/pull/1386))
 * Enable Vulkan GPU acceleration for FFmpeg ([pull #1460](https://github.com/bytedeco/javacpp-presets/pull/1460))
 * Include `timeapi.h` for system API of Windows ([pull #1447](https://github.com/bytedeco/javacpp-presets/pull/1447))
 * Add Android and Windows builds to presets for DepthAI ([pull #1441](https://github.com/bytedeco/javacpp-presets/pull/1441))
 * Add presets for nvCOMP 3.0.5 ([pull #1434](https://github.com/bytedeco/javacpp-presets/pull/1434)), SentencePiece 0.1.99 ([pull #1384](https://github.com/bytedeco/javacpp-presets/pull/1384))
 * Refactor and improve presets for PyTorch ([pull #1360](https://github.com/bytedeco/javacpp-presets/pull/1360))
 * Include `mkl_lapack.h` header file in presets for MKL ([issue #1388](https://github.com/bytedeco/javacpp-presets/issues/1388))
 * Map new higher-level C++ API of Triton Inference Server ([pull #1361](https://github.com/bytedeco/javacpp-presets/pull/1361))
 * Upgrade presets for OpenCV 4.9.0, FFmpeg 6.1.1, HDF5 1.14.3, MKL 2024.0, DNNL 3.3.4, OpenBLAS 0.3.26, ARPACK-NG 3.9.1, CPython 3.12.1, NumPy 1.26.3, SciPy 1.12.0, LLVM 17.0.6, Leptonica 1.84.1, Tesseract 5.3.4, CUDA 12.3.2, cuDNN 8.9.7, NCCL 2.19.3, OpenCL 3.0.15, PyTorch 2.1.2 ([pull #1426](https://github.com/bytedeco/javacpp-presets/pull/1426)), TensorFlow Lite 2.15.0, Triton Inference Server 2.41.0, DepthAI 2.24.0, ONNX 1.15.0, ONNX Runtime 1.16.3, TVM 0.14.0, and their dependencies

### June 6, 2023 version 1.5.9
 * Virtualize `nvinfer1::IGpuAllocator` from TensorRT to allow customization ([pull #1367](https://github.com/bytedeco/javacpp-presets/pull/1367))
 * Add new `SampleJpegDecoder` and `SampleJpegEncoder` code for nvJPEG module of CUDA ([pull #1365](https://github.com/bytedeco/javacpp-presets/pull/1365))
 * Map `std::vector` of `CameraParams`, `ImageFeatures`, and `MatchesInfo` from `cv::detail` ([issue bytedeco/javacv#2027](https://github.com/bytedeco/javacv/issues/2027))
 * Fix H.264 decoder of FFmpeg by increasing MAX_SLICES to 256 ([pull #1349](https://github.com/bytedeco/javacpp-presets/pull/1349))
 * Link FFmpeg with latest version of VA-API libraries ([pull #1296](https://github.com/bytedeco/javacpp-presets/pull/1296))
 * Build HDF5 with support for SZIP enabled ([pull #1334](https://github.com/bytedeco/javacpp-presets/pull/1334))
 * Map missing functions from `mkl_trans.h` in presets for MKL ([issue #1331](https://github.com/bytedeco/javacpp-presets/issues/1331))
 * Bundle the official Java API of HDF5 via the `hdf5_java` library ([pull #1327](https://github.com/bytedeco/javacpp-presets/pull/1327))
 * Map missing `cblas_?axpby()` functions in presets for MKL ([issue #1326](https://github.com/bytedeco/javacpp-presets/issues/1326))
 * Prefix with "fisheye" all functions from the `cv::fisheye::` namespace to avoid collisions ([pull #1324](https://github.com/bytedeco/javacpp-presets/pull/1324))
 * Remove mapping for platform-dependent `enum` values in presets for libffi ([pull #1318](https://github.com/bytedeco/javacpp-presets/pull/1318))
 * Fix mapping of `cv::fisheye::calibrate()` function from `opencv_calib3d` ([issue #1185](https://github.com/bytedeco/javacpp-presets/issues/1185))
 * Add an RPATH to the `tesseract` program to avoid loading issues ([issue #1314](https://github.com/bytedeco/javacpp-presets/issues/1314))
 * Bundle the CUPTI module of CUDA in the presets for PyTorch ([pull #1307](https://github.com/bytedeco/javacpp-presets/pull/1307))
 * Build FFmpeg with AOMedia AV1 and SVT-AV1 codecs ([pull #1303](https://github.com/bytedeco/javacpp-presets/pull/1303))
 * Map `c10::OptionalArrayRef<int64_t>` from PyTorch to `long...` as well for convenience ([issue #1300](https://github.com/bytedeco/javacpp-presets/issues/1300))
 * Remove mapping for `c10::OptionalArrayRef` to simplify calls in presets for PyTorch ([issue #1300](https://github.com/bytedeco/javacpp-presets/issues/1300))
 * Virtualize `btMotionState` and `btDefaultMotionState` from Bullet Physics SDK to allow callbacks ([pull #1297](https://github.com/bytedeco/javacpp-presets/pull/1297))
 * Define `STRING_BYTES_CHARSET` to "UTF-8" for FFmpeg since it appears to assume that ([issue bytedeco/javacv#1945](https://github.com/bytedeco/javacv/issues/1945))
 * Map `at::ITensorListRef` as used by `at::cat()` in presets for PyTorch ([issue #1293](https://github.com/bytedeco/javacpp-presets/issues/1293))
 * Map `torch::data::datasets::ChunkDataReader` and related data loading classes from PyTorch ([issue #1215](https://github.com/bytedeco/javacpp-presets/issues/1215))
 * Add missing predefined `AVChannelLayout` in presets for FFmpeg ([issue #1286](https://github.com/bytedeco/javacpp-presets/issues/1286))
 * Map `c10::impl::GenericDict` as returned by `c10::IValue::toGenericDict()` in presets for PyTorch
 * Introduce `linux-armhf` and `linux-x86` builds to presets for TensorFlow Lite ([pull #1268](https://github.com/bytedeco/javacpp-presets/pull/1268))
 * Add presets for LibRaw 0.21.1 ([pull #1211](https://github.com/bytedeco/javacpp-presets/pull/1211))
 * Upgrade presets for OpenCV 4.7.0, FFmpeg 6.0 ([issue bytedeco/javacv#1693](https://github.com/bytedeco/javacv/issues/1693)), HDF5 1.14.1, Hyperscan 5.4.2 ([issue #1308](https://github.com/bytedeco/javacpp-presets/issues/1308)), Spinnaker 3.0.0.118 ([pull #1313](https://github.com/bytedeco/javacpp-presets/pull/1313)), librealsense2 2.53.1 ([pull #1305](https://github.com/bytedeco/javacpp-presets/pull/1305)), MKL 2023.1, DNNL 3.1, OpenBLAS 0.3.23, ARPACK-NG 3.9.0, CPython 3.11.3, NumPy 1.24.3, SciPy 1.10.1, LLVM 16.0.4, Leptonica 1.83.0, Tesseract 5.3.1, CUDA 12.1.1, cuDNN 8.9.1, NCCL 2.18.1, OpenCL 3.0.14, NVIDIA Video Codec SDK 12.1.14, PyTorch 2.0.1, TensorFlow Lite 2.12.0, TensorRT 8.6.1.6, Triton Inference Server 2.33.0, DepthAI 2.21.2, ONNX 1.14.0, ONNX Runtime 1.15.0, TVM 0.12.0, Bullet Physics SDK 3.25, and their dependencies

### November 2, 2022 version 1.5.8
 * Fix mapping of `torch::ExpandingArrayWithOptionalElem` in presets for PyTorch ([issue #1250](https://github.com/bytedeco/javacpp-presets/issues/1250))
 * Disable OpenMP for Tesseract to work around performance issues ([pull #1201](https://github.com/bytedeco/javacpp-presets/pull/1201))
 * Enable NVIDIA GPU acceleration for FFmpeg on ARM and POWER as well ([issue bytedeco/javacv#1894](https://github.com/bytedeco/javacv/issues/1894))
 * Add support for HEVC and Opus to FLV format in presets for FFmpeg ([pull #1228](https://github.com/bytedeco/javacpp-presets/pull/1228))
 * Build Leptonica with OpenJPEG for JPEG 2000 support
 * Introduce `linux-arm64` and `macosx-arm64` builds to presets for libpostal ([pull #1199](https://github.com/bytedeco/javacpp-presets/pull/1199) and [pull #1205](https://github.com/bytedeco/javacpp-presets/pull/1205))
 * Map missing factory functions in `torch::` namespace using `torch_` as prefix in presets for PyTorch ([issue #1197](https://github.com/bytedeco/javacpp-presets/issues/1197))
 * Add presets for the nvJPEG module of CUDA ([issue #1193](https://github.com/bytedeco/javacpp-presets/issues/1193))
 * Introduce Android builds for TensorFlow Lite ([discussion #1180](https://github.com/bytedeco/javacpp-presets/discussions/1180))
 * Map `std::vector<cv::Ptr<cv::mcc::CChecker> >` for `CCheckerDetector.getListColorChecker()` ([issue bytedeco/javacpp#571](https://github.com/bytedeco/javacpp/issues/571))
 * Include missing `opencv2/mcc/ccm.hpp` header file in presets for OpenCV ([discussion bytedeco/javacpp#568](https://github.com/bytedeco/javacpp/discussions/568))
 * Fix a few incorrectly mapped instances of `std::unordered_map` for PyTorch ([issue #1164](https://github.com/bytedeco/javacpp-presets/issues/1164))
 * Migrate builds for Leptonica and Tesseract to CMake ([issue #1163](https://github.com/bytedeco/javacpp-presets/issues/1163))
 * Introduce `macosx-arm64` builds for LZ4 ([pull #1243](https://github.com/bytedeco/javacpp-presets/pull/1243)), libffi ([issue #1182](https://github.com/bytedeco/javacpp-presets/issues/1182)), Leptonica, and Tesseract ([issue #814](https://github.com/bytedeco/javacpp-presets/issues/814))
 * Map instances of `torch::OrderedDict` using C++ templates from PyTorch ([issue #623](https://github.com/bytedeco/javacpp-presets/issues/623))
 * Add presets for Bullet Physics SDK 3.24 ([pull #1153](https://github.com/bytedeco/javacpp-presets/pull/1153))
 * Add `long[] pytorch.Tensor.shape()` method for convenience ([pull #1161](https://github.com/bytedeco/javacpp-presets/pull/1161))
 * Enable DNNL codegen as BYOC backend in presets for TVM
 * Allow passing raw pointer as deleter to `from_blob()`, etc functions of PyTorch ([discussion #1160](https://github.com/bytedeco/javacpp-presets/discussions/1160))
 * Include `cudnn_backend.h` header file in presets for CUDA ([issue #1158](https://github.com/bytedeco/javacpp-presets/issues/1158))
 * Bundle `zlibwapi.dll` required by cuDNN in redist artifacts of presets for CUDA ([issue bytedeco/javacv#1767](https://github.com/bytedeco/javacv/issues/1767))
 * Harmonize string and buffer pointer types of function parameters from DepthAI ([issue #1155](https://github.com/bytedeco/javacpp-presets/issues/1155))
 * Bundle correctly OpenMP library for PyTorch builds on Mac as well ([issue #1225](https://github.com/bytedeco/javacpp-presets/issues/1225))
 * Remove dependency on CUDA from presets for Triton Inference Server ([pull #1151](https://github.com/bytedeco/javacpp-presets/pull/1151))
 * Disable signal handlers of DepthAI known to cause issues with the JDK ([issue #1118](https://github.com/bytedeco/javacpp-presets/issues/1118))
 * Upgrade presets for OpenCV 4.6.0, FFmpeg 5.1.2, HDF5 1.12.2, LZ4 1.9.4, MKL 2022.2, DNNL 2.7.1, OpenBLAS 0.3.21 ([issue #1171](https://github.com/bytedeco/javacpp-presets/issues/1171)), CPython 3.10.8, NumPy 1.23.4, SciPy 1.9.3, Gym 0.26.2, LLVM 15.0.3, libffi 3.4.4, Tesseract 5.2.0, CUDA 11.8.0, cuDNN 8.6.0, NCCL 2.15.5, OpenCL 3.0.12, MXNet 1.9.1, PyTorch 1.12.1, TensorFlow Lite 2.10.0, TensorRT 8.4.3.1, Triton Inference Server 2.26.0, ALE 0.8.0, DepthAI 2.18.0, ONNX 1.12.0, ONNX Runtime 1.13.1, TVM 0.10.0, Skia 2.88.3, cpu_features 0.7.0, ModSecurity 3.0.8, and their dependencies

### February 11, 2022 version 1.5.7
 * Build FFmpeg with WebP encoding support ([pull #1133](https://github.com/bytedeco/javacpp-presets/pull/1133))
 * Include `sys/ipc.h` and `sys/shm.h` for system APIs of Linux and Mac OS X ([pull #1132](https://github.com/bytedeco/javacpp-presets/pull/1132))
 * Map `c10::ArrayRef<at::Tensor>(std::vector<at::Tensor>&)` constructor from PyTorch for convenience ([discussion #1128](https://github.com/bytedeco/javacpp-presets/discussions/1128))
 * Add `long rs2_get_frame_data_address()` to reduce garbage for real-time applications using librealsense2 ([discussion bytedeco/javacpp#532](https://github.com/bytedeco/javacpp/discussions/532))
 * Add to `torch.Tensor` convenient `create()`, `createBuffer()`, and `createIndexer()` factory methods for PyTorch
 * Upgrade requirements to Android 7.0 for camera support in OpenCV and FFmpeg ([issue bytedeco/javacv#1692](https://github.com/bytedeco/javacv/issues/1692))
 * Include new `llvm-c/Transforms/PassBuilder.h` header file in presets for LLVM ([pull #1093](https://github.com/bytedeco/javacpp-presets/pull/1093))
 * Introduce `macosx-arm64` builds to presets for OpenCV, FFmpeg, OpenBLAS ([issue #1069](https://github.com/bytedeco/javacpp-presets/issues/1069)), LLVM ([pull #1092](https://github.com/bytedeco/javacpp-presets/pull/1092))
 * Add presets for LZ4 1.9.3 ([pull #1094](https://github.com/bytedeco/javacpp-presets/pull/1094)), Triton Inference Server 2.18.0 ([pull #1085](https://github.com/bytedeco/javacpp-presets/pull/1085))
 * Add presets for the NvToolsExt (NVTX) module of CUDA ([issue #1068](https://github.com/bytedeco/javacpp-presets/issues/1068))
 * Increase the amount of function pointers available for callbacks in presets for Qt ([pull #1080](https://github.com/bytedeco/javacpp-presets/pull/1080))
 * Map C++ JIT classes and functions of TorchScript in presets for PyTorch ([issue #1068](https://github.com/bytedeco/javacpp-presets/issues/1068))
 * Synchronize `cachePackage()` and prevent repeated package caching in all presets ([pull #1071](https://github.com/bytedeco/javacpp-presets/pull/1071))
 * Build FFmpeg with VA-API enabled and bundle its libraries to avoid loading issues ([issue bytedeco/javacv#1188](https://github.com/bytedeco/javacv/issues/1188))
 * Upgrade presets for OpenCV 4.5.5, FFmpeg 5.0 ([pull #1125](https://github.com/bytedeco/javacpp-presets/pull/1125)), librealsense2 2.50.0, Arrow 6.0.1, MKL 2022.0, DNNL 2.5.2, OpenBLAS 0.3.19, FFTW 3.3.10, CPython 3.10.2, NumPy 1.22.2, SciPy 1.8.0, Gym 0.21.0, LLVM 13.0.1, libpostal 1.1, Leptonica 1.82.0, Tesseract 5.0.1, CUDA 11.6.0, cuDNN 8.3.2, NCCL 2.11.4, MXNet 1.9.0, PyTorch 1.10.2, TensorFlow Lite 2.8.0, TensorRT 8.2.3.0, ALE 0.7.3, DepthAI 2.14.1, ONNX 1.10.2, ONNX Runtime 1.10.0, TVM 0.8.0, ModSecurity 3.0.6, and their dependencies

### August 2, 2021 version 1.5.6
 * Change `opencv_core.Mat` constructors to create column vectors out of arrays for consistency ([issue #1064](https://github.com/bytedeco/javacpp-presets/issues/1064))
 * Add presets for the new `barcode` and `wechat_qrcode` modules of OpenCV
 * Work around loading issues with execution providers in presets for ONNX Runtime
 * Annotate the presets for LLVM with `@NoException` to reduce unneeded C++ overhead ([pull #1052](https://github.com/bytedeco/javacpp-presets/pull/1052))
 * Update samples for LLVM 12 including new `samples/llvm/OrcJit.java` using libffi ([pull #1050](https://github.com/bytedeco/javacpp-presets/pull/1050))
 * Enable GTK support in presets for OpenCV when building on ARM as well
 * Correct `enum` classes in presets for Spinnaker ([pull #1048](https://github.com/bytedeco/javacpp-presets/pull/1048))
 * Add Windows build for ONNX ([issue #983](https://github.com/bytedeco/javacpp-presets/issues/983))
 * Add `linux-arm64` builds to presets for DNNL, OpenCL, TensorRT ([pull #1044](https://github.com/bytedeco/javacpp-presets/pull/1044)), and ONNX Runtime
 * Build FFmpeg with libxml2, enabling support for DASH demuxing ([pull #1033](https://github.com/bytedeco/javacpp-presets/pull/1033)), and libsrt for SRT protocol support ([pull #1036](https://github.com/bytedeco/javacpp-presets/pull/1036))
 * Add `@MemberGetter` for `av_log_default_callback()` in presets for FFmpeg ([issue #812](https://github.com/bytedeco/javacpp-presets/issues/812))
 * Include `cudaGL.h` and `cuda_gl_interop.h` header files in presets for CUDA ([pull #1027](https://github.com/bytedeco/javacpp-presets/pull/1027))
 * Add presets for libffi 3.4.2 ([issue #833](https://github.com/bytedeco/javacpp-presets/issues/833)), NVIDIA Video Codec SDK 11.1.5 ([pull #1020](https://github.com/bytedeco/javacpp-presets/pull/1020)), PyTorch 1.9.0 ([issue #623](https://github.com/bytedeco/javacpp-presets/issues/623)), TensorFlow Lite 2.5.0, DepthAI 2.8.0, ModSecurity 3.0.5 ([pull #1012](https://github.com/bytedeco/javacpp-presets/pull/1012))
 * Map `std::vector<cv::Range>` to `RangeVector` in `opencv_core.Mat` for convenience ([issue bytedeco/javacv#1607](https://github.com/bytedeco/javacv/issues/1607))
 * Include `genericaliasobject.h`, `context.h`, `tracemalloc.h`, and `datetime.h` for CPython ([issue #1017](https://github.com/bytedeco/javacpp-presets/issues/1017))
 * Add samples using LLVM modules to deal with bitcode and object files ([pull #1016](https://github.com/bytedeco/javacpp-presets/pull/1016))
 * Upgrade presets for OpenCV 4.5.3, FFmpeg 4.4 ([pull #1030](https://github.com/bytedeco/javacpp-presets/pull/1030)), Spinnaker 2.4.0.143 ([pull #1040](https://github.com/bytedeco/javacpp-presets/pull/1040)), librealsense2 2.44.0 ([pull #1031](https://github.com/bytedeco/javacpp-presets/pull/1031)), Arrow 4.0.1, HDF5 1.12.1, MKL 2021.3, DNNL 2.3, OpenBLAS 0.3.17, GSL 2.7, CPython 3.9.6, NumPy 1.21.1, SciPy 1.7.0, Gym 0.18.3, LLVM 12.0.1 ([pull #1065](https://github.com/bytedeco/javacpp-presets/pull/1065)), Leptonica 1.81.1, CUDA 11.4.0, cuDNN 8.2.2, NCCL 2.10.3, TensorRT 8.0.1.6, ONNX 1.9.0, ONNX Runtime 1.8.1, Skia 2.80.3, and their dependencies

### March 8, 2021 version 1.5.5
 * Bundle LLD executable in presets for LLVM as required by TVM on Windows
 * Prevent `public static final` objects from getting deallocated by `PointerScope` ([issue bytedeco/javacv#1599](https://github.com/bytedeco/javacv/issues/1599))
 * Fix compatibility of Leptonica with JavaFX by upgrading to libpng 1.6.37 ([pull #1007](https://github.com/bytedeco/javacpp-presets/pull/1007))
 * Introduce `linux-arm64` build for CUDA, cuDNN, and NCCL ([issue #735](https://github.com/bytedeco/javacpp-presets/issues/735))
 * Add new array constructors to `opencv_core.Mat` that copy data for convenience ([pull #1002](https://github.com/bytedeco/javacpp-presets/pull/1002))
 * Rebase `PrimitiveScalar` on `PrimitiveScalarBase` in presets for Arrow for easy access to `data()` ([issue #998](https://github.com/bytedeco/javacpp-presets/issues/998))
 * Add `NamedMetadataOperations.h` implementing data retrieval operations for LLVM nodes ([pull #995](https://github.com/bytedeco/javacpp-presets/pull/995))
 * Enable OpenMP for ONNX Runtime on Mac once again ([issue #917](https://github.com/bytedeco/javacpp-presets/issues/917))
 * Build OpenCV without OpenBLAS when environment variable `NOOPENBLAS=yes` ([pull #987](https://github.com/bytedeco/javacpp-presets/pull/987))
 * Enable OpenCL acceleration for DNNL ([issue #938](https://github.com/bytedeco/javacpp-presets/issues/938))
 * Introduce monkey patching when loading presets for CPython to relocate home more reliably
 * Add display sample for librealsense2 ([pull #978](https://github.com/bytedeco/javacpp-presets/pull/978))
 * Fix builds for libpostal on Mac and Windows ([issue #903](https://github.com/bytedeco/javacpp-presets/issues/903))
 * Fix builds for NumPy and SciPy on Linux when using a cross compiler or not
 * Update presets for Visual Studio 2019 on Windows
 * Add presets for OpenCL 3.0, TVM 0.7.0 and bundle its official Java API (TVM4J) via the `jnitvm_runtime` library
 * Include `free()` in presets for FTTW as required by `fftw_export_wisdom_to_string()` ([issue bytedeco/javacpp#429](https://github.com/bytedeco/javacpp/issues/429))
 * Include all missing header files from the `opencv_ximgproc` module ([issue #958](https://github.com/bytedeco/javacpp-presets/issues/958))
 * Disable assembly optimizations for libx264 with FFmpeg on Mac to work around crashes ([issue bytedeco/javacv#1519](https://github.com/bytedeco/javacv/issues/1519))
 * Add `linux-armhf` and `linux-arm64` builds for librealsense and librealsense2 ([pull #951](https://github.com/bytedeco/javacpp-presets/pull/951))
 * License default builds of FFmpeg under LGPL v3 and move GPL-enabled builds to `-gpl` extension ([pull #950](https://github.com/bytedeco/javacpp-presets/pull/950))
 * Upgrade presets for OpenCV 4.5.1, FFmpeg 4.3.2, Arrow 3.0.0, Hyperscan 5.4.0, MKL 2021.1, OpenBLAS 0.3.13, ARPACK-NG 3.8.0, CMINPACK 1.3.8, FFTW 3.3.9, librealsense2 2.40.0 ([pull #946](https://github.com/bytedeco/javacpp-presets/pull/946)), DNNL 2.1.1, CPython 3.9.2, NumPy 1.20.1, SciPy 1.6.1, Gym 0.18.0, LLVM 11.1.0 ([pull #1001](https://github.com/bytedeco/javacpp-presets/pull/1001)), OpenPose 1.7.0, CUDA 11.2.1, cuDNN 8.1.1, NCCL 2.8.4, MXNet 1.8.0, TensorFlow 1.15.5, TensorRT 7.2.3.4, ONNX 1.8.1, ONNX Runtime 1.7.0, Qt 5.15.2, Skia 2.80.2, cpu_features 0.6.0, and their dependencies

### September 9, 2020 version 1.5.4
 * Bundle `libpostal_data` program, executable via `Loader.load()` for convenience ([issue #939](https://github.com/bytedeco/javacpp-presets/issues/939))
 * Enable all stable target architectures in the presets for LLVM ([pull #937](https://github.com/bytedeco/javacpp-presets/pull/937))
 * Virtualize `QObject` and its subclasses from Qt to allow customization ([issue bytedeco/javacpp#419](https://github.com/bytedeco/javacpp/issues/419))
 * Bundle programs from Clang and LLVM, executable via `Loader.load()` for convenience ([issue #833](https://github.com/bytedeco/javacpp-presets/issues/833))
 * Include `nnvm/c_api.h` header file in presets for MXNet ([issue #912](https://github.com/bytedeco/javacpp-presets/issues/912))
 * Enable OpenMP for DNNL on Mac using same library name as MKL to prevent conflicts ([issue #907](https://github.com/bytedeco/javacpp-presets/issues/907))
 * Fix loading issue with `opencv_ximgproc` ([issue #911](https://github.com/bytedeco/javacpp-presets/issues/911))
 * Build LibTIFF after WebP to make sure they link correctly in presets for Leptonica
 * Virtualize `IInt8Calibrator` plus subclasses from TensorRT to allow customization ([issue #902](https://github.com/bytedeco/javacpp-presets/issues/902))
 * Replace `requires` with `requires static` in JPMS `.platform` modules ([pull #900](https://github.com/bytedeco/javacpp-presets/pull/900))
 * Add presets for OpenPose 1.6.0 ([pull #898](https://github.com/bytedeco/javacpp-presets/pull/898))
 * Add comparison against MKL in `llvm/samples/polly/MatMulBenchmark.java`
 * Add `requires org.bytedeco.javacpp.${javacpp.platform.module}` to load `jnijavacpp` with JPMS ([pull #893](https://github.com/bytedeco/javacpp-presets/pull/893))
 * Bundle configuration files required by AOT compilation with GraalVM ([issue eclipse/deeplearning4j#7362](https://github.com/eclipse/deeplearning4j/issues/7362))
 * Add support for Windows to presets for Qt ([issue #862](https://github.com/bytedeco/javacpp-presets/issues/862))
 * Fix JPMS modules for CUDA, ARPACK-NG, GSL, SciPy, Gym, MXNet ([pull #880](https://github.com/bytedeco/javacpp-presets/pull/880) and [pull #881](https://github.com/bytedeco/javacpp-presets/pull/881)), OpenCV, CPython, LLVM, Tesseract, Qt ([pull #928](https://github.com/bytedeco/javacpp-presets/pull/928))
 * Build OpenBLAS with a `TARGET` even for `DYNAMIC_ARCH` to avoid SIGILL ([issue eclipse/deeplearning4j#8747](https://github.com/eclipse/deeplearning4j/issues/8747))
 * Upgrade presets for OpenCV 4.4.0, FFmpeg 4.3.1 ([pull #891](https://github.com/bytedeco/javacpp-presets/pull/891)), Arrow 1.0.1, Hyperscan 5.3.0, MKL 2020.3, MKL-DNN 0.21.5, DNNL 1.6.2, OpenBLAS 0.3.10, CPython 3.7.9, NumPy 1.19.1, SciPy 1.5.2, Gym 0.17.2, LLVM 10.0.1, Leptonica 1.80.0, CUDA 11.0.3, cuDNN 8.0.3, NCCL 2.7.8, MXNet 1.7.0, TensorFlow 1.15.3, TensorRT 7.1, ONNX 1.7.0 ([pull #882](https://github.com/bytedeco/javacpp-presets/pull/882)), ONNX Runtime 1.4.0 ([pull #887](https://github.com/bytedeco/javacpp-presets/pull/887)), Qt 5.15.0, Skia 2.80.1, and their dependencies
 * Add `FullOptimization.h` allowing users to fully optimize LLVM modules ([pull #869](https://github.com/bytedeco/javacpp-presets/pull/869))

### April 14, 2020 version 1.5.3
 * Add presets for the new `intensity_transform` and `rapid` modules of OpenCV
 * Add support for Polly optimizer to presets for LLVM ([pull #864](https://github.com/bytedeco/javacpp-presets/pull/864))
 * Fix loading issue with `opencv_dnn_superres` ([issue bytedeco/javacv#1396](https://github.com/bytedeco/javacv/issues/1396))
 * Add support for Windows to presets for TensorRT ([pull #860](https://github.com/bytedeco/javacpp-presets/pull/860))
 * Add dependency on presets for `jnijavacpp` and `javacpp-platform` artifact to fix issues at load time ([issue bytedeco/javacv#1305](https://github.com/bytedeco/javacv/issues/1305))
 * Bundle the official Java API of ONNX Runtime via the `jnionnxruntime` library
 * Add CUDA-enabled build for ONNX Runtime via `-gpu` extension
 * Fix presets for LLVM 9.0 where libclang would fail to load on Windows ([issue #830](https://github.com/bytedeco/javacpp-presets/issues/830))
 * Add Windows build for ONNX Runtime, map the C++ API, and refine support for DNNL ([pull #841](https://github.com/bytedeco/javacpp-presets/pull/841))
 * Add convenient `Py_AddPath()` helper method to presets for CPython
 * Include `OrcBindings.h` and other missing header files for LLVM ([issue #833](https://github.com/bytedeco/javacpp-presets/issues/833))
 * Fix `-platform` artifacts on JPMS by commenting out requires to Android modules ([issue #814](https://github.com/bytedeco/javacpp-presets/issues/814) and [pull #831](https://github.com/bytedeco/javacpp-presets/pull/831))
 * Include `timecode.h`, among other missing header files, in the `avutil` module of FFmpeg ([issue #822](https://github.com/bytedeco/javacpp-presets/issues/822))
 * Map a few more inherited constructors missing from the presets for MKL-DNN and DNNL
 * Make sure `clone()` actually returns new `PIX`, `FPIX`, or `DPIX` objects with presets for Leptonica
 * Add `opencv_python3` module and corresponding loader class with sample code to the presets for OpenCV ([issue #756](https://github.com/bytedeco/javacpp-presets/issues/756))
 * Bundle OpenSSL in the presets for CPython for consistency across platforms ([issue #796](https://github.com/bytedeco/javacpp-presets/issues/796))
 * Add presets for Arrow 0.16.0, SciPy 1.4.1 ([issue #747](https://github.com/bytedeco/javacpp-presets/issues/747)), Gym 0.17.1, Hyperscan 5.2.1 ([pull #849](https://github.com/bytedeco/javacpp-presets/pull/849))
 * Upgrade presets for OpenCV 4.3.0, FFmpeg 4.2.2, Spinnaker 1.27.0.48, HDF5 1.12.0, MKL 2020.1, MKL-DNN 0.21.4, DNNL 1.3, OpenBLAS 0.3.9, CPython 3.7.7, NumPy 1.18.2, LLVM 10.0.0, CUDA 10.2, cuDNN 7.6.5, NCCL 2.6.4, MXNet 1.6.0, TensorFlow 1.15.2, TensorRT 7.0, ALE 0.6.1, Leptonica 1.79.0, Tesseract 4.1.1, ONNX Runtime 1.2.0, Qt 5.14.2, Skia 1.68.1, and their dependencies

### November 5, 2019 version 1.5.2
 * Add presets for the `cudacodec`, `cudafeatures2d`, `cudastereo`, and `cudabgsegm` modules of OpenCV ([issue #806](https://github.com/bytedeco/javacpp-presets/issues/806))
 * Fix mapping of `warpAffine` and `warpPerspective` from `opencv_cudawarping` ([issue #806](https://github.com/bytedeco/javacpp-presets/issues/806))
 * Add `linux-armhf` and `linux-arm64` builds for HDF5 ([issue #794](https://github.com/bytedeco/javacpp-presets/issues/794))
 * Add build for Mac OS X to presets for nGraph ([issue #799](https://github.com/bytedeco/javacpp-presets/issues/799))
 * Update presets for Visual Studio 2017 on Windows
 * Bundle the `opencv_annotation`, `opencv_interactive-calibration`, `opencv_version`, `opencv_visualisation`, and `tesseract` programs
 * Add `linux-armhf`, `linux-arm64`, `linux-ppc64le`, and `windows-x86` builds for CPython, NumPy, and LLVM ([pull #768](https://github.com/bytedeco/javacpp-presets/pull/768))
 * Include `audio_ops.h`, `list_ops.h`, `lookup_ops.h`, and `manip_ops.h` for TensorFlow
 * Add necessary platform properties to build `-gpu` extensions on `linux-arm64` and `linux-ppc64le` ([issue #769](https://github.com/bytedeco/javacpp-presets/issues/769))
 * Add packages missing from TensorFlow ([issue #773](https://github.com/bytedeco/javacpp-presets/issues/773))
 * Fix JPMS module names for OpenBLAS and Tesseract ([issue #772](https://github.com/bytedeco/javacpp-presets/issues/772))
 * Include `env.h`, `kernels.h`, and `ops.h` to allow creating custom operations using the C API of TensorFlow
 * Add profiles to parent `pom.xml` to detect host and use its artifacts, for example: `mvn -Djavacpp.platform.custom -Djavacpp.platform.host ...`
 * Add `-platform-gpu`, `-platform-python`, `-platform-python-gpu`, and `-platform-redist` artifacts for convenience
 * Add presets for librealsense2 2.29.0, DNNL 1.1, ONNX Runtime 0.5.0
 * Upgrade presets for OpenCV 4.1.2, FFmpeg 4.2.1, librealsense 1.12.4, MKL 2019.5, MKL-DNN 0.21.2, OpenBLAS 0.3.7, GSL 2.6, CPython 3.7.5, NumPy 1.17.3, LLVM 9.0.0, CUDA 10.1 Update 2, cuDNN 7.6.4, NCCL 2.4.8, MXNet 1.5.1, TensorFlow 1.15.0, TensorRT 6.0, ONNX 1.6.0 (pull #795), nGraph 0.26.0, Qt 5.13.1, cpu_features 0.4.1, and their dependencies

### July 9, 2019 version 1.5.1
 * Add `linux-arm64` CI builds for OpenCV, FFmpeg, OpenBLAS, FFTW, GSL, Leptonica, Tesseract, and others ([issue bytedeco/javacv#1021](https://github.com/bytedeco/javacv/issues/1021))
 * Add convenient `Tensor.create(boolean[] data, shape)` factory methods for TensorFlow
 * Set correct default path to `javacpp.platform.compiler` for Android builds on Mac OS X ([issue #733](https://github.com/bytedeco/javacpp-presets/issues/733))
 * Call `Loader.checkVersion()` in all presets to log warnings with potentially incompatible versions of JavaCPP
 * Add missing `mkl_gnu_thread` preload in presets for OpenBLAS, MKL-DNN, and TensorFlow ([pull #729](https://github.com/bytedeco/javacpp-presets/pull/729))
 * Overload `Tensor.create()` factory methods for TensorFlow with handy `long... shape` ([issue bytedeco/javacpp#301](https://github.com/bytedeco/javacpp/issues/301))
 * Add build for `linux-arm64` to presets for OpenBLAS ([pull #726](https://github.com/bytedeco/javacpp-presets/pull/726))
 * Bundle complete binary packages of CPython itself for convenience ([issue #712](https://github.com/bytedeco/javacpp-presets/issues/712))
 * Fix and refine mapping for `HoughLines`, `HoughLinesP`, and `HoughCircles` ([issue #717](https://github.com/bytedeco/javacpp-presets/issues/717))
 * Add Python-enabled builds for TensorFlow via the `-python` and `-python-gpu` extensions
 * Map the C/C++ API supporting eager execution in the presets for TensorFlow
 * Load the symbols from the `python` library globally as often required by Python libraries ([issue ContinuumIO/anaconda-issues#6401](https://github.com/ContinuumIO/anaconda-issues/issues/6401))
 * Link OpenCV with OpenBLAS/MKL to accelerate some matrix operations
 * Add presets for the `quality` module of OpenCV
 * Upgrade presets for OpenCV 4.1.0, libdc1394 2.2.6, MKL 2019.4, MKL-DNN 0.20, OpenBLAS 0.3.6, CPython 3.7.3, NumPy 1.16.4, Tesseract 4.1.0, CUDA 10.1 Update 1, cuDNN 7.6, MXNet 1.5.0.rc2, TensorFlow 1.14.0, ONNX 1.5.0, nGraph 0.22.0, Qt 5.13.0, cpu_features 0.3.0, and their dependencies

### April 11, 2019 version 1.5
 * Include `setlocale()` in presets for Tesseract to work around issues with locale ([issue #694](https://github.com/bytedeco/javacpp-presets/issues/694))
 * Bundle the `python` program, executable via `Loader.load()` for convenience
 * Bundle Vector Mathematical Library (VML) in redist artifacts of the presets for MKL ([issue #705](https://github.com/bytedeco/javacpp-presets/issues/705))
 * Add `org.bytedeco.tensorflow.StringArray.put(BytePointer)` method to change character encoding ([issue bytedeco/javacpp#293](https://github.com/bytedeco/javacpp/issues/293))
 * Bundle `ffmpeg` and `ffprobe` programs, executable via `Loader.load()` for convenience ([issue bytedeco/javacv#307](https://github.com/bytedeco/javacv/issues/307))
 * Add functions related to threading missing from presets for CPython
 * Lengthen `Mat` size and step getters to support `long` indexing ([pull #700](https://github.com/bytedeco/javacpp-presets/pull/700))
 * Rename `groupId` to "org.bytedeco" and use ModiTect to modularize all presets and comply with JPMS ([pull #681](https://github.com/bytedeco/javacpp-presets/pull/681))
 * Make `nvinfer1::Weights::values` settable in presets for TensorRT ([issue #698](https://github.com/bytedeco/javacpp-presets/issues/698))
 * Fix mapping of `HoughLines`, `HoughLinesP`, `HoughCircles`, and `Subdiv2D` from `opencv_imgproc` (issues [bytedeco/javacv#913](https://github.com/bytedeco/javacv/issues/913) and [bytedeco/javacv#1146](https://github.com/bytedeco/javacv/issues/1146))
 * Add basic mapping of stdio streams to presets for GSL since it relies on them for serialization
 * Fix crash in Leptonica on CentOS 6 by downgrading to libpng 1.5.30 ([issue #680](https://github.com/bytedeco/javacpp-presets/issues/680))
 * Add `GetComponentImagesExample`, `IteratorOverClassifierChoicesExample`, `OrientationAndScriptDetectionExample`, and `ResultIteratorExample` for Tesseract ([pull #673](https://github.com/bytedeco/javacpp-presets/pull/673) and [pull #675](https://github.com/bytedeco/javacpp-presets/pull/675))
 * Add presets for NumPy 1.16.2, NCCL 2.4.2, nGraph 0.15.0 ([pull #642](https://github.com/bytedeco/javacpp-presets/pull/642)), Qt 5.12.2 ([pull #674](https://github.com/bytedeco/javacpp-presets/pull/674)), and cpu_features 0.2.0 ([issue #526](https://github.com/bytedeco/javacpp-presets/issues/526))
 * Upgrade presets for FFmpeg 4.1.3, libfreenect 0.5.7, HDF5 1.10.5, MKL 2019.3, MKL-DNN 0.18.1, LLVM 8.0.0, Leptonica 1.78.0, ARPACK-NG 3.7.0, CUDA 10.1, cuDNN 7.5, MXNet 1.4.0, TensorFlow 1.13.1, TensorRT 5.1, ONNX 1.4.1 ([pull #676](https://github.com/bytedeco/javacpp-presets/pull/676)), LiquidFun, Skia 1.68.0, and their dependencies including NCCL
 * Build OpenCV without UI when environment variable `HEADLESS=yes` ([pull #667](https://github.com/bytedeco/javacpp-presets/pull/667))

### January 11, 2019 version 1.4.4
 * Bundle the full version of MKL now that its new license permits it ([issue #601](https://github.com/bytedeco/javacpp-presets/issues/601))
 * Bundle libraries from raspberrypi/userland to avoid loading issues on `linux-armhf` devices other than Raspberry Pi ([issue bytedeco/javacv#1118](https://github.com/bytedeco/javacv/issues/1118))
 * Bundle the new official Java/Scala API of MXNet via the `jnimxnet` library
 * Add `QuickSpinC.h`, `SpinVideoC.h`, and `TransportLayer*C.h` for Spinnaker ([pull #660](https://github.com/bytedeco/javacpp-presets/pull/660))
 * Add `FlyCapture2Video.h` and `FlyCapture2VideoDef.h`, and remove `AVIRecorder.h` for FlyCapture ([pull #613](https://github.com/bytedeco/javacpp-presets/pull/613))
 * Switch to Clang for Android builds with recent versions of the NDK ([issue #562](https://github.com/bytedeco/javacpp-presets/issues/562))
 * Include `sys/sysinfo.h` for system API of Linux
 * Include `ucrtbase.dll` when bundling the runtime for Visual Studio 2015 on Windows ([issue bytedeco/javacv#1098](https://github.com/bytedeco/javacv/issues/1098))
 * Add support for N-dimensional arrays to `opencv_core.Mat.createIndexer()` ([pull #647](https://github.com/bytedeco/javacpp-presets/pull/647))
 * Add for `CvMat`, `IplImage`, and `PIX`, helper `create(..., Pointer data)` factory methods that prevent premature deallocation ([issue bytedeco/javacpp#272](https://github.com/bytedeco/javacpp/issues/272) and [issue bytedeco/javacv#1101](https://github.com/bytedeco/javacv/issues/1101))
 * Enable x265 multilib depth support at 8, 10, and 12 bits for FFmpeg ([pull #619](https://github.com/bytedeco/javacpp-presets/pull/619))
 * Include all header files from `Python.h` in presets for CPython
 * Fix mapping of `initCameraMatrix2D`, `calibrateCamera`, and `stereoCalibrate` functions from `opencv_calib3d`
 * Build OpenCV with pthreads instead of OpenMP or GCD due to thread-safety and usability issues ([issue bytedeco/javacv#396](https://github.com/bytedeco/javacv/issues/396))
 * Include IR, optimizer, and version converter for ONNX ([pull #622](https://github.com/bytedeco/javacpp-presets/pull/622))
 * Add build for Mac OS X to presets for ONNX ([issue #638](https://github.com/bytedeco/javacpp-presets/issues/638))
 * Allow MKL-DNN to link with the full version of MKL at runtime ([issue #629](https://github.com/bytedeco/javacpp-presets/issues/629))
 * Add builds for `linux-mips64el` to presets for ARToolKitPlus, Chilitags, flandmark, OpenBLAS, and FFTW ([pull #637](https://github.com/bytedeco/javacpp-presets/pull/637))
 * Update sample code for GSL with a more complex example ([issue #636](https://github.com/bytedeco/javacpp-presets/issues/636))
 * Fix CUDA build for OpenCV on Mac OS X missing `libopencv_cudev.dylib` ([issue #626](https://github.com/bytedeco/javacpp-presets/issues/626))
 * Upgrade presets for OpenCV 4.0.1, FFmpeg 4.1, FlyCapture 2.13.3.31, Spinnaker 1.19.0.22, HDF5 1.10.4, MKL 2019.1, MKL-DNN 0.17.2, OpenBLAS 0.3.5, LLVM 7.0.1, Leptonica 1.77.0, Tesseract 4.0.0, cuDNN 7.4, MXNet 1.4.0.rc0, TensorFlow 1.12.0, and their dependencies

### October 15, 2018 version 1.4.3
 * Keep globally shared dummy deallocator for `TF_Tensor` out of `PointerScope`
 * Add build for `linux-mips64el` to presets for OpenCV ([pull #621](https://github.com/bytedeco/javacpp-presets/pull/621))
 * Remove calls to deprecated functions from sample code for FFmpeg ([pull #323](https://github.com/bytedeco/javacpp-presets/pull/323))
 * Call `Pointer.setNull()` in custom deallocators for the C API of TensorFlow to prevent double free from occurring
 * Add profiles to parent `pom.xml` that allow multiple platforms: `mvn -Djavacpp.platform.none -Djavacpp.platform.linux-x86_64 -Djavacpp.platform.windows-x86_64 ...`
 * Add support for Windows to presets for LiquidFun ([pull #536](https://github.com/bytedeco/javacpp-presets/pull/536)) and MXNet ([pull #309](https://github.com/bytedeco/javacpp-presets/pull/309))
 * Add CUDA-enabled build for MXNet via `-gpu` extension ([pull #609](https://github.com/bytedeco/javacpp-presets/pull/609))
 * Prevent MKL-DNN from compiling code with `-march=native` ([pull #618](https://github.com/bytedeco/javacpp-presets/pull/618))
 * Add an RPATH to `libmkldnn.so.0` to avoid loading issues on Linux ([issue deeplearning4j/deeplearning4j#6366](https://github.com/deeplearning4j/deeplearning4j/issues/6366))
 * Fix logic in `tensorflow.Tensor.createIndexer()` to support scalar tensors
 * Bundle `libgomp.so.1` in JAR file of MKL-DNN for Linux
 * Enable OpenMP for MKL-DNN also on Mac and Windows by building with GCC
 * Fix loading order of runtime libraries for Visual Studio 2015 on Windows ([issue #606](https://github.com/bytedeco/javacpp-presets/issues/606))
 * Add methods overloaded with `PointerPointer` for MKL-DNN ([issue bytedeco/javacpp#251](https://github.com/bytedeco/javacpp/issues/251))
 * Bundle native resources (header files and import libraries) of MKL-DNN
 * Make MSBuild compile more efficiently on multiple processors ([pull #599](https://github.com/bytedeco/javacpp-presets/pull/599))
 * Add samples for Clang ([pull #598](https://github.com/bytedeco/javacpp-presets/pull/598))
 * Include `tag_constants.h`, `signature_constants.h`, `graph_runner.h`, `shape_refiner.h`, `python_api.h`, and enable Python API for TensorFlow ([issue #602](https://github.com/bytedeco/javacpp-presets/issues/602))
 * Add presets for Spinnaker 1.15.x ([pull #553](https://github.com/bytedeco/javacpp-presets/pull/553)), CPython 3.6.x, ONNX 1.3.0 ([pull #547](https://github.com/bytedeco/javacpp-presets/pull/547))
 * Define `std::vector<tensorflow::OpDef>` type to `OpDefVector` for TensorFlow
 * Link HDF5 with zlib on Windows also ([issue deeplearning4j/deeplearning4j#6017](https://github.com/deeplearning4j/deeplearning4j/issues/6017))
 * Enable MKL-DNN for MXNet and TensorFlow
 * Upgrade presets for OpenCV 3.4.3, FFmpeg 4.0.2, HDF5 1.10.3, MKL 2019.0, MKL-DNN 0.16, OpenBLAS 0.3.3, ARPACK-NG 3.6.3, LLVM 7.0.0, Tesseract 4.0.0-rc2, CUDA 10.0, cuDNN 7.3, MXNet 1.3.0, TensorFlow 1.11.0, TensorRT 5.0, and their dependencies
 * Fix loading issue with `opencv_cudaobjdetect` and `opencv_cudaoptflow` on Windows ([issue #592](https://github.com/bytedeco/javacpp-presets/issues/592))

### July 17, 2018 version 1.4.2
 * Fix FFmpeg build of libvpx with Linux on ARM ([issue #586](https://github.com/bytedeco/javacpp-presets/issues/586))
 * Enable MediaCodec acceleration for FFmpeg on Android ([pull #589](https://github.com/bytedeco/javacpp-presets/pull/589))
 * Include `c_api_internal.h` for TensorFlow ([issue #585](https://github.com/bytedeco/javacpp-presets/issues/585))
 * Build all presets on CentOS 6 with Developer Toolset 6 and move almost all Linux builds to CentOS 6
 * Fix functions from `facemark.hpp` and `face_alignment.hpp` that crash when called with `cv::Mat` objects
 * Virtualize `TensorBuffer` and make constructor with helper method in `Tensor` public to allow zero-copy
 * Add factory methods for `TF_Status`, `TF_Buffer`, `TF_Tensor`, `TF_SessionOptions`, `TF_Graph`, `TF_ImportGraphDefOptions`, and `TF_Session` that register deallocators
 * Bundle the libraries of CUDA and cuDNN, allowing OpenCV, Caffe, and TensorFlow to use GPUs with no CUDA installation
 * Make it possible to set the `TF_Buffer::data` field for C API of TensorFlow
 * Add `openblas_nolapack` class to make it easier to load BLAS libraries missing LAPACK such as MKLML
 * Map instances of `google::protobuf::Map` to access more of TensorFlow's configuration ([issue #533](https://github.com/bytedeco/javacpp-presets/issues/533))
 * Fix presets for OpenBLAS failing to load MKL when symbolic links are enabled on Windows
 * Define `CV__LEGACY_PERSISTENCE` to get back functions for `KeyPointVector` and `DMatchVector` ([issue bytedeco/javacv#1012](https://github.com/bytedeco/javacv/issues/1012))
 * Skip by default OpenBLAS functions missing from MKL 2017
 * Fix presets for OpenBLAS on `linux-ppc64le` not bundling correct libraries ([issue deeplearning4j/deeplearning4j#5447](https://github.com/deeplearning4j/deeplearning4j/issues/5447))
 * Fix CUDA build for TensorFlow on Windows ([pull #567](https://github.com/bytedeco/javacpp-presets/pull/567))
 * Disable optimized kernels of OpenBLAS on iOS as they return incorrect results ([issue #571](https://github.com/bytedeco/javacpp-presets/issues/571))
 * Get exception messages from `H5::Exception` for HDF5 ([issue deeplearning4j/deeplearning4j#5379](https://github.com/deeplearning4j/deeplearning4j/issues/5379))
 * Add more samples for TensorFlow including a complete training example ([pull #563](https://github.com/bytedeco/javacpp-presets/pull/563))
 * Add helper for `PIX`, `FPIX`, and `DPIX` of Leptonica, facilitating access to image data of Tesseract ([issue #517](https://github.com/bytedeco/javacpp-presets/issues/517))
 * Add presets for the NVBLAS, NVGRAPH, NVRTC, and NVML modules of CUDA ([issue deeplearning4j/nd4j#2895](https://github.com/deeplearning4j/nd4j/issues/2895))
 * Link OpenBLAS with `-Wl,-z,noexecstack` on `linux-armhf` as required by the JDK ([issue deeplearning4j/libnd4j#700](https://github.com/deeplearning4j/libnd4j/issues/700))
 * Include `textDetector.hpp` from the `opencv_text` module
 * Include `feature.pb.h`, `example.pb.h`, `record_reader.h`, and `record_writer.h` for TensorFlow ([issue tensorflow/tensorflow#17390](https://github.com/tensorflow/tensorflow/issues/17390))
 * Enhance presets for `ALE` with access to `theOSystem`, etc ([issue #551](https://github.com/bytedeco/javacpp-presets/issues/551))
 * Add presets for the `saliency` module of OpenCV ([pull #555](https://github.com/bytedeco/javacpp-presets/pull/555))
 * Add build for `linux-arm64` to presets for FFmpeg ([pull #556](https://github.com/bytedeco/javacpp-presets/pull/556))
 * Add support for Windows to presets for TensorFlow ([issue #111](https://github.com/bytedeco/javacpp-presets/issues/111))
 * Add Android utility classes from the official Java API of OpenCV and TensorFlow ([issue #549](https://github.com/bytedeco/javacpp-presets/issues/549))
 * Update build for FFmpeg on Raspbian Stretch ([pull #548](https://github.com/bytedeco/javacpp-presets/pull/548))
 * Add presets for MKL-DNN 0.15 and TensorRT 4.0
 * Fix build for FFmpeg on `android-x86` and `android-x86_64` platforms ([issue bytedeco/javacv#945](https://github.com/bytedeco/javacv/issues/945))
 * Upgrade presets for OpenCV 3.4.2, FFmpeg 4.0.1, HDF5 1.10.2, MKL 2018.3, OpenBLAS 0.3.0, ARPACK-NG 3.6.1, FFTW 3.3.8, GSL 2.5, LLVM 6.0.1, Leptonica 1.76.0, Tesseract 4.0.0-beta.3 ([issue #385](https://github.com/bytedeco/javacpp-presets/issues/385)), Caffe 1.0, CUDA 9.2, MXNet 1.2.1.rc1, TensorFlow 1.9.0, and their dependencies

### March 29, 2018 version 1.4.1
 * Disable unneeded error messages from LibTIFF in presets for Leptonica ([issue deeplearning4j/DataVec#518](https://github.com/deeplearning4j/DataVec/pull/518))
 * Bundle the official Java APIs of OpenCV and TensorFlow, via the `opencv_java` and `tensorflow_cc` modules
 * Correct loading order of `mkl_core` to fix MKL issues on Windows ([issue deeplearning4j/deeplearning4j#4776](https://github.com/deeplearning4j/deeplearning4j/issues/4776))
 * Add presets for the `aruco`, `bgsegm`, `img_hash`, `phase_unwrapping`, `plot`, `structured_light`, `tracking`, and `xphoto` modules of OpenCV ([issue #319](https://github.com/bytedeco/javacpp-presets/issues/319))
 * Add bindings for `b2DynamicTree::Query` and `RayCast` for LiquidFun ([pull #531](https://github.com/bytedeco/javacpp-presets/pull/531))
 * Add support for Windows to presets for LLVM ([pull #530](https://github.com/bytedeco/javacpp-presets/pull/530))
 * Add builds for `android-arm64` and `android-x86_64` platforms ([issue #52](https://github.com/bytedeco/javacpp-presets/issues/52))
 * Fix x265 encoding with FFmpeg on Android ([issue bytedeco/javacv#866](https://github.com/bytedeco/javacv/issues/866))
 * Add presets for ARPACK-NG, CMINPACK
 * Add iOS builds for OpenCV, OpenBLAS, and Skia ([pull #525](https://github.com/bytedeco/javacpp-presets/pull/525))
 * Let GSL link with OpenBLAS, MKL, Accelerate, etc automatically instead of GSL CBLAS ([issue #18](https://github.com/bytedeco/javacpp-presets/issues/18))
 * Append `@NoException` annotation to presets for libdc1394, libfreenect, MKL, OpenBLAS, FFTW, GSL, Leptonica, CUDA, and system APIs to reduce unneeded C++ overhead
 * Fix mapping of `fftwf_iodim` and `fftwf_iodim64` for FFTW ([issue #523](https://github.com/bytedeco/javacpp-presets/issues/523))
 * Add support for iOS and Accelerate to presets for OpenBLAS ([pull #515](https://github.com/bytedeco/javacpp-presets/pull/515))
 * Add "org.bytedeco.javacpp.openblas.load" system property to use libraries from Accelerate, etc ([pull #444](https://github.com/bytedeco/javacpp-presets/pull/444))
 * Upgrade presets for OpenCV 3.4.1, FFmpeg 3.4.2, Leptonica 1.75.3, cuDNN 7.1, MXNet 1.1.0, TensorFlow 1.7.0-rc1, and their dependencies
 * Include `facemark.hpp`, `facemarkLBF.hpp`, `facemarkAAM.hpp`, `face_alignment.hpp` from the `opencv_face` module
 * Add `AxpyLayer` to presets for Caffe ([pull #508](https://github.com/bytedeco/javacpp-presets/pull/508))

### January 16, 2018 version 1.4
 * Fix some integer types in HDF5 being mistakenly mapped to smaller integer types
 * Remove the need for empty artifacts of unsupported platforms ([issue #434](https://github.com/bytedeco/javacpp-presets/issues/434))
 * Link `jnivideoInputLib.dll` statically to avoid missing dependencies ([issue bytedeco/javacv#864](https://github.com/bytedeco/javacv/issues/864))
 * Add "org.bytedeco.javacpp.openblas.nomkl" system property to let users disable MKL easily
 * Add initial set of CUDA bindings for OpenCV ([pull #416](https://github.com/bytedeco/javacpp-presets/pull/416))
 * Add CUDA/OpenCL-enabled builds for OpenCV, Caffe, and TensorFlow via `-gpu` extension ([issue bytedeco/javacv#481](https://github.com/bytedeco/javacv/issues/481))
 * Enable NVIDIA CUDA, CUVID, and NVENC acceleration for FFmpeg ([pull #492](https://github.com/bytedeco/javacpp-presets/pull/492))
 * Include `message_lite.h`, `checkpoint_reader.h`, `saver.pb.h`, `meta_graph.pb.h`, and `loader.h` for TensorFlow ([issue #494](https://github.com/bytedeco/javacpp-presets/issues/494))
 * Add `getString()` helper methods to `CXString`, `CXTUResourceUsageKind`, and `CXEvalResult` for `clang` ([issue bytedeco/javacpp#51](https://github.com/bytedeco/javacpp/issues/51))
 * Add support for Mac OS X and Windows to presets for librealsense ([issue #447](https://github.com/bytedeco/javacpp-presets/issues/447))
 * Enable MMAL and OpenMAX acceleration for FFmpeg on `linux-armhf` ([pull #388](https://github.com/bytedeco/javacpp-presets/pull/388))
 * Enable V4L2 for OpenCV on ARM platforms as well ([issue bytedeco/javacv#850](https://github.com/bytedeco/javacv/issues/850))
 * Enable Intel QSV acceleration via libmfx for FFmpeg ([pull #485](https://github.com/bytedeco/javacpp-presets/pull/485))
 * Add support for Mac OS X and Windows to presets for libfreenect2 ([issue bytedeco/javacv#837](https://github.com/bytedeco/javacv/issues/837))
 * Enable VisualOn AMR-WB encoder library support for FFmpeg ([pull #487](https://github.com/bytedeco/javacpp-presets/pull/487))
 * Use native threads for FFmpeg on Windows to prevent deadlocks with pthreads ([pull #481](https://github.com/bytedeco/javacpp-presets/pull/481))
 * Add cross-compilation support for `linux-ppc64le` ([pull #471](https://github.com/bytedeco/javacpp-presets/pull/471))
 * Fix definitions of `fftw_iodim` and `fftw_iodim64` for FFTW ([issue #466](https://github.com/bytedeco/javacpp-presets/issues/466))
 * Add `Mat(Point)` and `Mat(Scalar)` constructors to OpenCV for convenience ([issue bytedeco/javacv#738](https://github.com/bytedeco/javacv/issues/738))
 * Include the `libavutil/time.h` for FFmpeg ([issue bytedeco/javacv#735](https://github.com/bytedeco/javacv/issues/735))
 * Map `AVERROR()` for error codes from `errno.h` used by FFmpeg ([pull #459](https://github.com/bytedeco/javacpp-presets/pull/459))
 * Use `long` instead of `int` for constants starting with `AV_CH_LAYOUT_` in `avutil` ([pull #455](https://github.com/bytedeco/javacpp-presets/pull/455))
 * Build FFmpeg with FreeType, enabling the `drawtext` filter ([pull #452](https://github.com/bytedeco/javacpp-presets/pull/452)), and support for Opus ([pull #457](https://github.com/bytedeco/javacpp-presets/pull/457))
 * Enable cuDNN support for Caffe by default ([issue #187](https://github.com/bytedeco/javacpp-presets/issues/187))
 * Use the new `BUILD_PATH` environment variable passed by the `javacpp` plugin to the `cppbuild.sh` script files
 * Let presets pick up `include` or `link` paths from cached resources ([pull #101](https://github.com/bytedeco/javacpp-presets/pull/101))
 * Update presets for Visual Studio 2015 ([issue #298](https://github.com/bytedeco/javacpp-presets/issues/298))
 * Reuse the presets for HDF5 and OpenBLAS with Caffe and MXNet
 * Replace the `exec-maven-plugin` with the `javacpp` plugin itself to execute the `cppbuild.sh` script files
 * Link TensorFlow statically with `cudart` to avoid dependency on CUDA ([issue #396](https://github.com/bytedeco/javacpp-presets/issues/396))
 * Add missing call to `Loader.load()` in helper class for `opencv_ml` ([issue bytedeco/javacv#638](https://github.com/bytedeco/javacv/issues/638))
 * Work around issues with TensorFlow on some versions of Mac OS X ([issue #335](https://github.com/bytedeco/javacpp-presets/issues/335))
 * Upgrade presets for OpenCV 3.4.0, FFmpeg 3.4.1, FlyCapture 2.11.3.121 ([pull #424](https://github.com/bytedeco/javacpp-presets/pull/424)), libdc1394 2.2.5, librealsense 1.12.1, Chilitags, HDF5 1.10.1, OpenBLAS 0.2.20, FFTW 3.3.7, GSL 2.4, LLVM 5.0.1 ([pull #404](https://github.com/bytedeco/javacpp-presets/pull/404)), Leptonica 1.74.4, Tesseract 3.05.01, Caffe 1.0, CUDA 9.1, cuDNN 7.0, MXNet 1.0.0, TensorFlow 1.5.0-rc1, and their dependencies
 * Add presets for libfreenect2 ([pull #340](https://github.com/bytedeco/javacpp-presets/pull/340)), MKL 2018.1 ([issue #112](https://github.com/bytedeco/javacpp-presets/issues/112)), libpostal 1.1-alpha ([pull #502](https://github.com/bytedeco/javacpp-presets/pull/502)), The Arcade Learning Environment 0.6.0, LiquidFun ([pull #356](https://github.com/bytedeco/javacpp-presets/pull/356)), Skia ([pull #418](https://github.com/bytedeco/javacpp-presets/pull/418)), and system APIs (Linux, Mac OS X, and Windows)
 * Fix the `FlyCapture2` module for some versions on Windows ([issue #337](https://github.com/bytedeco/javacpp-presets/issues/337))
 * Add functions missing from the presets of MXNet ([issue #332](https://github.com/bytedeco/javacpp-presets/issues/332))
 * Add presets for the `text` module of OpenCV 3.x ([pull #333](https://github.com/bytedeco/javacpp-presets/pull/333))

### December 7, 2016 version 1.3
 * Fix FFmpeg builds on ARM when not using a cross compiler ([issue #322](https://github.com/bytedeco/javacpp-presets/issues/322))
 * Add `blas_extra.h` to presets for OpenBLAS, containing `blas_set_num_threads()` and `blas_get_vendor()` functions
 * Introduce platform artifacts that depend on binaries for all available platforms and work with any build system (sbt, Gradle, M2Eclipse, etc)
 * Map more functions of the OpenCV Transparent API with `UMat` and `UMatVector` parameters ([issue bytedeco/javacv#518](https://github.com/bytedeco/javacv/issues/518))
 * Add support for `android-arm` and `android-x86` platforms to TensorFlow presets ([pull #297](https://github.com/bytedeco/javacpp-presets/pull/297))
 * Keep a reference of `tensorflow.SessionOptions` in `AbstractSession` to prevent premature deallocation ([pull #297](https://github.com/bytedeco/javacpp-presets/pull/297))
 * Enable CUDA in `cppbuild.sh` script for TensorFlow ([issue #294](https://github.com/bytedeco/javacpp-presets/issues/294))
 * Bundle `libgomp.so.1` in JAR files of OpenCV for the sake of some Linux distributions ([issue bytedeco/javacv#436](https://github.com/bytedeco/javacv/issues/436))
 * Fix `linux-armhf` and `linux-ppc64le` builds for all presets ([pull #279](https://github.com/bytedeco/javacpp-presets/pull/279))
 * Fix `libdc1394` not properly linking with `libusb-1.0` on Mac OS X ([issue bytedeco/javacv#501](https://github.com/bytedeco/javacv/issues/501))
 * Add presets for the `bioinspired` module of OpenCV 3.1 ([pull #282](https://github.com/bytedeco/javacpp-presets/pull/282))
 * Include `tensorflow/core/graph/dot.h` header file from TensorFlow ([pull #272](https://github.com/bytedeco/javacpp-presets/pull/272))
 * Add presets for librealsense, HDF5, and OpenBLAS/MKL ([issue #112](https://github.com/bytedeco/javacpp-presets/issues/112))
 * Make Caffe work on CPU-only machines ([issue #219](https://github.com/bytedeco/javacpp-presets/issues/219))
 * Fix loading issue with `opencv_face` ([issue bytedeco/javacv#470](https://github.com/bytedeco/javacv/issues/470))
 * Fix presets for CUDA on the `linux-ppc64le` platform
 * Upgrade presets for FFmpeg 3.2.1, FFTW 3.3.5, GSL 2.2.1, LLVM 3.9.0, CUDA 8.0, cuDNN 5.1, Caffe, MXNet, TensorFlow 0.11.0, and their dependencies
 * Set default options in `tensorflow/cppbuild.sh` to prevent console reads during build
 * Add `Tensor.createStringArray()` method to access `DT_STRING` data ([issue #249](https://github.com/bytedeco/javacpp-presets/issues/249))
 * Fix Javadoc links for externally referenced classes
 * Work around build issues for TensorFlow on some Linux distributions ([issue bazelbuild/bazel#1322](https://github.com/bazelbuild/bazel/issues/1322))
 * Fix `cppbuild.sh` script to build properly FFmpeg from source on Windows
 * Stop using Zeranoe FFmpeg builds, often not available for release versions ([issue #225](https://github.com/bytedeco/javacpp-presets/issues/225))
 * Add `linux-ppc64le` to `cppbuild.sh` scripts of OpenCV, FFmpeg, Leptonica, and Tesseract

### May 15, 2016 version 1.2
 * Build libdc1394 for the Windows platform as well ([issue bytedeco/procamcalib#4](https://github.com/bytedeco/procamcalib/issues/4))
 * Lower Maven prerequisite in the `pom.xml` file to 3.0 ([issue bytedeco/javacpp#93](https://github.com/bytedeco/javacpp/issues/93))
 * Include the `Descriptor` and `Message` APIs in the presets for Caffe ([issue #196](https://github.com/bytedeco/javacpp-presets/issues/196))
 * Prevent creating text relocations for shared libraries on Android ([issue bytedeco/javacv#245](https://github.com/bytedeco/javacv/issues/245))
 * Make sure to include only native libraries in platform specific JAR files ([pull bytedeco/javacpp#89](https://github.com/bytedeco/javacpp/pull/89))
 * Execute the `cppbuild.sh` scripts within the Maven build lifecycle, can be skipped with `-Djavacpp.cppbuild.skip` ([pull #175](https://github.com/bytedeco/javacpp-presets/pull/175))
 * Fix Caffe crashing in GPU mode: Do not define `CPU_ONLY` ([issue #147](https://github.com/bytedeco/javacpp-presets/issues/147))
 * Make OpenBLAS build for Caffe more generic ([issue #154](https://github.com/bytedeco/javacpp-presets/issues/154))
 * Include missing `graph_constructor.h` header file from the `tensorflow` module ([issue #165](https://github.com/bytedeco/javacpp-presets/issues/165))
 * Add missing `GraphDefBuilder.Options.WithAttr()` methods from the `tensorflow` module ([issue #160](https://github.com/bytedeco/javacpp-presets/issues/160))
 * Add `linux-armhf` platform to the `cppbuild.sh` scripts of OpenCV, FFmpeg, etc ([pull #177](https://github.com/bytedeco/javacpp-presets/pull/177))
 * Add support for Motion JPEG to the minimal configuration proposed for FFmpeg in the `cppbuild.sh` file
 * Make `mvn -Djavacpp.platform=...` and `mvn -Djavacpp.platform.dependency=...` commands work correctly
 * Add presets for the `dnn` module of OpenCV 3.1 ([issue #145](https://github.com/bytedeco/javacpp-presets/issues/145))
 * Prepend "javacpp." to all properties associated with Maven in the `pom.xml` files to avoid name clashes
 * Add a `Mat(CvArr arr)` constructor for convenience ([issue bytedeco/javacv#317](https://github.com/bytedeco/javacv/issues/317))
 * Fix loading issue with `opencv_stitching` and `opencv_xfeatures2d` ([issue bytedeco/javacv#316](https://github.com/bytedeco/javacv/issues/316), [issue bytedeco/javacv#336](https://github.com/bytedeco/javacv/issues/336))
 * Virtualize all `Solver` classes from Caffe ([issue #143](https://github.com/bytedeco/javacpp-presets/issues/143))
 * Work around GSL not loading on Android ([issue bytedeco/javacpp#55](https://github.com/bytedeco/javacpp/issues/55))
 * Fix Android builds of FFmpeg, FFTW, GSL, Leptonica, and Tesseract causing errors under Mac OS X ([issue #45](https://github.com/bytedeco/javacpp-presets/issues/45))
 * Avoid versioning of FFTW and GSL libraries, preventing them from working on Android ([issue #127](https://github.com/bytedeco/javacpp-presets/issues/127))
 * Upgrade presets for OpenCV 3.1.0, FFmpeg 3.0.2, OpenSSL 1.0.2h, x265 1.9, FlyCapture 2.9.3.43, libdc1394 2.2.4, videoInput, GSL 2.1, LLVM 3.8.0, Leptonica 1.73, Tesseract 3.04.01, cuDNN 5, and Caffe rc3, including the latest versions of their dependencies ([issue bytedeco/javacpp#55](https://github.com/bytedeco/javacpp/issues/55))
 * Add presets for MXNet and TensorFlow 0.8.0 ([issue #111](https://github.com/bytedeco/javacpp-presets/issues/111))
 * Virtualize `opencv_videostab.IFrameSource` to let us implement it in Java ([issue bytedeco/javacv#277](https://github.com/bytedeco/javacv/issues/277))
 * Fix MinGW-w64 builds with recent versions of GCC 5.x and potential issue with using "w32threads" for FFmpeg
 * Add missing `StatModel.loadXXX()` methods in the `opencv_ml` module ([issue #109](https://github.com/bytedeco/javacpp-presets/issues/109))
 * Define commonly used Caffe `std::vector` types (`DatumVector`, `FloatCallbackVector`, and `DoubleCallbackVector`) for ease of use and performance reasons
 * Fix the `cppbuild.sh` script for FFmpeg, failing to build x264 and OpenH264 properly on Windows

### October 25, 2015 version 1.1
 * Build the Maven artifacts for Linux in a CentOS 6 Docker container, for maximum compatibility ([issue #22](https://github.com/bytedeco/javacpp-presets/issues/22))
 * Cache files downloaded by `cppbuild.sh` in the `downloads` subdirectory to prevent having to redownload everything after a clean
 * Add the `clang` module to the presets for LLVM
 * Propose for FFmpeg in the `cppbuild.sh` file a minimal configuration to support MPEG-4 streams with H.264 and AAC
 * Add the non-GPL OpenH264 as an alternative H.264 encoder to x264 in FFmpeg
 * Hack together `log_callback.h` to be able to redirect to Java log messages from FFmpeg
 * Pick up `OLDCC`, `OLDCXX`, and `OLDFC` environment variables in `cppbuild.sh` and `platform.oldcompiler` system property in Maven to build with it libraries that can tolerate an older version of the C/C++ compiler on Linux
 * Upgrade all Maven dependencies and plugins to latest versions, thus bumping minimum requirements to Java SE 7, Android 4.0, and Maven 3.0
 * Provide `cppbuild.sh` script for Caffe that includes all dependencies ([pull #77](https://github.com/bytedeco/javacpp-presets/pull/77))
 * Upgrade presets for Caffe, CUDA 7.5, cuDNN 3, FFmpeg 2.8.1, Speex 1.2rc2, x265 1.8, FlyCapture 2.8.3.1, libfreenect 0.5.3, LLVM 3.7.0, Tesseract 3.04
 * Include `motion_vector.h`, `fifo.h`, and `audio_fifo.h` header files in the `avutil` module of FFmpeg ([issue #79](https://github.com/bytedeco/javacpp-presets/issues/79))
 * Add presets for Chilitags, thanks to Chris Walters for the financial contribution
 * Let users choose the runtime type of `layer_by_name()` from `FloatNet` or `DoubleNet` in `caffe` ([issue bytedeco/javacpp#25](https://github.com/bytedeco/javacpp/issues/25))
 * Add presets for the `face`, `optflow`, and `xfeatures2d` modules of OpenCV 3.0 ([issue bytedeco/javacv#196](https://github.com/bytedeco/javacv/issues/196), [issue bytedeco/javacv#239](https://github.com/bytedeco/javacv/issues/239), [issue #54](https://github.com/bytedeco/javacpp-presets/issues/54))
 * Switch to GCC 4.9 by default on Android, probably dropping support for Android 2.2, because GCC 4.6 has issues with OpenMP ([issue bytedeco/javacv#179](https://github.com/bytedeco/javacv/issues/179))
 * Resolve missing dependency for GSL on `windows-x86` by linking statically whatever it needs from `libgcc_s_dw2-1.dll`

### July 11, 2015 version 1.0
 * Build OpenCV, GSL, Leptonica, and Tesseract from source code on all platforms for better consistency in functionality across platforms
 * Add presets for CUDA 7.0 (including cuBLAS, cuDNN, cuFFT, cuRAND, cuSOLVER, cuSPARSE, and NPP)
 * Offer the Apache License, Version 2.0, as a new choice of license, in addition to the GPLv2 with Classpath exception
 * Add libvpx in the `cppbuild.sh` script for FFmpeg to support the WebM format ([issue #33](https://github.com/bytedeco/javacpp-presets/issues/33))
 * Upgrade presets for OpenCV 3.0.0, FFmpeg 2.7.1, OpenSSL 1.0.2d, x265 1.7, Leptonica 1.72, LLVM 3.6.1, and the latest of Caffe
 * Define commonly used OpenCV `std::vector` types (`PointVector`, `Point2fVector`, `Point2dVector`, `SizeVector`, `RectVector`, `KeyPointVector`, `DMatchVector`) for ease of use and performance reasons
 * Map `cv::saturate_cast<>()` in a more meaningful way ([issue #53](https://github.com/bytedeco/javacpp-presets/issues/53)) and name these functions more consistently
 * In addition to Leptonica and Tesseract, use only the officially supported GCC compiler for FFmpeg, FFTW, and GSL under Windows as well, to prevent compatibility issues ([issue bytedeco/javacv#137](https://github.com/bytedeco/javacv/issues/137))
 * Make `flycapture/cppbuild.sh` fail if FlyCapture is not found installed on the system ([issue #46](https://github.com/bytedeco/javacpp-presets/issues/46))
 * Patch libdc1394, libdcfreenect, FFTW, GSL, Leptonica and Tesseract with missing `@rpath` needed by Mac OS X ([issue #46](https://github.com/bytedeco/javacpp-presets/issues/46))

### April 4, 2015 version 0.11
 * Remove unneeded `@Opaque` types from `gsl` and replace them with their definitions whose names end with "_struct"
 * Segregate methods using `java.awt` classes into the new `Java2DFrameConverter` class of JavaCV ([issue #12](https://github.com/bytedeco/javacpp-presets/issues/12))
 * Emphasize the need to install separately the parent `pom.xml` file ([issue #42](https://github.com/bytedeco/javacpp-presets/issues/42))
 * Make CMake configurable via `CMAKE` variable in `cppbuild.sh` ([pull #41](https://github.com/bytedeco/javacpp-presets/pull/41))
 * Add presets for Caffe ([issue #34](https://github.com/bytedeco/javacpp-presets/issues/34))
 * Let `createBuffer()` return `UByteIndexer` and `UShortIndexer` when appropriate for unsigned data types
 * Remove the need to set manually the `platform.dependency` system property for downstream modules without a `cppbuild.sh` file
 * Fix failing `cppbuild.sh` for FFmpeg on Ubuntu ([issue #32](https://github.com/bytedeco/javacpp-presets/issues/32))
 * Disable iconv, XCB, and SDL for more portables builds of FFmpeg, but enable `x11grab` and `avfoundation` to allow screen capture ([issue #39](https://github.com/bytedeco/javacpp-presets/issues/39))
 * Avoid versioning of Leptonica and Tesseract libraries, preventing them from working on Android ([issue #38](https://github.com/bytedeco/javacpp-presets/issues/38))
 * Add x265 in the `cppbuild.sh` script for FFmpeg, thanks to Mark Bolstad ([issue bytedeco/javacv#41](https://github.com/bytedeco/javacv/issues/41))
 * Upgrade presets for OpenCV 2.4.11, FFmpeg 2.6.1, OpenSSL 1.0.2a, FlyCapture 2.7.3.19, libdc1394 2.2.3, libfreenect 0.5.2, LLVM 3.6.0
 * Switch from `IntPointer` to `BoolPointer` for the `BOOL*` pointer type of Leptonica ([issue #36](https://github.com/bytedeco/javacpp-presets/issues/36))
 * Add `preload` for `gif`, `jpeg`, `png`, `tiff`, and `webp` libraries in presets for Leptonica ([issue #36](https://github.com/bytedeco/javacpp-presets/issues/36))
 * Include missing `ltrresultiterator.h` header file in the presets for Tesseract ([issue #36](https://github.com/bytedeco/javacpp-presets/issues/36))
 * Append `@Documented` to annotation types to have them picked up by Javadoc

### December 23, 2014 version 0.10
 * Update instructions in the `README.md` file for manual installation in Android Studio
 * Include presets for Leptonica 1.71 and Tesseract 3.03-rc1 on Windows too
 * Fix `Mat.createFrom(BufferedImage)` ([issue #30](https://github.com/bytedeco/javacpp-presets/issues/30))
 * Add Speex, OpenCORE (AMR-NB and AMR-WB), and OpenSSL in the `cppbuild.sh` script for FFmpeg to support common RTMPS streams, among other things ([issue #2](https://github.com/bytedeco/javacpp-presets/issues/2) and [issue bytedeco/javacv#71](https://github.com/bytedeco/javacv/issues/71))
 * Deprecate slow `get()` and `put()` methods of `CvMat` in favor of the fast ones from `createIndexer()` ([issue javacv:317](http://code.google.com/p/javacv/issues/detail?id=317))
 * Include `operations.hpp` and `mat.hpp` in `opencv_core` to get a few important functions such as `read()` and `write()` for `FileStorage`
 * Replace `install_name_tool` hack to set `@rpath` on Mac OS X with patches to do it properly on install ([issue bytedeco/javacpp#6](https://github.com/bytedeco/javacpp/issues/6) and [issue bytedeco/javacv#49](https://github.com/bytedeco/javacv/issues/49))
 * Disable DocLint, which prevents the build from succeeding on Java 8 ([issue bytedeco/javacpp#5](https://github.com/bytedeco/javacpp/issues/5))
 * Disable OpenCL detection when building FFmpeg, causing link failures ([issue #19](https://github.com/bytedeco/javacpp-presets/issues/19))
 * Document a bit the `create()` factory methods in the `helper` package of the OpenCV module, and their relationship with the `release()` methods
 * Include new `createIndexer()` method in `CvMat`, `IplImage`, `Mat`, etc. for easy and efficient multidimensional access of data ([issue #317](http://code.google.com/p/javacv/issues/detail?id=317))
 * Deprecate `get*Buffer()` methods in favor of a better named and generic `createBuffer()` method
 * Fix `java.lang.UnsatisfiedLinkError` when allocating `opencv_core.Mat`, among others ([issue bytedeco/javacv#9](https://github.com/bytedeco/javacv/issues/9) and [issue bytedeco/javacv#28](https://github.com/bytedeco/javacv/issues/28))
 * Force OpenCV to build with GCC 4.6, as newer versions are known to hang on Android 2.2 ([issue android:43819](https://code.google.com/p/android/issues/detail?id=43819))
 * Upgrade presets for OpenCV 2.4.10, FFmpeg 2.5.1, FlyCapture 2.7.3.13, libfreenect 0.5.1, ARToolKitPlus 2.3.1, LLVM 3.5.0, and videoInput 0.200, where the latest code got merged into the master branch
 * Add callbacks for Tesseract according to new functionality in JavaCPP
 * Fix missing dependency of `opencv_contrib` on `opencv_nonfree` ([issue javacv:490](https://code.google.com/p/javacv/issues/detail?id=490))
 * Skip functions that are not actually implemented in `avdevice`, causing load failures on Android
 * Update presets for FFmpeg where `avcodec` now inherits from `swresample` ([issue #13](https://github.com/bytedeco/javacpp-presets/issues/13))
 * Add a `README.md` file to each presets with links to original project, Java API documentation, and sample usage
 * Add missing overloaded methods with `PointerPointer` parameters in LLVM module

### July 27, 2014 version 0.9
 * Add libmp3lame to FFmpeg builds ([issue javacv:411](https://code.google.com/p/javacv/issues/detail?id=448))
 * Upgrade presets for FFmpeg 2.3, FlyCapture 2.6.3.4 ([pull #6](https://github.com/bytedeco/javacpp-presets/pull/6), [issue #8](https://github.com/bytedeco/javacpp-presets/issues/8)), libfreenect 0.5
 * Make the `cppbuild.sh` scripts install native libraries inside the `cppbuild` subdirectories, instead of on the system
 * Include new `platform.dependency` and `platform.dependencies` properties to let users depend easily on the artifacts that contain native libraries
 * Add presets for flandmark 1.07 ([pull #9](https://github.com/bytedeco/javacpp-presets/pull/9)), FFTW 3.3.4, GSL 1.16, LLVM 3.4.2, Leptonica 1.71, Tesseract 3.03-rc1
 * Fix missing `static` keyword on methods annotated with an `@Adapter` ([issue #3](https://github.com/bytedeco/javacpp-presets/issues/3))
 * Turn `Mat.createFrom()` into a static factory method, and make `Mat.copyFrom()` call `Mat.create()` as appropriate ([issue #1](https://github.com/bytedeco/javacpp-presets/issues/1))
 * Add missing `native_camera` modules of `opencv_highgui` for Android
 * Fix functions from `opencv_stitching` not accepting a `MatVector` as apparently intended by the API (issue javacv:466)

### April 28, 2014 version 0.8
 * Move from Google Code to GitHub as main source code repository
 * Rename the `com.googlecode.javacpp` package to `org.bytedeco.javacpp`
 * Appended the version of the parent artifact to the ones of the child modules, in an effort to avoid conflicts
 * Updated `cppbuild.sh` scripts with support for the "android-x86" platform (issue javacv:411), thanks to Xavier Hallade
 * Added presets for PGR FlyCapture 1.7
 * Fixed compilation errors on Android, Mac OS X, and Windows
 * Upgraded to OpenCV 2.4.9, FFmpeg 2.2.1, libdc1394 2.2.2, and libfreenect 0.4
 * Introduced build scripts, based on the CPPJARs package of JavaCV, to install native C/C++ libraries
 * Ported various helper classes and methods from JavaCV
 * Inserted missing dependency entries in the `pom.xml` files of the child modules
 * Added presets for the C++ API of OpenCV 2.4.8, which can now be parsed due to the latest changes in JavaCPP

### January 6, 2014 version 0.7
 * Fixed JavaCPP properties not getting set by the parent `pom.xml` file
 * Added presets for the C API of OpenCV 2.4.8, FFmpeg 2.1.x, libfreenect 0.2 (OpenKinect), videoInput 0.200, and ARToolkitPlus 2.3.0

### September 15, 2013 version 0.6
Initial release


Acknowledgments
---------------
This project was conceived at the [Okutomi & Tanaka Laboratory](http://www.ok.ctrl.titech.ac.jp/), Tokyo Institute of Technology, where I was supported for my doctoral research program by a generous scholarship from the Ministry of Education, Culture, Sports, Science and Technology (MEXT) of the Japanese Government. I extend my gratitude further to all who have reported bugs, donated code, or made suggestions for improvements (details above)!
