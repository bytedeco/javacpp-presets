/*
 * Copyright (C) 2015-2019 Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.bytedeco.caffe.presets;

import java.util.List;
import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

import org.bytedeco.hdf5.presets.*;
import org.bytedeco.opencv.presets.*;
import org.bytedeco.openblas.presets.*;

/**
 *
 * @author Samuel Audet
 */
@Properties(inherit = {opencv_highgui.class, hdf5.class, openblas.class}, target = "org.bytedeco.caffe", global = "org.bytedeco.caffe.global.caffe", value = {
    @Platform(value = {"linux", "macosx"}, compiler = "cpp11", define = {"NDEBUG", "CPU_ONLY", "SHARED_PTR_NAMESPACE boost", "USE_LEVELDB", "USE_LMDB", "USE_OPENCV"}, include = {"caffe/caffe.hpp",
        "caffe/util/device_alternate.hpp", "google/protobuf/stubs/common.h", "google/protobuf/arena.h", "google/protobuf/descriptor.h", "google/protobuf/message_lite.h", "google/protobuf/message.h", "caffe/common.hpp",
        "google/protobuf/generated_message_table_driven.h", "caffe/proto/caffe.pb.h", "caffe/util/blocking_queue.hpp", /*"caffe/data_reader.hpp",*/ "caffe/util/math_functions.hpp", "caffe/syncedmem.hpp",
        "caffe/blob.hpp", "caffe/data_transformer.hpp", "caffe/filler.hpp", "caffe/internal_thread.hpp", "caffe/util/hdf5.hpp", "caffe/layers/base_data_layer.hpp", "caffe/layers/data_layer.hpp",
        "caffe/layers/dummy_data_layer.hpp", "caffe/layers/hdf5_data_layer.hpp", "caffe/layers/hdf5_output_layer.hpp", "caffe/layers/image_data_layer.hpp", "caffe/layers/memory_data_layer.hpp",
        "caffe/layers/window_data_layer.hpp", "caffe/layer_factory.hpp", "caffe/layer.hpp", "caffe/layers/accuracy_layer.hpp", "caffe/layers/loss_layer.hpp", "caffe/layers/contrastive_loss_layer.hpp",
        "caffe/layers/euclidean_loss_layer.hpp", "caffe/layers/hinge_loss_layer.hpp", "caffe/layers/infogain_loss_layer.hpp", "caffe/layers/multinomial_logistic_loss_layer.hpp",
        "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp", "caffe/layers/softmax_loss_layer.hpp", "caffe/layers/neuron_layer.hpp", "caffe/layers/absval_layer.hpp", "caffe/layers/bnll_layer.hpp",
        "caffe/layers/dropout_layer.hpp", "caffe/layers/exp_layer.hpp", "caffe/layers/log_layer.hpp", "caffe/layers/power_layer.hpp", "caffe/layers/relu_layer.hpp", "caffe/layers/cudnn_relu_layer.hpp",
        "caffe/layers/sigmoid_layer.hpp", "caffe/layers/cudnn_sigmoid_layer.hpp", "caffe/layers/tanh_layer.hpp", "caffe/layers/cudnn_tanh_layer.hpp", "caffe/layers/threshold_layer.hpp",
        "caffe/layers/prelu_layer.hpp", "caffe/layers/argmax_layer.hpp", "caffe/layers/batch_norm_layer.hpp", "caffe/layers/batch_reindex_layer.hpp", "caffe/layers/concat_layer.hpp",
        "caffe/layers/eltwise_layer.hpp", "caffe/layers/embed_layer.hpp", "caffe/layers/filter_layer.hpp", "caffe/layers/flatten_layer.hpp", "caffe/layers/inner_product_layer.hpp",
        "caffe/layers/mvn_layer.hpp", "caffe/layers/reshape_layer.hpp", "caffe/layers/reduction_layer.hpp", "caffe/layers/silence_layer.hpp", "caffe/layers/softmax_layer.hpp",
        "caffe/layers/cudnn_softmax_layer.hpp", "caffe/layers/split_layer.hpp", "caffe/layers/slice_layer.hpp", "caffe/layers/tile_layer.hpp", "caffe/net.hpp", "caffe/parallel.hpp",
        "caffe/solver.hpp", "caffe/solver_factory.hpp", "caffe/sgd_solvers.hpp", "caffe/layers/input_layer.hpp", "caffe/layers/parameter_layer.hpp", "caffe/layers/base_conv_layer.hpp",
        "caffe/layers/conv_layer.hpp", "caffe/layers/crop_layer.hpp", "caffe/layers/deconv_layer.hpp", "caffe/layers/cudnn_conv_layer.hpp", "caffe/layers/im2col_layer.hpp",
        "caffe/layers/lrn_layer.hpp", "caffe/layers/cudnn_lrn_layer.hpp", "caffe/layers/cudnn_lcn_layer.hpp", "caffe/layers/pooling_layer.hpp", "caffe/layers/cudnn_pooling_layer.hpp",
        "caffe/layers/spp_layer.hpp", "caffe/layers/recurrent_layer.hpp", "caffe/layers/lstm_layer.hpp", "caffe/layers/rnn_layer.hpp", "caffe/util/benchmark.hpp", "caffe/util/db.hpp",
        "caffe/util/db_leveldb.hpp", "caffe/util/db_lmdb.hpp", "caffe/util/io.hpp", "caffe/util/rng.hpp", "caffe/util/im2col.hpp", "caffe/util/insert_splits.hpp", "caffe/util/mkl_alternate.hpp",
        "caffe/util/upgrade_proto.hpp", /* "caffe/util/cudnn.hpp" */}, link = "caffe@.1.0.0", /*resource = {"include", "lib"},*/ includepath = {"/usr/local/cuda/include/",
        "/System/Library/Frameworks/vecLib.framework/", "/System/Library/Frameworks/Accelerate.framework/"}, linkpath = "/usr/local/cuda/lib/", resource = {"include", "lib"}, linkresource = "lib"),
    @Platform(value = {"linux-arm64", "linux-ppc64le", "linux-x86_64", "macosx-x86_64"}, define = {"SHARED_PTR_NAMESPACE boost", "USE_LEVELDB", "USE_LMDB", "USE_OPENCV", "USE_CUDNN"}, extension = "-gpu") })
public class caffe implements LoadEnabled, InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "caffe"); }

    @Override public void init(ClassProperties properties) {
        String platform = properties.getProperty("platform");
        String extension = properties.getProperty("platform.extension");
        List<String> preloads = properties.get("platform.preload");
        List<String> resources = properties.get("platform.preloadresource");

        // Only apply this at load time since we don't want to copy the CUDA libraries here
        if (!Loader.isLoadLibraries() || extension == null || !extension.equals("-gpu")) {
            return;
        }
        int i = 0;
        String[] libs = {"cudart", "cublasLt", "cublas", "curand", "cudnn",
                         "cudnn_ops_infer", "cudnn_ops_train", "cudnn_adv_infer",
                         "cudnn_adv_train", "cudnn_cnn_infer", "cudnn_cnn_train"};
        for (String lib : libs) {
            if (platform.startsWith("linux")) {
                lib += lib.startsWith("cudnn") ? "@.8" : lib.equals("curand") ? "@.10" : lib.equals("cudart") ? "@.11.0" : "@.11";
            } else if (platform.startsWith("windows")) {
                lib += lib.startsWith("cudnn") ? "64_8" : lib.equals("curand") ? "64_10" : lib.equals("cudart") ? "64_110" : "64_11";
            } else {
                continue; // no CUDA
            }
            if (!preloads.contains(lib)) {
                preloads.add(i++, lib);
            }
        }
        if (i > 0) {
            resources.add("/org/bytedeco/cuda/");
        }
    }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("LIBPROTOBUF_EXPORT", "LIBPROTOC_EXPORT", "GOOGLE_PROTOBUF_ATTRIBUTE_NOINLINE", "GOOGLE_PROTOBUF_VERIFY_VERSION", "GOOGLE_ATTRIBUTE_ALWAYS_INLINE",
                             "GOOGLE_ATTRIBUTE_DEPRECATED", "GOOGLE_DLOG", "NOT_IMPLEMENTED", "NO_GPU", "CUDA_POST_KERNEL_CHECK", "PROTOBUF_CONSTEXPR", "PROTOBUF_CONSTEXPR_VAR",
                             "PROTOBUF_EXPORT", "PROTOBUF_ATTRIBUTE_REINITIALIZES", "PROTOBUF_NOINLINE").cppTypes().annotations())
               .put(new Info("NDEBUG", "CPU_ONLY", "GFLAGS_GFLAGS_H_", "SWIG", "USE_CUDNN").define())
               .put(new Info("defined(_WIN32) && defined(GetMessage)", "LANG_CXX11", "GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER").define(false))
               .put(new Info("cublasHandle_t", "curandGenerator_t").cast().valueTypes("Pointer"))
               .put(new Info("CBLAS_TRANSPOSE", "cublasStatus_t", "curandStatus_t", "hid_t", "cudnnStatus_t", "cudnnDataType_t").cast().valueTypes("int"))
               .put(new Info("std::string").annotations("@StdString").valueTypes("BytePointer", "String").pointerTypes("@Cast({\"char*\", \"std::string*\"}) BytePointer"))
               .put(new Info("std::vector<std::string>").pointerTypes("StringVector").define())
               .put(new Info("std::vector<const google::protobuf::FieldDescriptor*>").pointerTypes("FieldDescriptorVector").define())
               .put(new Info("std::vector<caffe::Datum>").pointerTypes("DatumVector").define())
               .put(new Info("caffe::BlockingQueue<caffe::Datum*>").pointerTypes("DatumBlockingQueue"))

               .put(new Info("google::protobuf::int8", "google::protobuf::uint8").cast().valueTypes("byte").pointerTypes("BytePointer", "ByteBuffer", "byte[]"))
               .put(new Info("google::protobuf::int16", "google::protobuf::uint16").cast().valueTypes("short").pointerTypes("ShortPointer", "ShortBuffer", "short[]"))
               .put(new Info("google::protobuf::int32", "google::protobuf::uint32").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"))
               .put(new Info("google::protobuf::int64", "google::protobuf::uint64").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))
               .put(new Info("std::pair<google::protobuf::uint64,google::protobuf::uint64>").pointerTypes("LongLongPair").define())
               .put(new Info("leveldb::Iterator", "leveldb::DB", "MDB_txn", "MDB_cursor", "MDB_dbi", "MDB_env", "boost::mt19937").cast().pointerTypes("Pointer"))
               .put(new Info("google::protobuf::internal::CompileAssert", "google::protobuf::internal::ExplicitlyConstructed", "google::protobuf::MessageFactory::InternalRegisterGeneratedFile",
                             "google::protobuf::internal::LogMessage", "google::protobuf::internal::LogFinisher", "google::protobuf::LogHandler",
                             "google::protobuf::internal::FieldMetadata", "google::protobuf::internal::SerializationTable", "google::protobuf::internal::proto3_preserve_unknown_",
                             "google::protobuf::internal::MergePartialFromImpl", "google::protobuf::internal::UnknownFieldParse", "google::protobuf::internal::WriteLengthDelimited",
                             "google::protobuf::is_proto_enum", "google::protobuf::GetEnumDescriptor", "protobuf_caffe_2eproto::TableStruct", "TableStruct_caffe_2eproto",
                             "google::protobuf::RepeatedField", "google::protobuf::RepeatedPtrField", "boost::mutex").skip())

               .put(new Info("caffe::AccuracyParameterDefaultTypeInternal", "caffe::ArgMaxParameterDefaultTypeInternal", "caffe::BatchNormParameterDefaultTypeInternal",
                             "caffe::BiasParameterDefaultTypeInternal", "caffe::BlobProtoDefaultTypeInternal", "caffe::BlobProtoVectorDefaultTypeInternal", "caffe::BlobShapeDefaultTypeInternal",
                             "caffe::ConcatParameterDefaultTypeInternal", "caffe::ContrastiveLossParameterDefaultTypeInternal", "caffe::ConvolutionParameterDefaultTypeInternal",
                             "caffe::CropParameterDefaultTypeInternal", "caffe::DataParameterDefaultTypeInternal", "caffe::DatumDefaultTypeInternal", "caffe::DropoutParameterDefaultTypeInternal",
                             "caffe::DummyDataParameterDefaultTypeInternal", "caffe::ELUParameterDefaultTypeInternal", "caffe::EltwiseParameterDefaultTypeInternal", "caffe::EmbedParameterDefaultTypeInternal",
                             "caffe::ExpParameterDefaultTypeInternal", "caffe::FillerParameterDefaultTypeInternal", "caffe::FlattenParameterDefaultTypeInternal", "caffe::HDF5DataParameterDefaultTypeInternal",
                             "caffe::HDF5OutputParameterDefaultTypeInternal", "caffe::HingeLossParameterDefaultTypeInternal", "caffe::ImageDataParameterDefaultTypeInternal",
                             "caffe::InfogainLossParameterDefaultTypeInternal", "caffe::InnerProductParameterDefaultTypeInternal", "caffe::InputParameterDefaultTypeInternal",
                             "caffe::LRNParameterDefaultTypeInternal", "caffe::LayerParameterDefaultTypeInternal", "caffe::LogParameterDefaultTypeInternal", "caffe::LossParameterDefaultTypeInternal",
                             "caffe::MVNParameterDefaultTypeInternal", "caffe::MemoryDataParameterDefaultTypeInternal", "caffe::NetParameterDefaultTypeInternal", "caffe::NetStateDefaultTypeInternal",
                             "caffe::NetStateRuleDefaultTypeInternal", "caffe::PReLUParameterDefaultTypeInternal", "caffe::ParamSpecDefaultTypeInternal", "caffe::ParameterParameterDefaultTypeInternal",
                             "caffe::PoolingParameterDefaultTypeInternal", "caffe::PowerParameterDefaultTypeInternal", "caffe::PythonParameterDefaultTypeInternal", "caffe::ReLUParameterDefaultTypeInternal",
                             "caffe::RecurrentParameterDefaultTypeInternal", "caffe::ReductionParameterDefaultTypeInternal", "caffe::ReshapeParameterDefaultTypeInternal", "caffe::SPPParameterDefaultTypeInternal",
                             "caffe::ScaleParameterDefaultTypeInternal", "caffe::SigmoidParameterDefaultTypeInternal", "caffe::SliceParameterDefaultTypeInternal", "caffe::SoftmaxParameterDefaultTypeInternal",
                             "caffe::SolverParameterDefaultTypeInternal", "caffe::SolverStateDefaultTypeInternal", "caffe::TanHParameterDefaultTypeInternal", "caffe::ThresholdParameterDefaultTypeInternal",
                             "caffe::TileParameterDefaultTypeInternal", "caffe::TransformationParameterDefaultTypeInternal", "caffe::V0LayerParameterDefaultTypeInternal",
                             "caffe::V1LayerParameterDefaultTypeInternal", "caffe::WindowDataParameterDefaultTypeInternal").skip());

        String[] functionTemplates = { "caffe_cpu_gemm", "caffe_cpu_gemv", "caffe_axpy", "caffe_cpu_axpby", "caffe_copy", "caffe_set", "caffe_add_scalar",
                "caffe_scal", "caffe_sqr", "caffe_add", "caffe_sub", "caffe_mul", "caffe_div", "caffe_powx", "caffe_nextafter", "caffe_rng_uniform",
                "caffe_rng_gaussian", "caffe_rng_bernoulli", "caffe_rng_bernoulli", "caffe_exp", "caffe_log", "caffe_abs", "caffe_cpu_dot", "caffe_cpu_strided_dot",
                "caffe_cpu_hamming_distance", "caffe_cpu_asum", "caffe_sign", "caffe_cpu_scale", "caffe_gpu_gemm", "caffe_gpu_gemv", "caffe_gpu_axpy",
                "caffe_gpu_axpby", "caffe_gpu_memcpy", "caffe_gpu_set", "caffe_gpu_memset", "caffe_gpu_add_scalar", "caffe_gpu_scal", "caffe_gpu_add",
                "caffe_gpu_sub", "caffe_gpu_mul", "caffe_gpu_div", "caffe_gpu_abs", "caffe_gpu_exp", "caffe_gpu_log", "caffe_gpu_powx", "caffe_gpu_rng_uniform",
                "caffe_gpu_rng_gaussian", "caffe_gpu_rng_bernoulli", "caffe_gpu_dot", "caffe_gpu_hamming_distance", "caffe_gpu_asum", "caffe_gpu_sign",
                "caffe_gpu_sgnbit", "caffe_gpu_fabs", "caffe_gpu_scale", "hdf5_load_nd_dataset_helper", "hdf5_load_nd_dataset", "hdf5_save_nd_dataset",
                "im2col_nd_cpu", "im2col_cpu", "col2im_nd_cpu", "col2im_cpu", "im2col_nd_gpu", "im2col_gpu", "col2im_nd_gpu", "col2im_gpu" };
        for (String t : functionTemplates) {
            infoMap.put(new Info("caffe::" + t + "<float>").javaNames(t + "_float"))
                   .put(new Info("caffe::" + t + "<double>").javaNames(t + "_double"));
        }

        String classTemplates[] = { "Blob", "DataTransformer", "Filler", "ConstantFiller", "UniformFiller", "GaussianFiller", "PositiveUnitballFiller", "XavierFiller", "MSRAFiller", "BilinearFiller",
                "BaseDataLayer", "Batch", "BasePrefetchingDataLayer", "DataLayer", "DummyDataLayer", "HDF5DataLayer", "HDF5OutputLayer", "ImageDataLayer", "MemoryDataLayer",
                "WindowDataLayer", "Layer", "LayerRegistry", "LayerRegisterer", "AccuracyLayer", "LossLayer", "ContrastiveLossLayer", "EuclideanLossLayer", "HingeLossLayer",
                "InfogainLossLayer", "MultinomialLogisticLossLayer", "SigmoidCrossEntropyLossLayer", "SoftmaxWithLossLayer", "NeuronLayer", "AbsValLayer", "BNLLLayer",
                "DropoutLayer", "ExpLayer", "PowerLayer", "ReLULayer", "SigmoidLayer", "TanHLayer", "ThresholdLayer", "PReLULayer", "PythonLayer", "ArgMaxLayer", "BatchNormLayer",
                "BatchReindexLayer", "ConcatLayer", "EltwiseLayer", "EmbedLayer", "FilterLayer", "FlattenLayer", "InnerProductLayer", "MVNLayer", "ReshapeLayer", "ReductionLayer",
                "SilenceLayer", "SoftmaxLayer", "SplitLayer", "SliceLayer", "TileLayer", "Net", "Solver", "WorkerSolver", "SolverRegistry", "SolverRegisterer", "SGDSolver", "NesterovSolver",
                "AdaGradSolver", "RMSPropSolver", "AdaDeltaSolver", "AdamSolver",  "InputLayer", "ParameterLayer", "BaseConvolutionLayer", "ConvolutionLayer", "CropLayer", "DeconvolutionLayer",
                "Im2colLayer", "LRNLayer", "PoolingLayer", "SPPLayer", "RecurrentLayer", "LSTMLayer", "RNNLayer", "CuDNNLCNLayer", "CuDNNLRNLayer",
                "CuDNNReLULayer", "CuDNNSigmoidLayer", "CuDNNTanHLayer", "CuDNNSoftmaxLayer", "CuDNNConvolutionLayer", "CuDNNPoolingLayer" };
        for (String t : classTemplates) {
            boolean purify = t.equals("BaseDataLayer") || t.equals("LossLayer") || t.equals("NeuronLayer");
            boolean virtualize = t.endsWith("Layer") || t.endsWith("Solver");
            String[] annotations = t.startsWith("CuDNN") ? new String[] {"@Platform(value = {\"linux-x86_64\", \"macosx-x86_64\"}, extension = \"-gpu\")"} : null;
            infoMap.put(new Info("caffe::" + t + "<float>").annotations(annotations).pointerTypes("Float" + t).purify(purify).virtualize(virtualize))
                   .put(new Info("caffe::" + t + "<double>").annotations(annotations).pointerTypes("Double" + t).purify(purify).virtualize(virtualize));
        }
        infoMap.put(new Info("caffe::BasePrefetchingDataLayer<float>::InternalThreadEntry()",
                             "caffe::BasePrefetchingDataLayer<double>::InternalThreadEntry()").skip())

               .put(new Info("caffe::Solver<float>::Solve(std::string)", "caffe::Solver<double>::Solve(std::string)").javaText(
                             "public void Solve(String resume_file) { Solve(new BytePointer(resume_file)); }\n"
                           + "public void Solve() { Solve((BytePointer)null); }"))

               .put(new Info("caffe::Batch<float>::data_").javaText("@MemberGetter public native @ByRef FloatBlob data_();"))
               .put(new Info("caffe::Batch<double>::data_").javaText("@MemberGetter public native @ByRef DoubleBlob data_();"))
               .put(new Info("caffe::Batch<float>::label_").javaText("@MemberGetter public native @ByRef FloatBlob label_();"))
               .put(new Info("caffe::Batch<double>::label_").javaText("@MemberGetter public native @ByRef DoubleBlob label_();"))

               .put(new Info("caffe::GetFiller<float>").javaNames("GetFloatFiller"))
               .put(new Info("caffe::GetFiller<double>").javaNames("GetDoubleFiller"))
               .put(new Info("caffe::GetSolver<float>").javaNames("GetFloatSolver"))
               .put(new Info("caffe::GetSolver<double>").javaNames("GetDoubleSolver"))

               .put(new Info("boost::shared_ptr<caffe::Blob<float> >").annotations("@SharedPtr").pointerTypes("FloatBlob"))
               .put(new Info("boost::shared_ptr<caffe::Blob<double> >").annotations("@SharedPtr").pointerTypes("DoubleBlob"))
               .put(new Info("std::vector<boost::shared_ptr<caffe::Blob<float> > >").pointerTypes("FloatBlobSharedVector").define())
               .put(new Info("std::vector<boost::shared_ptr<caffe::Blob<double> > >").pointerTypes("DoubleBlobSharedVector").define())

               .put(new Info("boost::shared_ptr<caffe::Layer<float> >").annotations("@Cast({\"\", \"boost::shared_ptr<caffe::Layer<float> >\"}) @SharedPtr").pointerTypes("FloatLayer"))
               .put(new Info("boost::shared_ptr<caffe::Layer<double> >").annotations("@Cast({\"\", \"boost::shared_ptr<caffe::Layer<double> >\"}) @SharedPtr").pointerTypes("DoubleLayer"))
               .put(new Info("std::vector<boost::shared_ptr<caffe::Layer<float> > >").pointerTypes("FloatLayerSharedVector").define())
               .put(new Info("std::vector<boost::shared_ptr<caffe::Layer<double> > >").pointerTypes("DoubleLayerSharedVector").define())

               .put(new Info("boost::shared_ptr<caffe::Net<float> >").annotations("@SharedPtr").pointerTypes("FloatNet"))
               .put(new Info("boost::shared_ptr<caffe::Net<double> >").annotations("@SharedPtr").pointerTypes("DoubleNet"))
               .put(new Info("std::vector<boost::shared_ptr<caffe::Net<float> > >").pointerTypes("FloatNetSharedVector").define())
               .put(new Info("std::vector<boost::shared_ptr<caffe::Net<double> > >").pointerTypes("DoubleNetSharedVector").define())

               .put(new Info("std::vector<caffe::Blob<float>*>").pointerTypes("FloatBlobVector").define())
               .put(new Info("std::vector<caffe::Blob<double>*>").pointerTypes("DoubleBlobVector").define())
               .put(new Info("std::vector<std::vector<caffe::Blob<float>*> >").pointerTypes("FloatBlobVectorVector").define())
               .put(new Info("std::vector<std::vector<caffe::Blob<double>*> >").pointerTypes("DoubleBlobVectorVector").define())

               .put(new Info("caffe::LayerRegistry<float>::Creator").valueTypes("FloatLayerRegistry.Creator"))
               .put(new Info("caffe::LayerRegistry<double>::Creator").valueTypes("DoubleLayerRegistry.Creator"))
               .put(new Info("std::map<std::string,caffe::LayerRegistry<float>::Creator>").pointerTypes("FloatRegistry").define())
               .put(new Info("std::map<std::string,caffe::LayerRegistry<double>::Creator>").pointerTypes("DoubleRegistry").define())

               .put(new Info("std::vector<bool>").pointerTypes("BoolVector").define())
               .put(new Info("std::vector<std::vector<bool> >").pointerTypes("BoolVectorVector").define())
               .put(new Info("std::map<std::string,int>").pointerTypes("StringIntMap").define())

               .put(new Info("caffe::Net<float>::layer_by_name").javaText(
                       "public FloatLayer layer_by_name(BytePointer layer_name) { return layer_by_name(FloatLayer.class, layer_name); }\n"
                     + "public FloatLayer layer_by_name(String layer_name) { return layer_by_name(FloatLayer.class, layer_name); };\n"
                     + "public native @Const @Cast({\"\", \"boost::shared_ptr<caffe::Layer<float> >\"}) @SharedPtr @ByVal <L extends FloatLayer> L layer_by_name(Class<L> cls, @StdString BytePointer layer_name);\n"
                     + "public native @Const @Cast({\"\", \"boost::shared_ptr<caffe::Layer<float> >\"}) @SharedPtr @ByVal <L extends FloatLayer> L layer_by_name(Class<L> cls, @StdString String layer_name);\n"))
               .put(new Info("caffe::Net<double>::layer_by_name").javaText(
                       "public DoubleLayer layer_by_name(BytePointer layer_name) { return layer_by_name(DoubleLayer.class, layer_name); }\n"
                     + "public DoubleLayer layer_by_name(String layer_name) { return layer_by_name(DoubleLayer.class, layer_name); };\n"
                     + "public native @Const @Cast({\"\", \"boost::shared_ptr<caffe::Layer<double> >\"}) @SharedPtr @ByVal <L extends DoubleLayer> L layer_by_name(Class<L> cls, @StdString BytePointer layer_name);\n"
                     + "public native @Const @Cast({\"\", \"boost::shared_ptr<caffe::Layer<double> >\"}) @SharedPtr @ByVal <L extends DoubleLayer> L layer_by_name(Class<L> cls, @StdString String layer_name);\n"))

               .put(new Info("caffe::Net<float>::Callback").pointerTypes("FloatNet.Callback"))
               .put(new Info("caffe::Solver<float>::Callback").pointerTypes("FloatSolver.Callback"))
               .put(new Info("std::vector<caffe::Solver<float>::Callback*>").pointerTypes("FloatCallbackVector").define())
               .put(new Info("caffe::Net<double>::Callback").pointerTypes("DoubleNet.Callback"))
               .put(new Info("caffe::Solver<double>::Callback").pointerTypes("DoubleSolver.Callback"))
               .put(new Info("std::vector<caffe::Solver<double>::Callback*>").pointerTypes("DoubleCallbackVector").define())
               .put(new Info("boost::function<caffe::SolverAction::Enum()>").pointerTypes("ActionCallback"));
    }

    public static class ActionCallback extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    ActionCallback(Pointer p) { super(p); }
        protected ActionCallback() { allocate(); }
        private native void allocate();
        public native @Cast("caffe::SolverAction::Enum") int call();
    }
}
