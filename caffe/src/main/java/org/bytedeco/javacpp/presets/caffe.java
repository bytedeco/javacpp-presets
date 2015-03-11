/*
 * Copyright (C) 2015 Samuel Audet
 *
 * This file is part of JavaCPP.
 *
 * JavaCPP is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version (subject to the "Classpath" exception
 * as provided in the LICENSE.txt file that accompanied this code).
 *
 * JavaCPP is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with JavaCPP.  If not, see <http://www.gnu.org/licenses/>.
 */

package org.bytedeco.javacpp.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(inherit=opencv_core.class, target="org.bytedeco.javacpp.caffe", value={
    @Platform(define="SHARED_PTR_NAMESPACE boost", include={"caffe/caffe.hpp", "caffe/util/device_alternate.hpp",
        "caffe/common.hpp", "caffe/proto/caffe.pb.h", "caffe/util/math_functions.hpp", "caffe/syncedmem.hpp", "caffe/blob.hpp",
        "caffe/data_transformer.hpp", "caffe/filler.hpp", "caffe/internal_thread.hpp", "caffe/data_layers.hpp", // "caffe/layer_factory.hpp",
        "caffe/layer.hpp", "caffe/loss_layers.hpp", "caffe/neuron_layers.hpp", "caffe/common_layers.hpp", "caffe/net.hpp", "caffe/solver.hpp",
        "caffe/vision_layers.hpp", "caffe/util/benchmark.hpp", "caffe/util/db.hpp", "caffe/util/io.hpp", "caffe/util/rng.hpp"},
        includepath="/usr/local/cuda/include", link="caffe") })
public class caffe implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("NOT_IMPLEMENTED", "NO_GPU", "CUDA_POST_KERNEL_CHECK").cppTypes().annotations())
               .put(new Info("GFLAGS_GFLAGS_H_").define())
               .put(new Info("cublasHandle_t", "curandGenerator_t").cast().valueTypes("Pointer"))
               .put(new Info("CBLAS_TRANSPOSE", "cublasStatus_t", "curandStatus_t", "hid_t").cast().valueTypes("int"))
               .put(new Info("std::string").valueTypes("@StdString BytePointer", "@StdString String").pointerTypes("@Cast(\"std::string*\") Pointer"))

               .put(new Info("google::protobuf::int8", "google::protobuf::uint8").cast().valueTypes("byte").pointerTypes("BytePointer", "ByteBuffer", "byte[]"))
               .put(new Info("google::protobuf::int16", "google::protobuf::uint16").cast().valueTypes("short").pointerTypes("ShortPointer", "ShortBuffer", "short[]"))
               .put(new Info("google::protobuf::int32", "google::protobuf::uint32").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"))
               .put(new Info("google::protobuf::int64", "google::protobuf::uint64").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))
               .put(new Info("google::protobuf::Descriptor", "google::protobuf::EnumDescriptor", "google::protobuf::Message", "google::protobuf::Metadata",
                             "google::protobuf::UnknownFieldSet", "google::protobuf::io::CodedInputStream", "google::protobuf::io::CodedOutputStream",
                             "leveldb::Iterator", "leveldb::DB", "MDB_txn", "MDB_cursor", "MDB_dbi", "boost::mt19937").cast().pointerTypes("Pointer"))
               .put(new Info("google::protobuf::RepeatedField", "google::protobuf::RepeatedPtrField").skip());

        String[] functionTemplates = { "caffe_cpu_gemv", "caffe_axpy", "caffe_cpu_axpby", "caffe_copy", "caffe_set", "caffe_memset", "caffe_add_scalar",
                "caffe_scal", "caffe_sqr", "caffe_add", "caffe_sub", "caffe_mul", "caffe_div", "caffe_powx", "caffe_nextafter", "caffe_rng_uniform",
                "caffe_rng_gaussian", "caffe_rng_bernoulli", "caffe_rng_bernoulli", "caffe_exp", "caffe_abs", "caffe_cpu_dot", "caffe_cpu_strided_dot",
                "caffe_cpu_hamming_distance", "caffe_cpu_asum", "caffe_sign", "caffe_cpu_scale", "caffe_gpu_gemm", "caffe_gpu_gemv", "caffe_gpu_axpy",
                "caffe_gpu_axpby", "caffe_gpu_memcpy", "caffe_gpu_set", "caffe_gpu_memset", "caffe_gpu_add_scalar", "caffe_gpu_scal", "caffe_gpu_add",
                "caffe_gpu_sub", "caffe_gpu_mul", "caffe_gpu_div", "caffe_gpu_abs", "caffe_gpu_exp", "caffe_gpu_powx", "caffe_gpu_rng_uniform",
                "caffe_gpu_rng_gaussian", "caffe_gpu_rng_bernoulli", "caffe_gpu_dot", "caffe_gpu_hamming_distance", "caffe_gpu_asum", "caffe_gpu_sign",
                "caffe_gpu_sgnbit", "caffe_gpu_fabs", "caffe_gpu_scale", "hdf5_load_nd_dataset_helper", "hdf5_load_nd_dataset", "hdf5_save_nd_dataset" };
        for (String t : functionTemplates) {
            infoMap.put(new Info("caffe::" + t + "<float>").javaNames(t + "_float"))
                   .put(new Info("caffe::" + t + "<double>").javaNames(t + "_double"));
        }

        String classTemplates[] = { "Blob", "DataTransformer", "Filler", "ConstantFiller", "UniformFiller", "GaussianFiller", "PositiveUnitballFiller", "XavierFiller",
                "BaseDataLayer", "BasePrefetchingDataLayer", "DataLayer", "DummyDataLayer", "HDF5DataLayer", "HDF5OutputLayer", "ImageDataLayer", "MemoryDataLayer",
                "WindowDataLayer", "Layer", "LayerRegistry", "LayerRegisterer", "AccuracyLayer", "LossLayer", "ContrastiveLossLayer", "EuclideanLossLayer", "HingeLossLayer",
                "InfogainLossLayer", "MultinomialLogisticLossLayer", "SigmoidCrossEntropyLossLayer", "SoftmaxWithLossLayer", "NeuronLayer", "AbsValLayer", "BNLLLayer",
                "DropoutLayer", "ExpLayer", "PowerLayer", "ReLULayer", "SigmoidLayer", "TanHLayer", "ThresholdLayer", "ArgMaxLayer", "ConcatLayer", "EltwiseLayer",
                "FlattenLayer", "InnerProductLayer", "MVNLayer", "SilenceLayer", "SoftmaxLayer", "SplitLayer", "SliceLayer", "Net", "Solver", "SGDSolver", "NesterovSolver",
                "AdaGradSolver", "BaseConvolutionLayer", "ConvolutionLayer", "DeconvolutionLayer", "Im2colLayer", "LRNLayer", "PoolingLayer",
                /* "CuDNNReLULayer", "CuDNNSigmoidLayer", "CuDNNTanHLayer", "CuDNNSoftmaxLayer", "CuDNNConvolutionLayer", "CuDNNPoolingLayer" */ };
        for (String t : classTemplates) {
            boolean virtualize = t.equals("BaseDataLayer") || t.equals("NeuronLayer") || t.equals("LossLayer");
            infoMap.put(new Info("caffe::" + t + "<float>").pointerTypes("Float" + t).virtualize(virtualize))
                   .put(new Info("caffe::" + t + "<double>").pointerTypes("Double" + t).virtualize(virtualize));
        }

        infoMap.put(new Info("caffe::GetFiller<float>").javaNames("GetFloatFiller"))
               .put(new Info("caffe::GetFiller<double>").javaNames("GetDoubleFiller"))
               .put(new Info("caffe::GetSolver<float>").javaNames("GetFloatSolver"))
               .put(new Info("caffe::GetSolver<double>").javaNames("GetDoubleSolver"))

               .put(new Info("boost::shared_ptr<caffe::Blob<float> >").annotations("@SharedPtr").pointerTypes("FloatBlob"))
               .put(new Info("boost::shared_ptr<caffe::Blob<double> >").annotations("@SharedPtr").pointerTypes("DoubleBlob"))
               .put(new Info("std::vector<boost::shared_ptr<caffe::Blob<float> > >").pointerTypes("FloatBlobSharedVector").define())
               .put(new Info("std::vector<boost::shared_ptr<caffe::Blob<double> > >").pointerTypes("DoubleBlobSharedVector").define())

               .put(new Info("boost::shared_ptr<caffe::Layer<float> >").annotations("@SharedPtr").pointerTypes("FloatLayer"))
               .put(new Info("boost::shared_ptr<caffe::Layer<double> >").annotations("@SharedPtr").pointerTypes("DoubleLayer"))
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

               .put(new Info("std::vector<bool>").pointerTypes("BoolVector").define())
               .put(new Info("std::vector<std::vector<bool> >").pointerTypes("BoolVectorVector").define())
               .put(new Info("std::map<std::string,int>").pointerTypes("StringIntMap").define());
    }
}
