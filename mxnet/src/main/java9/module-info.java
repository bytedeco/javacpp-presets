module org.bytedeco.mxnet {
  requires transitive org.bytedeco.javacpp;
  requires transitive org.bytedeco.mkldnn;
  requires transitive org.bytedeco.openblas;
  requires transitive org.bytedeco.opencv;
  exports org.bytedeco.mxnet.global;
  exports org.bytedeco.mxnet.presets;
  exports org.bytedeco.mxnet;

  exports org.apache.mxnet;
  exports org.apache.mxnet.contrib;
  exports org.apache.mxnet.module;
  exports org.apache.mxnet.annotation;
  exports org.apache.mxnet.util;
  exports org.apache.mxnet.utils;
  exports org.apache.mxnet.infer;
  exports org.apache.mxnet.infer.javaapi;
  exports org.apache.mxnet.init;
  exports org.apache.mxnet.io;
  exports org.apache.mxnet.javaapi;
  exports org.apache.mxnet.spark;
  exports org.apache.mxnet.spark.utils;
  exports org.apache.mxnet.spark.example;
  exports org.apache.mxnet.spark.transformer;
  exports org.apache.mxnet.spark.io;
  exports org.apache.mxnet.optimizer;
}
