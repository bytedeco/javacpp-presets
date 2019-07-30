module org.bytedeco.tensorflow {
  requires transitive org.bytedeco.javacpp;
  requires transitive org.bytedeco.mkldnn;
  exports org.bytedeco.tensorflow.global;
  exports org.bytedeco.tensorflow.presets;
  exports org.bytedeco.tensorflow;

  exports org.tensorflow;
  exports org.tensorflow.contrib.android;
  exports org.tensorflow.examples;
  exports org.tensorflow.lite;
  exports org.tensorflow.op;
  exports org.tensorflow.op.annotation;
  exports org.tensorflow.op.core;
  exports org.tensorflow.op.audio;
  exports org.tensorflow.op.bitwise;
  exports org.tensorflow.op.collective;
  exports org.tensorflow.op.data;
  exports org.tensorflow.op.dtypes;
  exports org.tensorflow.op.image;
  exports org.tensorflow.op.io;
  exports org.tensorflow.op.linalg;
  exports org.tensorflow.op.math;
  exports org.tensorflow.op.nn;
  exports org.tensorflow.op.quantization;
  exports org.tensorflow.op.random;
  exports org.tensorflow.op.signal;
  exports org.tensorflow.op.sparse;
  exports org.tensorflow.op.strings;
  exports org.tensorflow.op.summary;
  exports org.tensorflow.op.train;
  exports org.tensorflow.processor;
  exports org.tensorflow.types;

}
