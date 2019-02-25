module org.bytedeco.tensorflow {
  requires transitive org.bytedeco.javacpp;
  requires transitive org.bytedeco.mkldnn;
  exports org.bytedeco.tensorflow.global;
  exports org.bytedeco.tensorflow;

  exports org.tensorflow;
  exports org.tensorflow.contrib.android;
  exports org.tensorflow.examples;
  exports org.tensorflow.lite;
  exports org.tensorflow.op;
  exports org.tensorflow.op.annotation;
  exports org.tensorflow.op.core;
  exports org.tensorflow.processor;
  exports org.tensorflow.types;
}
