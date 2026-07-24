module org.bytedeco.pytorch {
  requires transitive org.bytedeco.javacpp;
  requires transitive org.bytedeco.openblas;
  exports org.bytedeco.pytorch.global;
  exports org.bytedeco.pytorch.presets;
  exports org.bytedeco.pytorch.cuda;
  exports org.bytedeco.pytorch.gloo;
  exports org.bytedeco.pytorch.nccl;
  exports org.bytedeco.pytorch.rpc;
  exports org.bytedeco.pytorch.data;
  exports org.bytedeco.pytorch.nn;
  exports org.bytedeco.pytorch.jit;
  exports org.bytedeco.pytorch.optim;
  exports org.bytedeco.pytorch.serialize;
  exports org.bytedeco.pytorch.distributed;
  exports org.bytedeco.pytorch.inductor;
  exports org.bytedeco.pytorch.profiler;
  exports org.bytedeco.pytorch;
}
