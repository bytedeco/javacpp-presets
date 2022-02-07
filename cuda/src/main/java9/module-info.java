module org.bytedeco.cuda {
  requires transitive org.bytedeco.javacpp;
  exports org.bytedeco.cuda.global;
  exports org.bytedeco.cuda.presets;
  exports org.bytedeco.cuda.cublas;
  exports org.bytedeco.cuda.cudart;
  exports org.bytedeco.cuda.cudnn;
  exports org.bytedeco.cuda.cufftw;
  exports org.bytedeco.cuda.curand;
  exports org.bytedeco.cuda.cusolver;
  exports org.bytedeco.cuda.cusparse;
  exports org.bytedeco.cuda.nccl;
  exports org.bytedeco.cuda.nppc;
  exports org.bytedeco.cuda.nppicc;
//  exports org.bytedeco.cuda.nppicom;
  exports org.bytedeco.cuda.nppig;
//  exports org.bytedeco.cuda.nvgraph;
  exports org.bytedeco.cuda.nvml;
  exports org.bytedeco.cuda.nvrtc;
  exports org.bytedeco.cuda.nvToolsExt;
}
