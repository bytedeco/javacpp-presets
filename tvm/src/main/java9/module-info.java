module org.bytedeco.tvm {
  requires transitive org.bytedeco.javacpp;
  requires transitive org.bytedeco.dnnl;
  requires transitive org.bytedeco.llvm;
  requires transitive org.bytedeco.scipy;
  exports org.bytedeco.tvm.global;
  exports org.bytedeco.tvm.presets;
  exports org.bytedeco.tvm;
}
