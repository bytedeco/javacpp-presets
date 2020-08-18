module org.bytedeco.llvm {
  requires transitive org.bytedeco.javacpp;
  exports org.bytedeco.llvm.global;
  exports org.bytedeco.llvm.presets;
  exports org.bytedeco.llvm.program;
  exports org.bytedeco.llvm.clang;
  exports org.bytedeco.llvm.LLVM;
}
