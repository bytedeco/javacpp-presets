module org.bytedeco.llvm {
  requires transitive org.bytedeco.javacpp;
  exports org.bytedeco.llvm.global;
  exports org.bytedeco.llvm.clang;
  exports org.bytedeco.llvm.LLVM;
}
