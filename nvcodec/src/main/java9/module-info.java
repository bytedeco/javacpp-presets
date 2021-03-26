module org.bytedeco.nvcodec {
  requires transitive org.bytedeco.javacpp;
  requires transitive org.bytedeco.cuda;
  exports org.bytedeco.nvcodec.global;
  exports org.bytedeco.nvcodec.presets;
  exports org.bytedeco.nvcodec.nvcuvid;
  exports org.bytedeco.nvcodec.nvencodeapi;
}