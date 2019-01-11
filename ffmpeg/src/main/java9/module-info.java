module org.bytedeco.javacpp.ffmpeg {
  requires transitive org.bytedeco.javacpp;
  exports org.bytedeco.javacpp.avcodec;
  exports org.bytedeco.javacpp.avfilter;
  exports org.bytedeco.javacpp.avutil;
  exports org.bytedeco.javacpp.swscale;
  exports org.bytedeco.javacpp.avdevice;
  exports org.bytedeco.javacpp.avformat;
  exports org.bytedeco.javacpp.postproc;
  exports org.bytedeco.javacpp.swresample;
}
