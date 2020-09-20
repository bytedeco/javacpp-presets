module org.bytedeco.ffmpeg {
  requires transitive org.bytedeco.javacpp;
  exports org.bytedeco.ffmpeg;
  exports org.bytedeco.ffmpeg.global;
  exports org.bytedeco.ffmpeg.presets;
  exports org.bytedeco.ffmpeg.avcodec;
  exports org.bytedeco.ffmpeg.avdevice;
  exports org.bytedeco.ffmpeg.avfilter;
  exports org.bytedeco.ffmpeg.avformat;
  exports org.bytedeco.ffmpeg.avutil;
  exports org.bytedeco.ffmpeg.postproc;
  exports org.bytedeco.ffmpeg.swresample;
  exports org.bytedeco.ffmpeg.swscale;
}
