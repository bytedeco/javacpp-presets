module org.bytedeco.tesseract {
  requires transitive org.bytedeco.javacpp;
  requires transitive org.bytedeco.leptonica;
  exports org.bytedeco.tesseract.global;
  exports org.bytedeco.tesseract;
}
