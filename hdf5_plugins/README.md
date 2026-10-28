JavaCPP Presets for HDF5 Plugins
==============================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/hdf5_plugins/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/hdf5_plugins) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/hdf5_plugins.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![hdf5_plugins](https://github.com/bytedeco/javacpp-presets/workflows/hdf5_plugins/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Ahdf5_plugins)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * HDF5 Plugins 2.2.0  https://github.com/HDFGroup/hdf5_plugins

It bundles the HDF5 filter plugins for Zstandard, LZ4, JPEG, bzip2, Bitshuffle, Blosc, and Blosc2, with their compression libraries linked in statically. There is no C API to map: HDF5 loads the plugins at runtime. Call `HDF5Plugins.load()` once before using HDF5 to add the bundled plugins to its search path (`H5PLappend()`), then use the filter IDs, for example `HDF5Plugins.H5Z_FILTER_ZSTD`, with `H5Pset_filter()`. Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.

Note that the plugins are built from the sources bundled with the `hdf5_plugins` release, and that JPEG is not available on Windows, as per upstream.


Sample Usage
------------
After creating the `pom.xml` and `ZstdSample.java` source files from the `samples/` directory, simply execute on the command line:
```bash
 $ mvn compile exec:java
```
