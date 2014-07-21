`flandmark` is a JVM wrapper for [flandmark library](https://github.com/uricamic/flandmark).

##Sources and Native Binaries
It is assumed that:
 
* `opencv` headers and binaries are available in `../opencv/cppbuild`
* `flanddmark` sources can be found in `cppbuild/flandmark/sources` 
* Native binaries were build in `cppbuild/flandmark/build/${architecture}`, for instance, `cppbuild/flandmark/build/x64`.
 The `${architecture}` convention is similar one used by opencv.
* Native binaries were build for `RelWithDebInfo`.