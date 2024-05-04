/* Add the necessary C++ calls to present a unified C++ API whatever the platform */

namespace javacpp {

   inline c10::Half *allocate_Half(jfloat value) {
#if defined(__aarch64__) && !defined(C10_MOBILE) && !defined(__CUDACC__)
       return new c10::Half((float16_t) value);
#else
       return new c10::Half((float) value);
#endif
   }

   inline float cast_Half_to_float(c10::Half *h) {
#if defined(__aarch64__) && !defined(C10_MOBILE) && !defined(__CUDACC__)
       return (float) (float16_t) *h;
#else
       return (float) *h;
#endif
   }

}