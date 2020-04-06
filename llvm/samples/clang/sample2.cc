#include "sample.h"
typedef int sampleInt;

class C {
  void f();
};

void C::f() { }

void hoge() {
  sampleInt a = 10;
  C c;
  c.f();
}