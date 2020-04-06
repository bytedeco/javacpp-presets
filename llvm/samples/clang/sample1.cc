typedef int sampleInt;

class C {
public:
  void f();
};

void C::f() { }

void hoge() {
  sampleInt a = 10;
  C c;
  c.f();
}