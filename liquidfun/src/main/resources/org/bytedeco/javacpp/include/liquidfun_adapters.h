#include <Box2D/Common/b2Settings.h>

class b2DynamicTreeQueryCallback {
 public:
  virtual bool QueryCallback(int32 nodeId) = 0;
};

class b2DynamicTreeRayCastCallback {
 public:
  virtual bool RayCastCallback(b2RayCastInput& subInput, int32 nodeId) = 0;
};
