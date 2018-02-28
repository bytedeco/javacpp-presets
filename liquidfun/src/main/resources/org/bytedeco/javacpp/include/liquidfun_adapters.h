#include <Box2D/Common/b2Settings.h>

class b2DynamicTreeQueryCallback {
 public:
  virtual bool QueryCallback(int32 nodeId) = 0;
};
