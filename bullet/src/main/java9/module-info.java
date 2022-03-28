module org.bytedeco.bullet {
  requires transitive org.bytedeco.javacpp;
  exports org.bytedeco.bullet.global;
  exports org.bytedeco.bullet.presets;
  exports org.bytedeco.bullet.LinearMath;
  exports org.bytedeco.bullet.BulletCollision;
  exports org.bytedeco.bullet.BulletDynamics;
  exports org.bytedeco.bullet.BulletSoftBody;
  exports org.bytedeco.bullet.Bullet3Common;
  exports org.bytedeco.bullet.Bullet3Collision;
  exports org.bytedeco.bullet.Bullet3Dynamics;
  exports org.bytedeco.bullet.Bullet3OpenCL;
}
