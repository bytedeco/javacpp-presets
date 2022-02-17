module org.bytedeco.bullet {
  requires transitive org.bytedeco.javacpp;
  exports org.bytedeco.bullet.global;
  exports org.bytedeco.bullet.presets;
  exports org.bytedeco.bullet.LinearMath;
  exports org.bytedeco.bullet.BulletCollision;
  exports org.bytedeco.bullet.BulletDynamics;
  exports org.bytedeco.bullet.BulletSoftBody;
}
