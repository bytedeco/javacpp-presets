JavaCPP Presets for Bullet Physics SDK
======================================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * Bullet Physics SDK 3.21  https://pybullet.org/wordpress

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/bullet/apidocs/


Special Mappings
----------------

Mappings of `btAlignedObjectArray`'s instances for privitime types:

| C++                              | Java            |
|----------------------------------|-----------------|
| `btAlignedObjectArray<bool>`     | `btBoolArray`   |
| `btAlignedObjectArray<char>`     | `btCharArray`   |
| `btAlignedObjectArray<int>`      | `btIntArray`    |
| `btAlignedObjectArray<btScalar>` | `btScalarArray` |

Name of a Java class, corresponding to an instance of `btAlignedObjectArray`
for a composite type, is constructed by adding `Array` suffix to the name of
the composite type, e.g. `btAlignedObjectArray<btQuaternion>` maps to
`btQuaternionArray`.


Sample Usage
------------

See [samples](samples).
