# OpenVINO

The [OpenVINO](https://github.com/openvinotoolkit/openvino) preset packages the OpenVINO Runtime C API and shared libraries.

Supported platforms:
- `linux-x86_64`
- `windows-x86_64`

The packaged runtime includes the CPU, GPU, and NPU plugin shared libraries from the official OpenVINO wheel distribution.

## Sample

A minimal sample is available under `openvino/samples/`:

```bash
mvn -f openvino/samples/pom.xml compile exec:java
```
