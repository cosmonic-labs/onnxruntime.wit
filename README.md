# ONNX Runtime.wit
Wit interface exposing the ONNX Runtime API


## Compile

1. point to your local wasi-sdk >= v34
```shell
export WASI_SDK_PATH=/path/to/wasi-sdk
```

2. Compile
```shell
cmake -B build -G Ninja
cmake --build build
```
