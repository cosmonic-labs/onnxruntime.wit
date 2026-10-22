# SqueezeNet example

Classifies a generated image with SqueezeNet through the `onnxruntime:onnxruntime` WIT interface.

## Build

1. Build the example (nightly Rust, `wasm32-wasip3`):

```sh
cargo +nightly build --release
```

2. Build ONNX Runtime for WASI (see [docs/WASI.md](../../../../docs/WASI.md)), then the component that exports the interface. `$ORT_BUILD` is the build's `Release` directory:

```sh
wit-bindgen c ../../wit --world onnxruntime --async=-all --out-dir gen
$WASI_SDK_PATH/bin/clang++ --target=wasm32-wasip3 -mexec-model=reactor -std=c++20 -O2 \
   -fwasm-exceptions -mllvm -wasm-use-legacy-eh=false \
   -I../../../../include/onnxruntime/core/session -Igen \
   ../../onnxruntime_component.cc gen/onnxruntime.c gen/onnxruntime_component_type.o \
   $(find $ORT_BUILD -name '*.a' -not -path '*/CMakeFiles/*' -not -name '*test*') \
   -lunwind -lwasi-emulated-signal -lwasi-emulated-mman -lwasi-emulated-getpid \
   -lwasi-emulated-process-clocks -Wl,-z,stack-size=5242880 -o onnxruntime.wasm
```

3. Plug it into the example:

```sh
wac plug target/wasm32-wasip3/release/squeezenet.wasm --plug onnxruntime.wasm -o squeezenet.wasm
```

## Run

From `onnxruntime/test/testdata`, with wasmtime 49 or later:

```sh
wasmtime run -S p3=y -W component-model-async=y,exceptions=y --dir=.::/ path/to/squeezenet.wasm squeezenet/model.onnx cpu
```

For `webgpu` (the default provider), build ONNX Runtime with `--use_webgpu` and run on a host that implements `wasi:webgpu`.
