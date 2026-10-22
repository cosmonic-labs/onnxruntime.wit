// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Implements the onnxruntime:onnxruntime WIT world (wit/onnxruntime.wit) on top of the ONNX Runtime C++ API.
//
// The bindings are generated with `wit-bindgen c --async=-all`: the WIT declares the long-running functions async, but
// they are lifted synchronously here and block until done, like the rest of ONNX Runtime on WASI.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnxruntime_cxx_api.h"

#include "onnxruntime.h"

// The C bindings name the resource representations; we define them.
struct exports_onnxruntime_onnxruntime_types_tensor_t {
  Ort::Value value;
};

struct exports_onnxruntime_onnxruntime_inference_session_session_t {
  Ort::Session session;
};

struct exports_onnxruntime_onnxruntime_inference_session_io_binding_t {
  Ort::IoBinding binding;
};

namespace {

using Tensor = exports_onnxruntime_onnxruntime_types_tensor_t;
using Session = exports_onnxruntime_onnxruntime_inference_session_session_t;
using IoBinding = exports_onnxruntime_onnxruntime_inference_session_io_binding_t;
using Error = exports_onnxruntime_onnxruntime_types_error_t;
using ElementType = exports_onnxruntime_onnxruntime_types_tensor_element_type_t;
using SessionOptions = exports_onnxruntime_onnxruntime_inference_session_session_options_t;
using RunOptions = exports_onnxruntime_onnxruntime_inference_session_run_options_t;

Ort::Env& GetEnv() {
  static Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "onnxruntime-component"};
  return env;
}

std::string ToString(const onnxruntime_string_t& s) {
  return std::string(reinterpret_cast<const char*>(s.ptr), s.len);
}

onnxruntime_string_t ToWitString(const std::string& s) {
  onnxruntime_string_t ret;
  onnxruntime_string_dup_n(&ret, s.data(), s.size());
  return ret;
}

// Constructors can't return errors in WIT, so invalid arguments trap.
[[noreturn]] void Trap(const std::string& message) {
  std::fprintf(stderr, "onnxruntime: %s\n", message.c_str());
  std::abort();
}

void SetError(Error* err, OrtErrorCode code, const std::string& message) {
  switch (code) {
    case ORT_INVALID_ARGUMENT:
      err->code = EXPORTS_ONNXRUNTIME_ONNXRUNTIME_TYPES_ERROR_CODE_INVALID_ARGUMENT;
      break;
    case ORT_NO_SUCHFILE:
      err->code = EXPORTS_ONNXRUNTIME_ONNXRUNTIME_TYPES_ERROR_CODE_NO_SUCH_FILE;
      break;
    case ORT_NO_MODEL:
      err->code = EXPORTS_ONNXRUNTIME_ONNXRUNTIME_TYPES_ERROR_CODE_NO_MODEL;
      break;
    case ORT_ENGINE_ERROR:
      err->code = EXPORTS_ONNXRUNTIME_ONNXRUNTIME_TYPES_ERROR_CODE_ENGINE_ERROR;
      break;
    case ORT_RUNTIME_EXCEPTION:
      err->code = EXPORTS_ONNXRUNTIME_ONNXRUNTIME_TYPES_ERROR_CODE_RUNTIME_EXCEPTION;
      break;
    case ORT_INVALID_PROTOBUF:
      err->code = EXPORTS_ONNXRUNTIME_ONNXRUNTIME_TYPES_ERROR_CODE_INVALID_PROTOBUF;
      break;
    case ORT_MODEL_LOADED:
      err->code = EXPORTS_ONNXRUNTIME_ONNXRUNTIME_TYPES_ERROR_CODE_MODEL_LOADED;
      break;
    case ORT_NOT_IMPLEMENTED:
      err->code = EXPORTS_ONNXRUNTIME_ONNXRUNTIME_TYPES_ERROR_CODE_NOT_IMPLEMENTED;
      break;
    case ORT_INVALID_GRAPH:
      err->code = EXPORTS_ONNXRUNTIME_ONNXRUNTIME_TYPES_ERROR_CODE_INVALID_GRAPH;
      break;
    case ORT_EP_FAIL:
      err->code = EXPORTS_ONNXRUNTIME_ONNXRUNTIME_TYPES_ERROR_CODE_EP_FAIL;
      break;
    default:
      err->code = EXPORTS_ONNXRUNTIME_ONNXRUNTIME_TYPES_ERROR_CODE_FAIL;
      break;
  }
  err->message = ToWitString(message);
}

// Runs `fn`, converting exceptions into a WIT error. Returns false on error.
template <typename Fn>
bool Try(Error* err, Fn&& fn) {
  try {
    fn();
    return true;
  } catch (const Ort::Exception& e) {
    SetError(err, e.GetOrtErrorCode(), e.what());
  } catch (const std::exception& e) {
    SetError(err, ORT_FAIL, e.what());
  }
  return false;
}

struct ElementTypeInfo {
  ElementType wit;
  ONNXTensorElementDataType ort;
  size_t size;  // 0 for strings
};

constexpr ElementTypeInfo kElementTypes[] = {
    {EXPORTS_ONNXRUNTIME_ONNXRUNTIME_TYPES_TENSOR_ELEMENT_TYPE_FLOAT32, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 4},
    {EXPORTS_ONNXRUNTIME_ONNXRUNTIME_TYPES_TENSOR_ELEMENT_TYPE_FLOAT16, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, 2},
    {EXPORTS_ONNXRUNTIME_ONNXRUNTIME_TYPES_TENSOR_ELEMENT_TYPE_BFLOAT16, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16, 2},
    {EXPORTS_ONNXRUNTIME_ONNXRUNTIME_TYPES_TENSOR_ELEMENT_TYPE_FLOAT64, ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, 8},
    {EXPORTS_ONNXRUNTIME_ONNXRUNTIME_TYPES_TENSOR_ELEMENT_TYPE_INT8, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, 1},
    {EXPORTS_ONNXRUNTIME_ONNXRUNTIME_TYPES_TENSOR_ELEMENT_TYPE_INT16, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, 2},
    {EXPORTS_ONNXRUNTIME_ONNXRUNTIME_TYPES_TENSOR_ELEMENT_TYPE_INT32, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, 4},
    {EXPORTS_ONNXRUNTIME_ONNXRUNTIME_TYPES_TENSOR_ELEMENT_TYPE_INT64, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 8},
    {EXPORTS_ONNXRUNTIME_ONNXRUNTIME_TYPES_TENSOR_ELEMENT_TYPE_UINT8, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, 1},
    {EXPORTS_ONNXRUNTIME_ONNXRUNTIME_TYPES_TENSOR_ELEMENT_TYPE_UINT16, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, 2},
    {EXPORTS_ONNXRUNTIME_ONNXRUNTIME_TYPES_TENSOR_ELEMENT_TYPE_UINT32, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, 4},
    {EXPORTS_ONNXRUNTIME_ONNXRUNTIME_TYPES_TENSOR_ELEMENT_TYPE_UINT64, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, 8},
    {EXPORTS_ONNXRUNTIME_ONNXRUNTIME_TYPES_TENSOR_ELEMENT_TYPE_BOOL, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, 1},
    {EXPORTS_ONNXRUNTIME_ONNXRUNTIME_TYPES_TENSOR_ELEMENT_TYPE_STRING, ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, 0},
};

const ElementTypeInfo* FindElementType(ElementType wit) {
  for (const auto& info : kElementTypes) {
    if (info.wit == wit) return &info;
  }
  return nullptr;
}

const ElementTypeInfo* FindElementType(ONNXTensorElementDataType ort) {
  for (const auto& info : kElementTypes) {
    if (info.ort == ort) return &info;
  }
  return nullptr;
}

exports_onnxruntime_onnxruntime_types_own_tensor_t NewTensor(Ort::Value value) {
  return exports_onnxruntime_onnxruntime_types_tensor_new(new Tensor{std::move(value)});
}

// ORT registers execution providers under these names. Accept the lowercase names used by the JavaScript API.
std::string OrtProviderName(const std::string& name) {
  static const std::unordered_map<std::string, std::string> kNames{
      {"webgpu", "WebGPU"},
      {"xnnpack", "XNNPACK"},
  };
  auto it = kNames.find(name);
  return it == kNames.end() ? name : it->second;
}

OrtLoggingLevel ToOrtLoggingLevel(exports_onnxruntime_onnxruntime_inference_session_log_severity_t severity) {
  // log-severity matches OrtLoggingLevel's order.
  return static_cast<OrtLoggingLevel>(severity);
}

Ort::SessionOptions ToOrtSessionOptions(const SessionOptions* options) {
  Ort::SessionOptions so;
  if (!options) return so;

  for (size_t i = 0; i < options->execution_providers.len; ++i) {
    const auto& ep = options->execution_providers.ptr[i];
    const std::string name = ToString(ep.name);
    if (name == "cpu") continue;  // always available as the fallback
    std::unordered_map<std::string, std::string> ep_options;
    for (size_t j = 0; j < ep.options.len; ++j) {
      ep_options[ToString(ep.options.ptr[j].f0)] = ToString(ep.options.ptr[j].f1);
    }
    so.AppendExecutionProvider(OrtProviderName(name), ep_options);
  }

  if (options->graph_optimization_level.is_some) {
    static constexpr GraphOptimizationLevel kLevels[] = {ORT_DISABLE_ALL, ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED,
                                                         ORT_ENABLE_LAYOUT, ORT_ENABLE_ALL};
    so.SetGraphOptimizationLevel(kLevels[options->graph_optimization_level.val]);
  }
  if (options->execution_mode.is_some) {
    so.SetExecutionMode(options->execution_mode.val == EXPORTS_ONNXRUNTIME_ONNXRUNTIME_INFERENCE_SESSION_EXECUTION_MODE_PARALLEL
                            ? ORT_PARALLEL
                            : ORT_SEQUENTIAL);
  }
  if (options->intra_op_num_threads.is_some) {
    so.SetIntraOpNumThreads(static_cast<int>(options->intra_op_num_threads.val));
  }
  for (size_t i = 0; i < options->free_dimension_overrides.len; ++i) {
    const auto& dim = options->free_dimension_overrides.ptr[i];
    so.AddFreeDimensionOverrideByName(ToString(dim.f0).c_str(), dim.f1);
  }
  if (options->log_id.is_some) {
    so.SetLogId(ToString(options->log_id.val).c_str());
  }
  if (options->log_severity_level.is_some) {
    so.SetLogSeverityLevel(ToOrtLoggingLevel(options->log_severity_level.val));
  }
  for (size_t i = 0; i < options->config_entries.len; ++i) {
    const auto& entry = options->config_entries.ptr[i];
    so.AddConfigEntry(ToString(entry.f0).c_str(), ToString(entry.f1).c_str());
  }
  return so;
}

Ort::RunOptions ToOrtRunOptions(const RunOptions* options) {
  Ort::RunOptions ro;
  if (!options) return ro;
  if (options->tag.is_some) {
    ro.SetRunTag(ToString(options->tag.val).c_str());
  }
  if (options->log_severity_level.is_some) {
    ro.SetRunLogSeverityLevel(ToOrtLoggingLevel(options->log_severity_level.val));
  }
  for (size_t i = 0; i < options->config_entries.len; ++i) {
    const auto& entry = options->config_entries.ptr[i];
    ro.AddConfigEntry(ToString(entry.f0).c_str(), ToString(entry.f1).c_str());
  }
  return ro;
}

exports_onnxruntime_onnxruntime_inference_session_own_session_t NewSession(Ort::Session session) {
  return exports_onnxruntime_onnxruntime_inference_session_session_new(new Session{std::move(session)});
}

// Reads a stream<u8> to the end, blocking until the writer is done.
std::vector<uint8_t> ReadAll(exports_onnxruntime_onnxruntime_inference_session_stream_u8_t reader) {
  std::vector<uint8_t> bytes;
  constexpr size_t kChunk = 64 * 1024;
  onnxruntime_waitable_set_t set = 0;
  for (;;) {
    const size_t offset = bytes.size();
    bytes.resize(offset + kChunk);
    auto status = exports_onnxruntime_onnxruntime_inference_session_stream_u8_read(reader, bytes.data() + offset, kChunk);
    if (status == ONNXRUNTIME_WAITABLE_STATUS_BLOCKED) {
      if (set == 0) {
        set = onnxruntime_waitable_set_new();
        onnxruntime_waitable_join(reader, set);
      }
      onnxruntime_event_t event;
      onnxruntime_waitable_set_wait(set, &event);
      status = event.code;
    }
    bytes.resize(offset + ONNXRUNTIME_WAITABLE_COUNT(status));
    if (ONNXRUNTIME_WAITABLE_STATE(status) != ONNXRUNTIME_WAITABLE_COMPLETED) break;  // the writer is done
  }
  if (set != 0) {
    onnxruntime_waitable_join(reader, 0);
    onnxruntime_waitable_set_drop(set);
  }
  exports_onnxruntime_onnxruntime_inference_session_stream_u8_drop_readable(reader);
  return bytes;
}

exports_onnxruntime_onnxruntime_types_value_metadata_t ToValueMetadata(const std::string& name,
                                                                        const Ort::TypeInfo& type_info) {
  exports_onnxruntime_onnxruntime_types_value_metadata_t metadata{};
  metadata.name = ToWitString(name);
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) return metadata;

  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  if (const auto* type = FindElementType(tensor_info.GetElementType())) {
    metadata.element_type = {true, type->wit};
  }
  const auto shape = tensor_info.GetShape();
  std::vector<const char*> symbolic(shape.size());
  tensor_info.GetSymbolicDimensions(symbolic.data(), symbolic.size());

  auto* dims = static_cast<exports_onnxruntime_onnxruntime_types_dimension_t*>(
      std::malloc(shape.size() * sizeof(exports_onnxruntime_onnxruntime_types_dimension_t)));
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] >= 0) {
      dims[i].tag = EXPORTS_ONNXRUNTIME_ONNXRUNTIME_TYPES_DIMENSION_FIXED;
      dims[i].val.fixed = shape[i];
    } else {
      dims[i].tag = EXPORTS_ONNXRUNTIME_ONNXRUNTIME_TYPES_DIMENSION_SYMBOLIC;
      dims[i].val.symbolic = ToWitString(symbolic[i] ? symbolic[i] : "");
    }
  }
  metadata.shape = {true, {dims, shape.size()}};
  return metadata;
}

template <typename List>
void ToMetadataList(const Ort::Session& session, bool inputs, List* ret) {
  Ort::AllocatorWithDefaultOptions allocator;
  const size_t count = inputs ? session.GetInputCount() : session.GetOutputCount();
  ret->len = count;
  ret->ptr = static_cast<decltype(ret->ptr)>(std::malloc(count * sizeof(*ret->ptr)));
  for (size_t i = 0; i < count; ++i) {
    auto name = inputs ? session.GetInputNameAllocated(i, allocator) : session.GetOutputNameAllocated(i, allocator);
    ret->ptr[i] = ToValueMetadata(name.get(), inputs ? session.GetInputTypeInfo(i) : session.GetOutputTypeInfo(i));
  }
}

template <typename List>
void ToOutputList(const std::vector<std::string>& names, std::vector<Ort::Value> values, List* ret) {
  ret->len = names.size();
  ret->ptr = static_cast<decltype(ret->ptr)>(std::malloc(names.size() * sizeof(*ret->ptr)));
  for (size_t i = 0; i < names.size(); ++i) {
    ret->ptr[i].f0 = ToWitString(names[i]);
    ret->ptr[i].f1 = NewTensor(std::move(values[i]));
  }
}

}  // namespace

extern "C" {

//
// onnxruntime:onnxruntime/types
//

exports_onnxruntime_onnxruntime_types_own_tensor_t exports_onnxruntime_onnxruntime_types_constructor_tensor(
    ElementType element_type, onnxruntime_list_s64_t* dims, onnxruntime_list_u8_t* data) {
  const auto* type = FindElementType(element_type);
  if (!type || type->size == 0) Trap("use tensor.from-strings to create string tensors");

  const std::vector<int64_t> shape(dims->ptr, dims->ptr + dims->len);
  onnxruntime_list_s64_free(dims);
  size_t count = 1;
  for (int64_t dim : shape) {
    if (dim < 0) Trap("tensor dimensions must not be negative");
    count *= static_cast<size_t>(dim);
  }
  if (data->len != count * type->size) {
    Trap("tensor data has " + std::to_string(data->len) + " bytes, but its shape needs " +
         std::to_string(count * type->size));
  }

  Ort::AllocatorWithDefaultOptions allocator;
  auto value = Ort::Value::CreateTensor(allocator, shape.data(), shape.size(), type->ort);
  std::memcpy(value.GetTensorMutableRawData(), data->ptr, data->len);
  onnxruntime_list_u8_free(data);
  return NewTensor(std::move(value));
}

exports_onnxruntime_onnxruntime_types_own_tensor_t exports_onnxruntime_onnxruntime_types_static_tensor_from_strings(
    onnxruntime_list_s64_t* dims, onnxruntime_list_string_t* data) {
  const std::vector<int64_t> shape(dims->ptr, dims->ptr + dims->len);
  onnxruntime_list_s64_free(dims);

  std::vector<std::string> strings;
  strings.reserve(data->len);
  for (size_t i = 0; i < data->len; ++i) strings.push_back(ToString(data->ptr[i]));
  onnxruntime_list_string_free(data);
  std::vector<const char*> c_strings;
  c_strings.reserve(strings.size());
  for (const auto& s : strings) c_strings.push_back(s.c_str());

  Ort::AllocatorWithDefaultOptions allocator;
  auto value = Ort::Value::CreateTensor(allocator, shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
  if (value.GetTensorTypeAndShapeInfo().GetElementCount() != c_strings.size()) {
    Trap("string tensor has " + std::to_string(c_strings.size()) + " elements, but its shape needs " +
         std::to_string(value.GetTensorTypeAndShapeInfo().GetElementCount()));
  }
  value.FillStringTensor(c_strings.data(), c_strings.size());
  return NewTensor(std::move(value));
}

ElementType exports_onnxruntime_onnxruntime_types_method_tensor_element_type(
    exports_onnxruntime_onnxruntime_types_borrow_tensor_t self) {
  const auto* type = FindElementType(self->value.GetTensorTypeAndShapeInfo().GetElementType());
  if (!type) Trap("tensor has an element type that the onnxruntime interface does not support");
  return type->wit;
}

void exports_onnxruntime_onnxruntime_types_method_tensor_dims(exports_onnxruntime_onnxruntime_types_borrow_tensor_t self,
                                                              onnxruntime_list_s64_t* ret) {
  const auto shape = self->value.GetTensorTypeAndShapeInfo().GetShape();
  ret->len = shape.size();
  ret->ptr = static_cast<int64_t*>(std::malloc(shape.size() * sizeof(int64_t)));
  std::memcpy(ret->ptr, shape.data(), shape.size() * sizeof(int64_t));
}

exports_onnxruntime_onnxruntime_types_device_type_t exports_onnxruntime_onnxruntime_types_method_tensor_location(
    exports_onnxruntime_onnxruntime_types_borrow_tensor_t self) {
  return self->value.GetTensorMemoryInfo().GetDeviceType() == OrtMemoryInfoDeviceType_CPU
             ? EXPORTS_ONNXRUNTIME_ONNXRUNTIME_TYPES_DEVICE_TYPE_CPU
             : EXPORTS_ONNXRUNTIME_ONNXRUNTIME_TYPES_DEVICE_TYPE_GPU;
}

bool exports_onnxruntime_onnxruntime_types_method_tensor_data(exports_onnxruntime_onnxruntime_types_borrow_tensor_t self,
                                                              onnxruntime_list_u8_t* ret, Error* err) {
  return Try(err, [&]() {
    auto info = self->value.GetTensorTypeAndShapeInfo();
    const auto* type = FindElementType(info.GetElementType());
    if (!type || type->size == 0) {
      throw Ort::Exception("tensor.data does not support this element type, use tensor.strings for strings",
                           ORT_INVALID_ARGUMENT);
    }
    const size_t size = info.GetElementCount() * type->size;

    const Ort::Value* cpu_value = &self->value;
    Ort::Value copy{nullptr};
    if (self->value.GetTensorMemoryInfo().GetDeviceType() != OrtMemoryInfoDeviceType_CPU) {
      Ort::AllocatorWithDefaultOptions allocator;
      const auto shape = info.GetShape();
      copy = Ort::Value::CreateTensor(allocator, shape.data(), shape.size(), type->ort);
      const OrtValue* src = self->value;
      OrtValue* dst = copy;
      Ort::ThrowOnError(Ort::GetApi().CopyTensors(GetEnv(), &src, &dst, nullptr, 1));
      cpu_value = &copy;
    }

    ret->len = size;
    ret->ptr = static_cast<uint8_t*>(std::malloc(size));
    std::memcpy(ret->ptr, cpu_value->GetTensorRawData(), size);
  });
}

bool exports_onnxruntime_onnxruntime_types_method_tensor_strings(
    exports_onnxruntime_onnxruntime_types_borrow_tensor_t self, onnxruntime_list_string_t* ret, Error* err) {
  return Try(err, [&]() {
    const size_t count = self->value.GetTensorTypeAndShapeInfo().GetElementCount();
    ret->len = count;
    ret->ptr = static_cast<onnxruntime_string_t*>(std::malloc(count * sizeof(onnxruntime_string_t)));
    for (size_t i = 0; i < count; ++i) {
      ret->ptr[i] = ToWitString(self->value.GetStringTensorElement(i));
    }
  });
}

void exports_onnxruntime_onnxruntime_types_tensor_destructor(Tensor* rep) { delete rep; }

//
// onnxruntime:onnxruntime/inference-session
//

bool exports_onnxruntime_onnxruntime_inference_session_static_session_create_from_path(
    onnxruntime_string_t* path, SessionOptions* maybe_options,
    exports_onnxruntime_onnxruntime_inference_session_own_session_t* ret, Error* err) {
  const std::string model_path = ToString(*path);
  onnxruntime_string_free(path);
  const bool ok = Try(err, [&]() {
    *ret = NewSession(Ort::Session(GetEnv(), model_path.c_str(), ToOrtSessionOptions(maybe_options)));
  });
  if (maybe_options) exports_onnxruntime_onnxruntime_inference_session_session_options_free(maybe_options);
  return ok;
}

bool exports_onnxruntime_onnxruntime_inference_session_static_session_create_from_stream(
    exports_onnxruntime_onnxruntime_inference_session_stream_u8_t model, SessionOptions* maybe_options,
    exports_onnxruntime_onnxruntime_inference_session_own_session_t* ret, Error* err) {
  const std::vector<uint8_t> bytes = ReadAll(model);
  const bool ok = Try(err, [&]() {
    *ret = NewSession(Ort::Session(GetEnv(), bytes.data(), bytes.size(), ToOrtSessionOptions(maybe_options)));
  });
  if (maybe_options) exports_onnxruntime_onnxruntime_inference_session_session_options_free(maybe_options);
  return ok;
}

void exports_onnxruntime_onnxruntime_inference_session_method_session_input_metadata(
    exports_onnxruntime_onnxruntime_inference_session_borrow_session_t self,
    exports_onnxruntime_onnxruntime_inference_session_list_value_metadata_t* ret) {
  ToMetadataList(self->session, /*inputs*/ true, ret);
}

void exports_onnxruntime_onnxruntime_inference_session_method_session_output_metadata(
    exports_onnxruntime_onnxruntime_inference_session_borrow_session_t self,
    exports_onnxruntime_onnxruntime_inference_session_list_value_metadata_t* ret) {
  ToMetadataList(self->session, /*inputs*/ false, ret);
}

bool exports_onnxruntime_onnxruntime_inference_session_method_session_run(
    exports_onnxruntime_onnxruntime_inference_session_borrow_session_t self,
    exports_onnxruntime_onnxruntime_inference_session_list_tuple2_string_borrow_tensor_t* feeds,
    onnxruntime_list_string_t* maybe_fetches, RunOptions* maybe_options,
    exports_onnxruntime_onnxruntime_inference_session_list_tuple2_string_own_tensor_t* ret, Error* err) {
  std::vector<std::string> input_names;
  std::vector<const OrtValue*> input_values;
  for (size_t i = 0; i < feeds->len; ++i) {
    input_names.push_back(ToString(feeds->ptr[i].f0));
    input_values.push_back(feeds->ptr[i].f1->value);
  }
  exports_onnxruntime_onnxruntime_inference_session_list_tuple2_string_borrow_tensor_free(feeds);

  std::vector<std::string> output_names;
  if (maybe_fetches) {
    for (size_t i = 0; i < maybe_fetches->len; ++i) output_names.push_back(ToString(maybe_fetches->ptr[i]));
    onnxruntime_list_string_free(maybe_fetches);
  } else {
    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < self->session.GetOutputCount(); ++i) {
      output_names.emplace_back(self->session.GetOutputNameAllocated(i, allocator).get());
    }
  }

  const bool ok = Try(err, [&]() {
    std::vector<const char*> input_name_ptrs, output_name_ptrs;
    for (const auto& name : input_names) input_name_ptrs.push_back(name.c_str());
    for (const auto& name : output_names) output_name_ptrs.push_back(name.c_str());
    std::vector<OrtValue*> outputs(output_names.size(), nullptr);

    Ort::RunOptions run_options = ToOrtRunOptions(maybe_options);
    Ort::ThrowOnError(Ort::GetApi().Run(self->session, run_options, input_name_ptrs.data(), input_values.data(),
                                        input_values.size(), output_name_ptrs.data(), output_name_ptrs.size(),
                                        outputs.data()));

    std::vector<Ort::Value> values;
    for (OrtValue* output : outputs) values.emplace_back(output);
    ToOutputList(output_names, std::move(values), ret);
  });
  if (maybe_options) exports_onnxruntime_onnxruntime_inference_session_run_options_free(maybe_options);
  return ok;
}

bool exports_onnxruntime_onnxruntime_inference_session_method_session_run_with_binding(
    exports_onnxruntime_onnxruntime_inference_session_borrow_session_t self,
    exports_onnxruntime_onnxruntime_inference_session_borrow_io_binding_t binding, RunOptions* maybe_options,
    Error* err) {
  const bool ok = Try(err, [&]() { self->session.Run(ToOrtRunOptions(maybe_options), binding->binding); });
  if (maybe_options) exports_onnxruntime_onnxruntime_inference_session_run_options_free(maybe_options);
  return ok;
}

void exports_onnxruntime_onnxruntime_inference_session_session_destructor(Session* rep) { delete rep; }

exports_onnxruntime_onnxruntime_inference_session_own_io_binding_t
exports_onnxruntime_onnxruntime_inference_session_constructor_io_binding(
    exports_onnxruntime_onnxruntime_inference_session_borrow_session_t session) {
  return exports_onnxruntime_onnxruntime_inference_session_io_binding_new(new IoBinding{Ort::IoBinding(session->session)});
}

bool exports_onnxruntime_onnxruntime_inference_session_method_io_binding_bind_input(
    exports_onnxruntime_onnxruntime_inference_session_borrow_io_binding_t self, onnxruntime_string_t* name,
    exports_onnxruntime_onnxruntime_inference_session_borrow_tensor_t value, Error* err) {
  const std::string input_name = ToString(*name);
  onnxruntime_string_free(name);
  return Try(err, [&]() { self->binding.BindInput(input_name.c_str(), value->value); });
}

bool exports_onnxruntime_onnxruntime_inference_session_method_io_binding_bind_output(
    exports_onnxruntime_onnxruntime_inference_session_borrow_io_binding_t self, onnxruntime_string_t* name,
    exports_onnxruntime_onnxruntime_inference_session_borrow_tensor_t value, Error* err) {
  const std::string output_name = ToString(*name);
  onnxruntime_string_free(name);
  return Try(err, [&]() { self->binding.BindOutput(output_name.c_str(), value->value); });
}

bool exports_onnxruntime_onnxruntime_inference_session_method_io_binding_bind_output_to_device(
    exports_onnxruntime_onnxruntime_inference_session_borrow_io_binding_t self, onnxruntime_string_t* name,
    exports_onnxruntime_onnxruntime_inference_session_device_type_t device, Error* err) {
  const std::string output_name = ToString(*name);
  onnxruntime_string_free(name);
  return Try(err, [&]() {
    if (device == EXPORTS_ONNXRUNTIME_ONNXRUNTIME_TYPES_DEVICE_TYPE_CPU) {
      self->binding.BindOutput(output_name.c_str(),
                               Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault));
    } else {
      // The WebGPU EP's device memory, onnxruntime::WEBGPU_BUFFER.
      self->binding.BindOutput(output_name.c_str(),
                               Ort::MemoryInfo("WebGPU_Buf", OrtDeviceAllocator, 0, OrtMemTypeDefault));
    }
  });
}

bool exports_onnxruntime_onnxruntime_inference_session_method_io_binding_bound_outputs(
    exports_onnxruntime_onnxruntime_inference_session_borrow_io_binding_t self,
    exports_onnxruntime_onnxruntime_inference_session_list_tuple2_string_own_tensor_t* ret, Error* err) {
  return Try(err, [&]() { ToOutputList(self->binding.GetOutputNames(), self->binding.GetOutputValues(), ret); });
}

void exports_onnxruntime_onnxruntime_inference_session_method_io_binding_clear_bound_inputs(
    exports_onnxruntime_onnxruntime_inference_session_borrow_io_binding_t self) {
  self->binding.ClearBoundInputs();
}

void exports_onnxruntime_onnxruntime_inference_session_method_io_binding_clear_bound_outputs(
    exports_onnxruntime_onnxruntime_inference_session_borrow_io_binding_t self) {
  self->binding.ClearBoundOutputs();
}

void exports_onnxruntime_onnxruntime_inference_session_io_binding_destructor(IoBinding* rep) { delete rep; }

}  // extern "C"
