#pragma once

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <onnxruntime_cxx_api.h>

/**
 * Minimal single-input/single-output float32 ONNX Runtime engine.
 *
 * This is a simulation-only CPU fallback for machines where CUDA/TensorRT is
 * unavailable.  The production default remains TensorRT; callers opt in with
 * SONIC_INFERENCE_BACKEND=onnxruntime.
 */
class CpuOnnxFloatEngine {
 public:
  bool Initialize(const std::string& model_path,
                  const std::string& expected_input,
                  const std::string& expected_output) {
    try {
      env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "sonic_cpu");
      Ort::SessionOptions options;
      const unsigned int hardware_threads = std::thread::hardware_concurrency();
      options.SetIntraOpNumThreads(static_cast<int>(std::max(1u, std::min(4u, hardware_threads))));
      options.SetInterOpNumThreads(1);
      options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
      session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), options);

      if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1) {
        std::cerr << "CPU ONNX backend requires exactly one input and one output" << std::endl;
        return false;
      }
      auto input_name = session_->GetInputNameAllocated(0, allocator_);
      auto output_name = session_->GetOutputNameAllocated(0, allocator_);
      input_name_ = input_name.get();
      output_name_ = output_name.get();
      if (input_name_ != expected_input || output_name_ != expected_output) {
        std::cerr << "CPU ONNX tensor mismatch: expected " << expected_input << " -> "
                  << expected_output << ", got " << input_name_ << " -> " << output_name_
                  << std::endl;
        return false;
      }

      // Keep the owning TypeInfo objects alive while querying their nested
      // TensorTypeAndShapeInfo views (ORT 1.16 does not extend that lifetime).
      auto input_type_info = session_->GetInputTypeInfo(0);
      auto output_type_info = session_->GetOutputTypeInfo(0);
      auto input_info = input_type_info.GetTensorTypeAndShapeInfo();
      auto output_info = output_type_info.GetTensorTypeAndShapeInfo();
      if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
          output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        std::cerr << "CPU ONNX backend only supports float32 tensors; input type="
                  << input_info.GetElementType() << ", output type="
                  << output_info.GetElementType() << std::endl;
        return false;
      }
      input_shape_ = input_info.GetShape();
      output_shape_ = output_info.GetShape();
      if (!ResolveStaticShape(input_shape_) || !ResolveStaticShape(output_shape_)) {
        std::cerr << "CPU ONNX backend cannot resolve a dynamic tensor shape" << std::endl;
        return false;
      }
      input_dimension_ = ElementCount(input_shape_);
      output_dimension_ = ElementCount(output_shape_);
      memory_info_ = std::make_unique<Ort::MemoryInfo>(
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
      return input_dimension_ > 0 && output_dimension_ > 0;
    } catch (const Ort::Exception& error) {
      std::cerr << "CPU ONNX initialization failed: " << error.what() << std::endl;
      return false;
    }
  }

  bool Infer(const std::vector<float>& input, std::vector<float>& output) {
    if (!session_ || !memory_info_ || input.size() != input_dimension_) return false;
    try {
      auto input_tensor = Ort::Value::CreateTensor<float>(
          *memory_info_, const_cast<float*>(input.data()), input.size(),
          input_shape_.data(), input_shape_.size());
      const char* input_names[] = {input_name_.c_str()};
      const char* output_names[] = {output_name_.c_str()};
      auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
                                   output_names, 1);
      if (outputs.size() != 1 || !outputs[0].IsTensor()) return false;
      const auto actual_count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
      if (actual_count != output_dimension_) return false;
      const float* values = outputs[0].GetTensorData<float>();
      output.assign(values, values + actual_count);
      return true;
    } catch (const Ort::Exception& error) {
      std::cerr << "CPU ONNX inference failed: " << error.what() << std::endl;
      return false;
    }
  }

  size_t InputDimension() const { return input_dimension_; }
  size_t OutputDimension() const { return output_dimension_; }

 private:
  static bool ResolveStaticShape(std::vector<int64_t>& shape) {
    if (shape.empty()) return false;
    for (auto& dim : shape) {
      if (dim == -1) dim = 1;
      if (dim <= 0) return false;
    }
    return true;
  }

  static size_t ElementCount(const std::vector<int64_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1),
                           [](size_t value, int64_t dim) {
                             return value * static_cast<size_t>(dim);
                           });
  }

  Ort::AllocatorWithDefaultOptions allocator_;
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::MemoryInfo> memory_info_;
  std::string input_name_;
  std::string output_name_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
  size_t input_dimension_ = 0;
  size_t output_dimension_ = 0;
};
