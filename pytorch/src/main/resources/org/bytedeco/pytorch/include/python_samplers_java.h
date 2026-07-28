/*
 * Python torch.utils.data.Sampler classes missing from C++ LibTorch, implemented
 * as real subclasses of torch::data::samplers::Sampler<> so JavaCPP peers
 * `extends org.bytedeco.pytorch.data.sampler.Sampler`.
 *
 * API is the C++ Sampler contract: reset / next(batch_size) / save / load
 * (not Python's __iter__ of single indices — DataLoader already batches via next).
 *
 * - SubsetRandomSampler: shuffle a fixed index list each epoch
 * - WeightedRandomSampler: multinomial over weights
 * - BatchSampler: wrap another Sampler with drop_last semantics
 */
#pragma once

#include <torch/data/samplers/base.h>
#include <torch/serialize/archive.h>
#include <torch/types.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

namespace torch::data::samplers {

// ============================================================================
// SubsetRandomSampler
// ============================================================================

/// Samples randomly from a fixed list of indices, without replacement within
/// an epoch (reshuffles on reset). Mirrors Python
/// {@code torch.utils.data.SubsetRandomSampler}, adapted to C++ Sampler<> which
/// yields batches via {@code next(batch_size)}.
class SubsetRandomSampler : public Sampler<> {
 public:
  explicit SubsetRandomSampler(std::vector<size_t> indices)
      : indices_(std::move(indices)) {
    reset(std::nullopt);
  }

  /// Convenience for Java long[] / int64 buffers.
  static SubsetRandomSampler from_int64(const std::vector<int64_t>& indices) {
    std::vector<size_t> out;
    out.reserve(indices.size());
    for (auto i : indices) {
      out.push_back(static_cast<size_t>(i));
    }
    return SubsetRandomSampler(std::move(out));
  }

  void reset(std::optional<size_t> new_size = std::nullopt) override {
    // new_size is ignored for subset (population is the index list).
    (void)new_size;
    order_ = torch::randperm(static_cast<int64_t>(indices_.size()), torch::kInt64);
    index_ = 0;
  }

  std::optional<std::vector<size_t>> next(size_t batch_size) override {
    const size_t n = indices_.size();
    if (index_ >= n) {
      return std::nullopt;
    }
    const size_t remaining = n - index_;
    const size_t take = std::min(batch_size, remaining);
    std::vector<size_t> batch(take);
    auto slice = order_.slice(/*dim=*/0, static_cast<int64_t>(index_),
                              static_cast<int64_t>(index_ + take));
    slice = slice.to(torch::kInt64);
    const auto* data = slice.const_data_ptr<int64_t>();
    for (size_t i = 0; i < take; ++i) {
      batch[i] = indices_[static_cast<size_t>(data[i])];
    }
    index_ += take;
    return batch;
  }

  void save(serialize::OutputArchive& archive) const override {
    archive.write(
        "index",
        torch::tensor(static_cast<int64_t>(index_), torch::kInt64),
        /*is_buffer=*/true);
    archive.write("order", order_, /*is_buffer=*/true);
    // indices as int64 tensor
    auto idx_t = torch::empty(
        {static_cast<int64_t>(indices_.size())}, torch::kInt64);
    auto* p = idx_t.mutable_data_ptr<int64_t>();
    for (size_t i = 0; i < indices_.size(); ++i) {
      p[i] = static_cast<int64_t>(indices_[i]);
    }
    archive.write("indices", idx_t, /*is_buffer=*/true);
  }

  void load(serialize::InputArchive& archive) override {
    auto tensor = torch::empty(1, torch::kInt64);
    archive.read("index", tensor, /*is_buffer=*/true);
    index_ = static_cast<size_t>(tensor.item<int64_t>());
    archive.read("order", order_, /*is_buffer=*/true);
    Tensor idx_t;
    archive.read("indices", idx_t, /*is_buffer=*/true);
    idx_t = idx_t.to(torch::kInt64).contiguous();
    const auto* p = idx_t.const_data_ptr<int64_t>();
    indices_.resize(static_cast<size_t>(idx_t.numel()));
    for (size_t i = 0; i < indices_.size(); ++i) {
      indices_[i] = static_cast<size_t>(p[i]);
    }
  }

  size_t index() const noexcept {
    return index_;
  }

  size_t size() const noexcept {
    return indices_.size();
  }

 private:
  std::vector<size_t> indices_;
  Tensor order_;
  size_t index_ = 0;
};

// ============================================================================
// WeightedRandomSampler
// ============================================================================

/// Draws {@code num_samples} indices with probabilities proportional to
/// {@code weights}. Mirrors Python {@code torch.utils.data.WeightedRandomSampler}.
/// On {@code reset()}, a new multinomial draw is performed for the epoch;
/// {@code next(batch_size)} walks that draw in order.
class WeightedRandomSampler : public Sampler<> {
 public:
  WeightedRandomSampler(
      std::vector<double> weights,
      size_t num_samples,
      bool replacement = true)
      : weights_(torch::empty(
                     {static_cast<int64_t>(weights.size())},
                     torch::dtype(torch::kDouble))),
        num_samples_(num_samples),
        replacement_(replacement) {
    TORCH_CHECK(!weights.empty(), "weights must be non-empty");
    TORCH_CHECK(num_samples > 0, "num_samples must be > 0");
    if (!replacement) {
      TORCH_CHECK(
          num_samples <= weights.size(),
          "Cannot sample n_sample > pop_size without replacement");
    }
    auto* p = weights_.mutable_data_ptr<double>();
    for (size_t i = 0; i < weights.size(); ++i) {
      p[i] = weights[i];
    }
    reset(std::nullopt);
  }

  /// Construct from a 1-D weights tensor (copied as double).
  WeightedRandomSampler(
      const Tensor& weights,
      size_t num_samples,
      bool replacement = true)
      : weights_(weights.to(torch::kDouble).contiguous()),
        num_samples_(num_samples),
        replacement_(replacement) {
    TORCH_CHECK(weights_.dim() == 1, "weights must be 1-D");
    TORCH_CHECK(weights_.numel() > 0, "weights must be non-empty");
    TORCH_CHECK(num_samples > 0, "num_samples must be > 0");
    if (!replacement) {
      TORCH_CHECK(
          static_cast<int64_t>(num_samples) <= weights_.numel(),
          "Cannot sample n_sample > pop_size without replacement");
    }
    reset(std::nullopt);
  }

  void reset(std::optional<size_t> new_size = std::nullopt) override {
    if (new_size.has_value()) {
      num_samples_ = *new_size;
      if (!replacement_) {
        TORCH_CHECK(
            static_cast<int64_t>(num_samples_) <= weights_.numel(),
            "Cannot sample n_sample > pop_size without replacement");
      }
    }
    // One multinomial draw for the whole epoch (Python __iter__ behavior).
    samples_ = torch::multinomial(
        weights_,
        static_cast<int64_t>(num_samples_),
        replacement_);
    samples_ = samples_.to(torch::kInt64).contiguous();
    index_ = 0;
  }

  std::optional<std::vector<size_t>> next(size_t batch_size) override {
    const size_t n = static_cast<size_t>(samples_.numel());
    if (index_ >= n) {
      return std::nullopt;
    }
    const size_t remaining = n - index_;
    const size_t take = std::min(batch_size, remaining);
    std::vector<size_t> batch(take);
    auto slice = samples_.slice(
        /*dim=*/0,
        static_cast<int64_t>(index_),
        static_cast<int64_t>(index_ + take));
    const auto* data = slice.const_data_ptr<int64_t>();
    for (size_t i = 0; i < take; ++i) {
      batch[i] = static_cast<size_t>(data[i]);
    }
    index_ += take;
    return batch;
  }

  void save(serialize::OutputArchive& archive) const override {
    archive.write(
        "index",
        torch::tensor(static_cast<int64_t>(index_), torch::kInt64),
        /*is_buffer=*/true);
    archive.write(
        "num_samples",
        torch::tensor(static_cast<int64_t>(num_samples_), torch::kInt64),
        /*is_buffer=*/true);
    archive.write(
        "replacement",
        torch::tensor(replacement_ ? 1 : 0, torch::kInt64),
        /*is_buffer=*/true);
    archive.write("weights", weights_, /*is_buffer=*/true);
    archive.write("samples", samples_, /*is_buffer=*/true);
  }

  void load(serialize::InputArchive& archive) override {
    auto t = torch::empty(1, torch::kInt64);
    archive.read("index", t, /*is_buffer=*/true);
    index_ = static_cast<size_t>(t.item<int64_t>());
    archive.read("num_samples", t, /*is_buffer=*/true);
    num_samples_ = static_cast<size_t>(t.item<int64_t>());
    archive.read("replacement", t, /*is_buffer=*/true);
    replacement_ = t.item<int64_t>() != 0;
    archive.read("weights", weights_, /*is_buffer=*/true);
    archive.read("samples", samples_, /*is_buffer=*/true);
  }

  size_t index() const noexcept {
    return index_;
  }

  size_t size() const noexcept {
    return num_samples_;
  }

 private:
  Tensor weights_;
  size_t num_samples_;
  bool replacement_;
  Tensor samples_;
  size_t index_ = 0;
};

// ============================================================================
// BatchSampler — drop_last wrapper around another Sampler<>
// ============================================================================

/// Wraps another {@code Sampler<>} and optionally drops the final incomplete
/// batch ({@code drop_last=true}), matching Python
/// {@code torch.utils.data.BatchSampler} drop_last semantics on top of the C++
/// {@code next(batch_size)} batching model.
///
/// Ownership: holds a {@code shared_ptr} to the inner sampler so Java can keep
/// both peers alive.
class BatchSampler : public Sampler<> {
 public:
  BatchSampler(std::shared_ptr<Sampler<>> sampler, bool drop_last)
      : sampler_(std::move(sampler)), drop_last_(drop_last) {
    TORCH_CHECK(sampler_ != nullptr, "BatchSampler requires a non-null sampler");
  }

  void reset(std::optional<size_t> new_size = std::nullopt) override {
    sampler_->reset(new_size);
  }

  std::optional<std::vector<size_t>> next(size_t batch_size) override {
    auto batch = sampler_->next(batch_size);
    if (!batch.has_value()) {
      return std::nullopt;
    }
    if (drop_last_ && batch->size() < batch_size) {
      return std::nullopt;
    }
    return batch;
  }

  void save(serialize::OutputArchive& archive) const override {
    archive.write(
        "drop_last",
        torch::tensor(drop_last_ ? 1 : 0, torch::kInt64),
        /*is_buffer=*/true);
    // Inner sampler serialization is the caller's responsibility if needed;
    // we only persist drop_last here to keep the wrapper self-contained.
  }

  void load(serialize::InputArchive& archive) override {
    auto t = torch::empty(1, torch::kInt64);
    archive.read("drop_last", t, /*is_buffer=*/true);
    drop_last_ = t.item<int64_t>() != 0;
  }

  Sampler<>& sampler() {
    return *sampler_;
  }

  bool drop_last() const noexcept {
    return drop_last_;
  }

 private:
  std::shared_ptr<Sampler<>> sampler_;
  bool drop_last_;
};

} // namespace torch::data::samplers
