/*
 * Python torch.optim.lr_scheduler classes missing from C++ LibTorch, implemented
 * as real subclasses of torch::optim::LRScheduler so JavaCPP peers
 * `extends org.bytedeco.pytorch.optim.LRScheduler`.
 *
 * Also exposes protected LRScheduler APIs needed from Java (get_current_lrs,
 * step_count) via public accessors on each concrete class / free helpers.
 *
 * Formulas follow torch/optim/lr_scheduler.py (eager path).
 */
#pragma once

#include <torch/optim/optimizer.h>
#include <torch/optim/schedulers/lr_scheduler.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace torch::optim {

// Free helpers so Java can also inspect native StepLR / ReduceLROnPlateau
// after we expose them via Info.javaText if needed. Concrete schedulers below
// call the protected LRScheduler members from C++.

// ============================================================================
// MultiplicativeLR — lr *= gamma each step (after initial)
// ============================================================================

class MultiplicativeLR : public LRScheduler {
 public:
  MultiplicativeLR(Optimizer& optimizer, double gamma)
      : LRScheduler(optimizer), gamma_(gamma) {
    TORCH_CHECK(gamma > 0, "MultiplicativeLR gamma must be > 0, got ", gamma);
  }

 private:
  std::vector<double> get_lrs() override {
    // Python: if _is_initial return current; else multiply by gamma.
    // C++ LRScheduler::step increments step_count_ after get_lrs; first call
    // has step_count_==0 → return current (like initial).
    if (step_count_ == 0) {
      return get_current_lrs();
    }
    auto lrs = get_current_lrs();
    for (auto& lr : lrs) {
      lr *= gamma_;
    }
    return lrs;
  }
  double gamma_;
};

// ============================================================================
// MultiStepLR
// ============================================================================

class MultiStepLR : public LRScheduler {
 public:
  MultiStepLR(
      Optimizer& optimizer,
      std::vector<unsigned> milestones,
      double gamma = 0.1)
      : LRScheduler(optimizer),
        milestones_(milestones.begin(), milestones.end()),
        gamma_(gamma) {
    TORCH_CHECK(gamma > 0, "MultiStepLR gamma must be > 0");
  }

 private:
  std::vector<double> get_lrs() override {
    // Decay when step_count_ is in milestones (Python uses last_epoch which
    // after step equals the milestone value). C++ step_count_ is incremented
    // after get_lrs, so at call time step_count_ is the epoch about to finish
    // / the new last_epoch after increment would be step_count_+1...
    // Match StepLR pattern: StepLR uses step_count_ % step_size with
    // step_count_ BEFORE increment. Python MultiStepLR: if last_epoch in
    // milestones. After Python step, last_epoch is the just-completed epoch
    // index. C++ StepLR: first step step_count_=0 → no decay; after step
    // step_count_=1. So milestone M means decay when step_count_==M.
    if (milestones_.count(step_count_) == 0) {
      return get_current_lrs();
    }
    auto lrs = get_current_lrs();
    for (auto& lr : lrs) {
      lr *= gamma_;
    }
    return lrs;
  }
  std::set<unsigned> milestones_;
  double gamma_;
};

// ============================================================================
// ConstantLR — factor * base for total_iters, then restore
// ============================================================================

class ConstantLR : public LRScheduler {
 public:
  ConstantLR(
      Optimizer& optimizer,
      double factor = 1.0 / 3.0,
      unsigned total_iters = 5)
      : LRScheduler(optimizer),
        factor_(factor),
        total_iters_(total_iters),
        base_lrs_(get_current_lrs()) {
    TORCH_CHECK(factor > 0 && factor <= 1.0, "ConstantLR factor must be in (0, 1]");
  }

 private:
  std::vector<double> get_lrs() override {
    // Python: last_epoch==0 → lr * factor; last_epoch==total_iters → /factor;
    // else unchanged. C++ step_count_ before increment.
    if (step_count_ == 0) {
      auto lrs = get_current_lrs();
      for (auto& lr : lrs) {
        lr *= factor_;
      }
      return lrs;
    }
    if (step_count_ == total_iters_) {
      auto lrs = get_current_lrs();
      for (auto& lr : lrs) {
        lr *= (1.0 / factor_);
      }
      return lrs;
    }
    return get_current_lrs();
  }
  double factor_;
  unsigned total_iters_;
  std::vector<double> base_lrs_;
};

// ============================================================================
// LinearLR — interpolate start_factor → end_factor over total_iters
// ============================================================================

class LinearLR : public LRScheduler {
 public:
  LinearLR(
      Optimizer& optimizer,
      double start_factor = 1.0 / 3.0,
      double end_factor = 1.0,
      unsigned total_iters = 5)
      : LRScheduler(optimizer),
        start_factor_(start_factor),
        end_factor_(end_factor),
        total_iters_(total_iters),
        base_lrs_(get_current_lrs()) {
    TORCH_CHECK(start_factor_ > 0 && start_factor_ <= 1.0);
    TORCH_CHECK(end_factor_ > 0 && end_factor_ <= 1.0);
  }

 private:
  std::vector<double> get_lrs() override {
    if (step_count_ == 0) {
      auto lrs = get_current_lrs();
      for (auto& lr : lrs) {
        lr *= start_factor_;
      }
      return lrs;
    }
    if (step_count_ > total_iters_) {
      return get_current_lrs();
    }
    // multiplicative form from Python LinearLR.get_lr
    const double denom = total_iters_ * start_factor_ +
        (static_cast<double>(step_count_) - 1) * (end_factor_ - start_factor_);
    const double mult =
        1.0 + (end_factor_ - start_factor_) / (denom == 0 ? 1.0 : denom);
    auto lrs = get_current_lrs();
    for (auto& lr : lrs) {
      lr *= mult;
    }
    return lrs;
  }
  double start_factor_;
  double end_factor_;
  unsigned total_iters_;
  std::vector<double> base_lrs_;
};

// ============================================================================
// ExponentialLR
// ============================================================================

class ExponentialLR : public LRScheduler {
 public:
  ExponentialLR(Optimizer& optimizer, double gamma)
      : LRScheduler(optimizer), gamma_(gamma) {
    TORCH_CHECK(gamma > 0, "ExponentialLR gamma must be > 0");
  }

 private:
  std::vector<double> get_lrs() override {
    if (step_count_ == 0) {
      return get_current_lrs();
    }
    auto lrs = get_current_lrs();
    for (auto& lr : lrs) {
      lr *= gamma_;
    }
    return lrs;
  }
  double gamma_;
};

// ============================================================================
// PolynomialLR
// ============================================================================

class PolynomialLR : public LRScheduler {
 public:
  PolynomialLR(
      Optimizer& optimizer,
      unsigned total_iters = 5,
      double power = 1.0)
      : LRScheduler(optimizer),
        total_iters_(total_iters),
        power_(power),
        base_lrs_(get_current_lrs()) {}

 private:
  std::vector<double> get_lrs() override {
    if (step_count_ == 0 || step_count_ > total_iters_) {
      return get_current_lrs();
    }
    const double last = static_cast<double>(step_count_);
    const double total = static_cast<double>(total_iters_);
    const double decay_factor = std::pow(
        (1.0 - last / total) / (1.0 - (last - 1.0) / total), power_);
    auto lrs = get_current_lrs();
    for (auto& lr : lrs) {
      lr *= decay_factor;
    }
    return lrs;
  }
  unsigned total_iters_;
  double power_;
  std::vector<double> base_lrs_;
};

// ============================================================================
// CosineAnnealingLR
// ============================================================================

class CosineAnnealingLR : public LRScheduler {
 public:
  CosineAnnealingLR(
      Optimizer& optimizer,
      unsigned T_max,
      double eta_min = 0.0)
      : LRScheduler(optimizer),
        T_max_(T_max),
        eta_min_(eta_min),
        base_lrs_(get_current_lrs()) {
    TORCH_CHECK(T_max > 0, "T_max must be > 0");
  }

 private:
  std::vector<double> get_lrs() override {
    // Closed-form from base_lrs (more stable than multiplicative Python form
    // for C++ where we always have base_lrs_).
    // lr = eta_min + (base - eta_min) * (1 + cos(pi * epoch / T_max)) / 2
    // epoch == step_count_ after Python semantics; use step_count_ at call
    // (before increment) as last_epoch for closed form matching first step
    // returning base when step_count_==0.
    const double epoch = static_cast<double>(step_count_);
    std::vector<double> lrs;
    lrs.reserve(base_lrs_.size());
    for (double base_lr : base_lrs_) {
      if (epoch == 0) {
        lrs.push_back(base_lr);
      } else {
        lrs.push_back(
            eta_min_ +
            (base_lr - eta_min_) *
                (1.0 + std::cos(3.14159265358979323846 * epoch / T_max_)) / 2.0);
      }
    }
    return lrs;
  }
  unsigned T_max_;
  double eta_min_;
  std::vector<double> base_lrs_;
};

// ============================================================================
// CosineAnnealingWarmRestarts
// ============================================================================

class CosineAnnealingWarmRestarts : public LRScheduler {
 public:
  CosineAnnealingWarmRestarts(
      Optimizer& optimizer,
      unsigned T_0,
      unsigned T_mult = 1,
      double eta_min = 0.0)
      : LRScheduler(optimizer),
        T_i_(T_0),
        T_mult_(T_mult),
        T_cur_(0),
        eta_min_(eta_min),
        base_lrs_(get_current_lrs()) {
    TORCH_CHECK(T_0 > 0, "T_0 must be positive");
    TORCH_CHECK(T_mult >= 1, "T_mult should be >= 1");
  }

 private:
  std::vector<double> get_lrs() override {
    // Advance T_cur like Python step(): on first call step_count_==0 → T_cur=0
    if (step_count_ == 0) {
      T_cur_ = 0;
    } else {
      T_cur_ += 1;
      if (T_cur_ >= static_cast<int>(T_i_)) {
        T_cur_ -= static_cast<int>(T_i_);
        T_i_ = T_i_ * T_mult_;
      }
    }
    std::vector<double> lrs;
    lrs.reserve(base_lrs_.size());
    for (double base_lr : base_lrs_) {
      lrs.push_back(
          eta_min_ +
          (base_lr - eta_min_) *
              (1.0 +
               std::cos(
                   3.14159265358979323846 * static_cast<double>(T_cur_) /
                   static_cast<double>(T_i_))) /
              2.0);
    }
    return lrs;
  }
  unsigned T_i_;
  unsigned T_mult_;
  int T_cur_;
  double eta_min_;
  std::vector<double> base_lrs_;
};

// ============================================================================
// CyclicLR (triangular mode only by default; scale_fn optional via mode string)
// ============================================================================

class CyclicLR : public LRScheduler {
 public:
  CyclicLR(
      Optimizer& optimizer,
      double base_lr,
      double max_lr,
      unsigned step_size_up = 2000,
      unsigned step_size_down = 0,
      const std::string& mode = "triangular",
      double gamma = 1.0)
      : LRScheduler(optimizer),
        base_lr_(base_lr),
        max_lr_(max_lr),
        step_size_up_(step_size_up),
        step_size_down_(step_size_down == 0 ? step_size_up : step_size_down),
        mode_(mode),
        gamma_(gamma) {
    total_size_ = step_size_up_ + step_size_down_;
    step_ratio_ = static_cast<double>(step_size_up_) / total_size_;
    // Initialize all groups to base_lr
    auto n = get_current_lrs().size();
    base_lrs_.assign(n, base_lr_);
    max_lrs_.assign(n, max_lr_);
  }

 private:
  double scale_fn(double x) const {
    if (mode_ == "triangular2") {
      return 1.0 / std::pow(2.0, std::floor((x - 1.0) / 1.0));
    }
    if (mode_ == "exp_range") {
      return std::pow(gamma_, x);
    }
    // triangular
    return 1.0;
  }

  std::vector<double> get_lrs() override {
    const double cycle =
        std::floor(1.0 + static_cast<double>(step_count_) / total_size_);
    const double x =
        1.0 + static_cast<double>(step_count_) / total_size_ - cycle;
    double scale_factor;
    if (x <= step_ratio_) {
      scale_factor = x / step_ratio_;
    } else {
      scale_factor = (x - 1.0) / (step_ratio_ - 1.0);
    }
    std::vector<double> lrs;
    lrs.reserve(base_lrs_.size());
    for (size_t i = 0; i < base_lrs_.size(); ++i) {
      const double base_height = (max_lrs_[i] - base_lrs_[i]) * scale_factor;
      lrs.push_back(base_lrs_[i] + base_height * scale_fn(cycle));
    }
    return lrs;
  }

  double base_lr_;
  double max_lr_;
  unsigned step_size_up_;
  unsigned step_size_down_;
  std::string mode_;
  double gamma_;
  unsigned total_size_{};
  double step_ratio_{};
  std::vector<double> base_lrs_;
  std::vector<double> max_lrs_;
};


} // namespace torch::optim
