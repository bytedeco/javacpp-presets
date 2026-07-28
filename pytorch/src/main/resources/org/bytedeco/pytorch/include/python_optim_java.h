/*
 * Python torch.optim algorithms missing from C++ LibTorch, implemented as
 * real subclasses of torch::optim::Optimizer so JavaCPP peers
 * `extends org.bytedeco.pytorch.optim.Optimizer` and reuse
 * OptimizerParamGroup / OptimizerParamGroupVector.
 *
 * Algorithms follow the single-tensor functionals in torch/optim single-tensor functionals.
 * Header-only: symbols are compiled into the JavaCPP JNI library.
 */
#pragma once

#include <torch/arg.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>
#include <torch/serialize/archive.h>
#include <torch/utils.h>

#include <c10/util/irange.h>

#include <cmath>
#include <tuple>
#include <utility>
#include <vector>

namespace torch {
namespace serialize {
class OutputArchive;
class InputArchive;
} // namespace serialize
} // namespace torch

namespace torch::optim {

// ============================================================================
// Adadelta
// ============================================================================

struct AdadeltaOptions : public OptimizerCloneableOptions<AdadeltaOptions> {
  AdadeltaOptions(double lr = 1.0) : lr_(lr) {}
  TORCH_ARG(double, lr) = 1.0;
  TORCH_ARG(double, rho) = 0.9;
  TORCH_ARG(double, eps) = 1e-6;
  TORCH_ARG(double, weight_decay) = 0;

 public:
  void serialize(torch::serialize::OutputArchive& archive) const override {
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(rho);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
  }
  void serialize(torch::serialize::InputArchive& archive) override {
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, rho);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
  }
  double get_lr() const override { return lr(); }
  void set_lr(const double lr) override { this->lr(lr); }
};

inline bool operator==(const AdadeltaOptions& lhs, const AdadeltaOptions& rhs) {
  return lhs.lr() == rhs.lr() && lhs.rho() == rhs.rho() &&
      lhs.eps() == rhs.eps() && lhs.weight_decay() == rhs.weight_decay();
}

struct AdadeltaParamState
    : public OptimizerCloneableParamState<AdadeltaParamState> {
  TORCH_ARG(int64_t, step) = 0;
  TORCH_ARG(torch::Tensor, square_avg);
  TORCH_ARG(torch::Tensor, acc_delta);

 public:
  void serialize(torch::serialize::OutputArchive& archive) const override {
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(square_avg);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(acc_delta);
  }
  void serialize(torch::serialize::InputArchive& archive) override {
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, square_avg);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, acc_delta);
  }
};

inline bool operator==(
    const AdadeltaParamState& lhs,
    const AdadeltaParamState& rhs) {
  return lhs.step() == rhs.step() &&
      torch::equal(lhs.square_avg(), rhs.square_avg()) &&
      torch::equal(lhs.acc_delta(), rhs.acc_delta());
}

class Adadelta : public Optimizer {
 public:
  explicit Adadelta(
      const std::vector<OptimizerParamGroup>& param_groups,
      AdadeltaOptions defaults = {})
      : Optimizer(
            param_groups,
            std::make_unique<AdadeltaOptions>(defaults)) {
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
    TORCH_CHECK(
        defaults.rho() >= 0 && defaults.rho() <= 1,
        "Invalid rho value: ",
        defaults.rho());
    TORCH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());
    TORCH_CHECK(
        defaults.weight_decay() >= 0,
        "Invalid weight_decay value: ",
        defaults.weight_decay());
  }
  explicit Adadelta(std::vector<Tensor> params, AdadeltaOptions defaults = {})
      : Adadelta(
            {OptimizerParamGroup(std::move(params))},
            std::move(defaults)) {}

  torch::Tensor step(LossClosure closure = nullptr) override {
    NoGradGuard no_grad;
    Tensor loss = {};
    if (closure != nullptr) {
      at::AutoGradMode enable_grad(true);
      loss = closure();
    }
    for (auto& group : param_groups_) {
      auto& options = static_cast<AdadeltaOptions&>(group.options());
      for (auto& p : group.params()) {
        if (!p.grad().defined()) {
          continue;
        }
        auto grad = p.grad();
        TORCH_CHECK(
            !grad.is_sparse(), "Adadelta does not support sparse gradients");
        auto param_state = state_.find(p.unsafeGetTensorImpl());
        if (param_state == state_.end()) {
          auto state = std::make_unique<AdadeltaParamState>();
          state->step(0);
          state->square_avg(torch::zeros_like(p, MemoryFormat::Preserve));
          state->acc_delta(torch::zeros_like(p, MemoryFormat::Preserve));
          state_[p.unsafeGetTensorImpl()] = std::move(state);
        }
        auto& state =
            static_cast<AdadeltaParamState&>(*state_[p.unsafeGetTensorImpl()]);
        state.step(state.step() + 1);
        if (options.weight_decay() != 0) {
          grad = grad.add(p, options.weight_decay());
        }
        auto& square_avg = state.square_avg();
        auto& acc_delta = state.acc_delta();
        square_avg.mul_(options.rho())
            .addcmul_(grad, grad, 1 - options.rho());
        auto std = square_avg.add(options.eps()).sqrt_();
        auto delta = acc_delta.add(options.eps()).sqrt_();
        delta.div_(std).mul_(grad);
        acc_delta.mul_(options.rho())
            .addcmul_(delta, delta, 1 - options.rho());
        p.add_(delta, -options.lr());
      }
    }
    return loss;
  }

  void save(serialize::OutputArchive& archive) const override {
    serialize(*this, archive);
  }
  void load(serialize::InputArchive& archive) override {
    serialize(*this, archive);
  }

 private:
  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(Adadelta);
  }
};

// ============================================================================
// Adamax
// ============================================================================

struct AdamaxOptions : public OptimizerCloneableOptions<AdamaxOptions> {
  AdamaxOptions(double lr = 2e-3) : lr_(lr) {}
  TORCH_ARG(double, lr) = 2e-3;
  typedef std::tuple<double, double> betas_t;
  TORCH_ARG(betas_t, betas) = std::make_tuple(0.9, 0.999);
  TORCH_ARG(double, eps) = 1e-8;
  TORCH_ARG(double, weight_decay) = 0;

 public:
  void serialize(torch::serialize::OutputArchive& archive) const override {
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(betas);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
  }
  void serialize(torch::serialize::InputArchive& archive) override {
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(betas_t, betas);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
  }
  double get_lr() const override { return lr(); }
  void set_lr(const double lr) override { this->lr(lr); }
};

inline bool operator==(const AdamaxOptions& lhs, const AdamaxOptions& rhs) {
  return lhs.lr() == rhs.lr() &&
      std::get<0>(lhs.betas()) == std::get<0>(rhs.betas()) &&
      std::get<1>(lhs.betas()) == std::get<1>(rhs.betas()) &&
      lhs.eps() == rhs.eps() && lhs.weight_decay() == rhs.weight_decay();
}

struct AdamaxParamState
    : public OptimizerCloneableParamState<AdamaxParamState> {
  TORCH_ARG(int64_t, step) = 0;
  TORCH_ARG(torch::Tensor, exp_avg);
  TORCH_ARG(torch::Tensor, exp_inf);

 public:
  void serialize(torch::serialize::OutputArchive& archive) const override {
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_inf);
  }
  void serialize(torch::serialize::InputArchive& archive) override {
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, exp_avg);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, exp_inf);
  }
};

inline bool operator==(
    const AdamaxParamState& lhs,
    const AdamaxParamState& rhs) {
  return lhs.step() == rhs.step() &&
      torch::equal(lhs.exp_avg(), rhs.exp_avg()) &&
      torch::equal(lhs.exp_inf(), rhs.exp_inf());
}

class Adamax : public Optimizer {
 public:
  explicit Adamax(
      const std::vector<OptimizerParamGroup>& param_groups,
      AdamaxOptions defaults = {})
      : Optimizer(param_groups, std::make_unique<AdamaxOptions>(defaults)) {
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
    TORCH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());
    auto betas = defaults.betas();
    TORCH_CHECK(
        0 <= std::get<0>(betas) && std::get<0>(betas) < 1.0,
        "Invalid beta parameter at index 0: ",
        std::get<0>(betas));
    TORCH_CHECK(
        0 <= std::get<1>(betas) && std::get<1>(betas) < 1.0,
        "Invalid beta parameter at index 1: ",
        std::get<1>(betas));
    TORCH_CHECK(
        defaults.weight_decay() >= 0,
        "Invalid weight_decay value: ",
        defaults.weight_decay());
  }
  explicit Adamax(std::vector<Tensor> params, AdamaxOptions defaults = {})
      : Adamax({OptimizerParamGroup(std::move(params))}, std::move(defaults)) {}

  torch::Tensor step(LossClosure closure = nullptr) override {
    NoGradGuard no_grad;
    Tensor loss = {};
    if (closure != nullptr) {
      at::AutoGradMode enable_grad(true);
      loss = closure();
    }
    for (auto& group : param_groups_) {
      auto& options = static_cast<AdamaxOptions&>(group.options());
      auto beta1 = std::get<0>(options.betas());
      auto beta2 = std::get<1>(options.betas());
      for (auto& p : group.params()) {
        if (!p.grad().defined()) {
          continue;
        }
        auto grad = p.grad();
        TORCH_CHECK(
            !grad.is_sparse(), "Adamax does not support sparse gradients");
        auto param_state = state_.find(p.unsafeGetTensorImpl());
        if (param_state == state_.end()) {
          auto state = std::make_unique<AdamaxParamState>();
          state->step(0);
          state->exp_avg(torch::zeros_like(p, MemoryFormat::Preserve));
          state->exp_inf(torch::zeros_like(p, MemoryFormat::Preserve));
          state_[p.unsafeGetTensorImpl()] = std::move(state);
        }
        auto& state =
            static_cast<AdamaxParamState&>(*state_[p.unsafeGetTensorImpl()]);
        state.step(state.step() + 1);
        if (options.weight_decay() != 0) {
          grad = grad.add(p, options.weight_decay());
        }
        auto& exp_avg = state.exp_avg();
        auto& exp_inf = state.exp_inf();
        exp_avg.mul_(beta1).add_(grad, 1 - beta1);
        // exp_inf = max(exp_inf * beta2, |grad| + eps)
        auto norm_buf = grad.abs().add_(options.eps());
        exp_inf.mul_(beta2);
        torch::max_out(exp_inf, exp_inf, norm_buf);
        auto bias_correction = 1 - std::pow(beta1, state.step());
        auto clr = options.lr() / bias_correction;
        p.addcdiv_(exp_avg, exp_inf, -clr);
      }
    }
    return loss;
  }

  void save(serialize::OutputArchive& archive) const override {
    serialize(*this, archive);
  }
  void load(serialize::InputArchive& archive) override {
    serialize(*this, archive);
  }

 private:
  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(Adamax);
  }
};

// ============================================================================
// ASGD
// ============================================================================

struct ASGDOptions : public OptimizerCloneableOptions<ASGDOptions> {
  ASGDOptions(double lr = 1e-2) : lr_(lr) {}
  TORCH_ARG(double, lr) = 1e-2;
  TORCH_ARG(double, lambd) = 1e-4;
  TORCH_ARG(double, alpha) = 0.75;
  TORCH_ARG(double, t0) = 1e6;
  TORCH_ARG(double, weight_decay) = 0;

 public:
  void serialize(torch::serialize::OutputArchive& archive) const override {
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lambd);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(alpha);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(t0);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
  }
  void serialize(torch::serialize::InputArchive& archive) override {
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lambd);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, alpha);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, t0);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
  }
  double get_lr() const override { return lr(); }
  void set_lr(const double lr) override { this->lr(lr); }
};

inline bool operator==(const ASGDOptions& lhs, const ASGDOptions& rhs) {
  return lhs.lr() == rhs.lr() && lhs.lambd() == rhs.lambd() &&
      lhs.alpha() == rhs.alpha() && lhs.t0() == rhs.t0() &&
      lhs.weight_decay() == rhs.weight_decay();
}

struct ASGDParamState : public OptimizerCloneableParamState<ASGDParamState> {
  TORCH_ARG(int64_t, step) = 0;
  TORCH_ARG(torch::Tensor, ax);
  // Stored as scalar tensors to match Python state layout loosely
  TORCH_ARG(double, eta) = 0;
  TORCH_ARG(double, mu) = 1;

 public:
  void serialize(torch::serialize::OutputArchive& archive) const override {
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(ax);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eta);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(mu);
  }
  void serialize(torch::serialize::InputArchive& archive) override {
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, ax);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eta);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, mu);
  }
};

inline bool operator==(const ASGDParamState& lhs, const ASGDParamState& rhs) {
  return lhs.step() == rhs.step() && torch::equal(lhs.ax(), rhs.ax()) &&
      lhs.eta() == rhs.eta() && lhs.mu() == rhs.mu();
}

class ASGD : public Optimizer {
 public:
  explicit ASGD(
      const std::vector<OptimizerParamGroup>& param_groups,
      ASGDOptions defaults = {})
      : Optimizer(param_groups, std::make_unique<ASGDOptions>(defaults)) {
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
    TORCH_CHECK(
        defaults.lambd() >= 0, "Invalid lambda value: ", defaults.lambd());
    TORCH_CHECK(
        defaults.alpha() >= 0, "Invalid alpha value: ", defaults.alpha());
    TORCH_CHECK(defaults.t0() >= 0, "Invalid t0 value: ", defaults.t0());
    TORCH_CHECK(
        defaults.weight_decay() >= 0,
        "Invalid weight_decay value: ",
        defaults.weight_decay());
  }
  explicit ASGD(std::vector<Tensor> params, ASGDOptions defaults = {})
      : ASGD({OptimizerParamGroup(std::move(params))}, std::move(defaults)) {}

  torch::Tensor step(LossClosure closure = nullptr) override {
    NoGradGuard no_grad;
    Tensor loss = {};
    if (closure != nullptr) {
      at::AutoGradMode enable_grad(true);
      loss = closure();
    }
    for (auto& group : param_groups_) {
      auto& options = static_cast<ASGDOptions&>(group.options());
      for (auto& p : group.params()) {
        if (!p.grad().defined()) {
          continue;
        }
        auto grad = p.grad();
        TORCH_CHECK(!grad.is_sparse(), "ASGD does not support sparse gradients");
        auto param_state = state_.find(p.unsafeGetTensorImpl());
        if (param_state == state_.end()) {
          auto state = std::make_unique<ASGDParamState>();
          state->step(0);
          state->ax(torch::zeros_like(p, MemoryFormat::Preserve));
          state->eta(options.lr());
          state->mu(1.0);
          state_[p.unsafeGetTensorImpl()] = std::move(state);
        }
        auto& state =
            static_cast<ASGDParamState&>(*state_[p.unsafeGetTensorImpl()]);
        state.step(state.step() + 1);
        if (options.weight_decay() != 0) {
          grad = grad.add(p, options.weight_decay());
        }
        const double eta = state.eta();
        p.mul_(1 - options.lambd() * eta);
        p.add_(grad, -eta);

        auto& ax = state.ax();
        if (state.mu() != 1.0) {
          ax.add_(p.sub(ax).mul_(state.mu()));
        } else {
          ax.copy_(p);
        }
        const double step = static_cast<double>(state.step());
        state.eta(
            options.lr() /
            std::pow(1.0 + options.lambd() * options.lr() * step, options.alpha()));
        state.mu(1.0 / std::max(1.0, step - options.t0()));
      }
    }
    return loss;
  }

  void save(serialize::OutputArchive& archive) const override {
    serialize(*this, archive);
  }
  void load(serialize::InputArchive& archive) override {
    serialize(*this, archive);
  }

 private:
  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(ASGD);
  }
};

// ============================================================================
// NAdam
// ============================================================================

struct NAdamOptions : public OptimizerCloneableOptions<NAdamOptions> {
  NAdamOptions(double lr = 2e-3) : lr_(lr) {}
  TORCH_ARG(double, lr) = 2e-3;
  typedef std::tuple<double, double> betas_t;
  TORCH_ARG(betas_t, betas) = std::make_tuple(0.9, 0.999);
  TORCH_ARG(double, eps) = 1e-8;
  TORCH_ARG(double, weight_decay) = 0;
  TORCH_ARG(double, momentum_decay) = 4e-3;
  TORCH_ARG(bool, decoupled_weight_decay) = false;

 public:
  void serialize(torch::serialize::OutputArchive& archive) const override {
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(betas);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(momentum_decay);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(decoupled_weight_decay);
  }
  void serialize(torch::serialize::InputArchive& archive) override {
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(betas_t, betas);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, momentum_decay);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, decoupled_weight_decay);
  }
  double get_lr() const override { return lr(); }
  void set_lr(const double lr) override { this->lr(lr); }
};

inline bool operator==(const NAdamOptions& lhs, const NAdamOptions& rhs) {
  return lhs.lr() == rhs.lr() &&
      std::get<0>(lhs.betas()) == std::get<0>(rhs.betas()) &&
      std::get<1>(lhs.betas()) == std::get<1>(rhs.betas()) &&
      lhs.eps() == rhs.eps() && lhs.weight_decay() == rhs.weight_decay() &&
      lhs.momentum_decay() == rhs.momentum_decay() &&
      lhs.decoupled_weight_decay() == rhs.decoupled_weight_decay();
}

struct NAdamParamState
    : public OptimizerCloneableParamState<NAdamParamState> {
  TORCH_ARG(int64_t, step) = 0;
  TORCH_ARG(torch::Tensor, exp_avg);
  TORCH_ARG(torch::Tensor, exp_avg_sq);
  TORCH_ARG(double, mu_product) = 1.0;

 public:
  void serialize(torch::serialize::OutputArchive& archive) const override {
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg_sq);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(mu_product);
  }
  void serialize(torch::serialize::InputArchive& archive) override {
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, exp_avg);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, exp_avg_sq);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, mu_product);
  }
};

inline bool operator==(
    const NAdamParamState& lhs,
    const NAdamParamState& rhs) {
  return lhs.step() == rhs.step() &&
      torch::equal(lhs.exp_avg(), rhs.exp_avg()) &&
      torch::equal(lhs.exp_avg_sq(), rhs.exp_avg_sq()) &&
      lhs.mu_product() == rhs.mu_product();
}

class NAdam : public Optimizer {
 public:
  explicit NAdam(
      const std::vector<OptimizerParamGroup>& param_groups,
      NAdamOptions defaults = {})
      : Optimizer(param_groups, std::make_unique<NAdamOptions>(defaults)) {
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
    TORCH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());
    auto betas = defaults.betas();
    TORCH_CHECK(
        0 <= std::get<0>(betas) && std::get<0>(betas) < 1.0,
        "Invalid beta parameter at index 0: ",
        std::get<0>(betas));
    TORCH_CHECK(
        0 <= std::get<1>(betas) && std::get<1>(betas) < 1.0,
        "Invalid beta parameter at index 1: ",
        std::get<1>(betas));
    TORCH_CHECK(
        defaults.weight_decay() >= 0,
        "Invalid weight_decay value: ",
        defaults.weight_decay());
    TORCH_CHECK(
        defaults.momentum_decay() >= 0,
        "Invalid momentum_decay value: ",
        defaults.momentum_decay());
  }
  explicit NAdam(std::vector<Tensor> params, NAdamOptions defaults = {})
      : NAdam({OptimizerParamGroup(std::move(params))}, std::move(defaults)) {}

  torch::Tensor step(LossClosure closure = nullptr) override {
    NoGradGuard no_grad;
    Tensor loss = {};
    if (closure != nullptr) {
      at::AutoGradMode enable_grad(true);
      loss = closure();
    }
    for (auto& group : param_groups_) {
      auto& options = static_cast<NAdamOptions&>(group.options());
      auto beta1 = std::get<0>(options.betas());
      auto beta2 = std::get<1>(options.betas());
      for (auto& p : group.params()) {
        if (!p.grad().defined()) {
          continue;
        }
        auto grad = p.grad();
        TORCH_CHECK(
            !grad.is_sparse(), "NAdam does not support sparse gradients");
        auto param_state = state_.find(p.unsafeGetTensorImpl());
        if (param_state == state_.end()) {
          auto state = std::make_unique<NAdamParamState>();
          state->step(0);
          state->exp_avg(torch::zeros_like(p, MemoryFormat::Preserve));
          state->exp_avg_sq(torch::zeros_like(p, MemoryFormat::Preserve));
          state->mu_product(1.0);
          state_[p.unsafeGetTensorImpl()] = std::move(state);
        }
        auto& state =
            static_cast<NAdamParamState&>(*state_[p.unsafeGetTensorImpl()]);
        state.step(state.step() + 1);
        const double step = static_cast<double>(state.step());

        if (options.weight_decay() != 0) {
          if (options.decoupled_weight_decay()) {
            p.mul_(1 - options.lr() * options.weight_decay());
          } else {
            grad = grad.add(p, options.weight_decay());
          }
        }

        const double mu =
            beta1 * (1.0 - 0.5 * std::pow(0.96, step * options.momentum_decay()));
        const double mu_next = beta1 *
            (1.0 - 0.5 * std::pow(0.96, (step + 1) * options.momentum_decay()));
        state.mu_product(state.mu_product() * mu);

        auto& exp_avg = state.exp_avg();
        auto& exp_avg_sq = state.exp_avg_sq();
        exp_avg.mul_(beta1).add_(grad, 1 - beta1);
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, 1 - beta2);

        const double bias_correction2 = 1 - std::pow(beta2, step);
        auto denom = exp_avg_sq.div(bias_correction2).sqrt().add_(options.eps());
        const double mu_product = state.mu_product();
        const double mu_product_next = mu_product * mu_next;
        p.addcdiv_(
            grad, denom, -options.lr() * (1.0 - mu) / (1.0 - mu_product));
        p.addcdiv_(
            exp_avg,
            denom,
            -options.lr() * mu_next / (1.0 - mu_product_next));
      }
    }
    return loss;
  }

  void save(serialize::OutputArchive& archive) const override {
    serialize(*this, archive);
  }
  void load(serialize::InputArchive& archive) override {
    serialize(*this, archive);
  }

 private:
  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(NAdam);
  }
};

// ============================================================================
// RAdam
// ============================================================================

struct RAdamOptions : public OptimizerCloneableOptions<RAdamOptions> {
  RAdamOptions(double lr = 1e-3) : lr_(lr) {}
  TORCH_ARG(double, lr) = 1e-3;
  typedef std::tuple<double, double> betas_t;
  TORCH_ARG(betas_t, betas) = std::make_tuple(0.9, 0.999);
  TORCH_ARG(double, eps) = 1e-8;
  TORCH_ARG(double, weight_decay) = 0;
  TORCH_ARG(bool, decoupled_weight_decay) = false;

 public:
  void serialize(torch::serialize::OutputArchive& archive) const override {
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(betas);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(decoupled_weight_decay);
  }
  void serialize(torch::serialize::InputArchive& archive) override {
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(betas_t, betas);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, decoupled_weight_decay);
  }
  double get_lr() const override { return lr(); }
  void set_lr(const double lr) override { this->lr(lr); }
};

inline bool operator==(const RAdamOptions& lhs, const RAdamOptions& rhs) {
  return lhs.lr() == rhs.lr() &&
      std::get<0>(lhs.betas()) == std::get<0>(rhs.betas()) &&
      std::get<1>(lhs.betas()) == std::get<1>(rhs.betas()) &&
      lhs.eps() == rhs.eps() && lhs.weight_decay() == rhs.weight_decay() &&
      lhs.decoupled_weight_decay() == rhs.decoupled_weight_decay();
}

struct RAdamParamState
    : public OptimizerCloneableParamState<RAdamParamState> {
  TORCH_ARG(int64_t, step) = 0;
  TORCH_ARG(torch::Tensor, exp_avg);
  TORCH_ARG(torch::Tensor, exp_avg_sq);

 public:
  void serialize(torch::serialize::OutputArchive& archive) const override {
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg_sq);
  }
  void serialize(torch::serialize::InputArchive& archive) override {
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, exp_avg);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, exp_avg_sq);
  }
};

inline bool operator==(
    const RAdamParamState& lhs,
    const RAdamParamState& rhs) {
  return lhs.step() == rhs.step() &&
      torch::equal(lhs.exp_avg(), rhs.exp_avg()) &&
      torch::equal(lhs.exp_avg_sq(), rhs.exp_avg_sq());
}

class RAdam : public Optimizer {
 public:
  explicit RAdam(
      const std::vector<OptimizerParamGroup>& param_groups,
      RAdamOptions defaults = {})
      : Optimizer(param_groups, std::make_unique<RAdamOptions>(defaults)) {
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
    TORCH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());
    auto betas = defaults.betas();
    TORCH_CHECK(
        0 <= std::get<0>(betas) && std::get<0>(betas) < 1.0,
        "Invalid beta parameter at index 0: ",
        std::get<0>(betas));
    TORCH_CHECK(
        0 <= std::get<1>(betas) && std::get<1>(betas) < 1.0,
        "Invalid beta parameter at index 1: ",
        std::get<1>(betas));
    TORCH_CHECK(
        defaults.weight_decay() >= 0,
        "Invalid weight_decay value: ",
        defaults.weight_decay());
  }
  explicit RAdam(std::vector<Tensor> params, RAdamOptions defaults = {})
      : RAdam({OptimizerParamGroup(std::move(params))}, std::move(defaults)) {}

  torch::Tensor step(LossClosure closure = nullptr) override {
    NoGradGuard no_grad;
    Tensor loss = {};
    if (closure != nullptr) {
      at::AutoGradMode enable_grad(true);
      loss = closure();
    }
    for (auto& group : param_groups_) {
      auto& options = static_cast<RAdamOptions&>(group.options());
      auto beta1 = std::get<0>(options.betas());
      auto beta2 = std::get<1>(options.betas());
      for (auto& p : group.params()) {
        if (!p.grad().defined()) {
          continue;
        }
        auto grad = p.grad();
        TORCH_CHECK(
            !grad.is_sparse(), "RAdam does not support sparse gradients");
        auto param_state = state_.find(p.unsafeGetTensorImpl());
        if (param_state == state_.end()) {
          auto state = std::make_unique<RAdamParamState>();
          state->step(0);
          state->exp_avg(torch::zeros_like(p, MemoryFormat::Preserve));
          state->exp_avg_sq(torch::zeros_like(p, MemoryFormat::Preserve));
          state_[p.unsafeGetTensorImpl()] = std::move(state);
        }
        auto& state =
            static_cast<RAdamParamState&>(*state_[p.unsafeGetTensorImpl()]);
        state.step(state.step() + 1);
        const double step = static_cast<double>(state.step());

        if (options.weight_decay() != 0) {
          if (options.decoupled_weight_decay()) {
            p.mul_(1 - options.lr() * options.weight_decay());
          } else {
            grad = grad.add(p, options.weight_decay());
          }
        }

        auto& exp_avg = state.exp_avg();
        auto& exp_avg_sq = state.exp_avg_sq();
        exp_avg.mul_(beta1).add_(grad, 1 - beta1);
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, 1 - beta2);

        const double bias_correction1 = 1 - std::pow(beta1, step);
        const double bias_correction2 = 1 - std::pow(beta2, step);
        auto bias_corrected_exp_avg = exp_avg / bias_correction1;

        const double rho_inf = 2.0 / (1.0 - beta2) - 1.0;
        const double rho_t =
            rho_inf - 2.0 * step * std::pow(beta2, step) / bias_correction2;

        if (rho_t > 5.0) {
          const double rect = std::sqrt(
              (rho_t - 4.0) * (rho_t - 2.0) * rho_inf /
              ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t));
          auto adaptive_lr =
              (std::sqrt(bias_correction2) /
               exp_avg_sq.sqrt().add_(options.eps()));
          p.add_(bias_corrected_exp_avg * options.lr() * adaptive_lr * rect, -1.0);
        } else {
          p.add_(bias_corrected_exp_avg * options.lr(), -1.0);
        }
      }
    }
    return loss;
  }

  void save(serialize::OutputArchive& archive) const override {
    serialize(*this, archive);
  }
  void load(serialize::InputArchive& archive) override {
    serialize(*this, archive);
  }

 private:
  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(RAdam);
  }
};

// ============================================================================
// Rprop
// ============================================================================

struct RpropOptions : public OptimizerCloneableOptions<RpropOptions> {
  RpropOptions(double lr = 1e-2) : lr_(lr) {}
  TORCH_ARG(double, lr) = 1e-2;
  typedef std::tuple<double, double> etas_t;
  TORCH_ARG(etas_t, etas) = std::make_tuple(0.5, 1.2);
  typedef std::tuple<double, double> step_sizes_t;
  TORCH_ARG(step_sizes_t, step_sizes) = std::make_tuple(1e-6, 50);

 public:
  void serialize(torch::serialize::OutputArchive& archive) const override {
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(etas);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step_sizes);
  }
  void serialize(torch::serialize::InputArchive& archive) override {
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(etas_t, etas);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(step_sizes_t, step_sizes);
  }
  double get_lr() const override { return lr(); }
  void set_lr(const double lr) override { this->lr(lr); }
};

inline bool operator==(const RpropOptions& lhs, const RpropOptions& rhs) {
  return lhs.lr() == rhs.lr() &&
      std::get<0>(lhs.etas()) == std::get<0>(rhs.etas()) &&
      std::get<1>(lhs.etas()) == std::get<1>(rhs.etas()) &&
      std::get<0>(lhs.step_sizes()) == std::get<0>(rhs.step_sizes()) &&
      std::get<1>(lhs.step_sizes()) == std::get<1>(rhs.step_sizes());
}

struct RpropParamState
    : public OptimizerCloneableParamState<RpropParamState> {
  TORCH_ARG(int64_t, step) = 0;
  TORCH_ARG(torch::Tensor, prev);
  TORCH_ARG(torch::Tensor, step_size);

 public:
  void serialize(torch::serialize::OutputArchive& archive) const override {
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(prev);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step_size);
  }
  void serialize(torch::serialize::InputArchive& archive) override {
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, prev);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, step_size);
  }
};

inline bool operator==(
    const RpropParamState& lhs,
    const RpropParamState& rhs) {
  return lhs.step() == rhs.step() && torch::equal(lhs.prev(), rhs.prev()) &&
      torch::equal(lhs.step_size(), rhs.step_size());
}

class Rprop : public Optimizer {
 public:
  explicit Rprop(
      const std::vector<OptimizerParamGroup>& param_groups,
      RpropOptions defaults = {})
      : Optimizer(param_groups, std::make_unique<RpropOptions>(defaults)) {
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
    auto etas = defaults.etas();
    TORCH_CHECK(
        0 < std::get<0>(etas) && std::get<0>(etas) < 1.0,
        "Invalid eta parameter at index 0: ",
        std::get<0>(etas));
    TORCH_CHECK(
        1 < std::get<1>(etas),
        "Invalid eta parameter at index 1: ",
        std::get<1>(etas));
  }
  explicit Rprop(std::vector<Tensor> params, RpropOptions defaults = {})
      : Rprop({OptimizerParamGroup(std::move(params))}, std::move(defaults)) {}

  torch::Tensor step(LossClosure closure = nullptr) override {
    NoGradGuard no_grad;
    Tensor loss = {};
    if (closure != nullptr) {
      at::AutoGradMode enable_grad(true);
      loss = closure();
    }
    for (auto& group : param_groups_) {
      auto& options = static_cast<RpropOptions&>(group.options());
      const double etaminus = std::get<0>(options.etas());
      const double etaplus = std::get<1>(options.etas());
      const double step_size_min = std::get<0>(options.step_sizes());
      const double step_size_max = std::get<1>(options.step_sizes());
      for (auto& p : group.params()) {
        if (!p.grad().defined()) {
          continue;
        }
        auto grad = p.grad().clone(MemoryFormat::Preserve);
        TORCH_CHECK(
            !grad.is_sparse(), "Rprop does not support sparse gradients");
        auto param_state = state_.find(p.unsafeGetTensorImpl());
        if (param_state == state_.end()) {
          auto state = std::make_unique<RpropParamState>();
          state->step(0);
          state->prev(torch::zeros_like(p, MemoryFormat::Preserve));
          state->step_size(
              grad.new_empty(grad.sizes()).fill_(options.lr()));
          state_[p.unsafeGetTensorImpl()] = std::move(state);
        }
        auto& state =
            static_cast<RpropParamState&>(*state_[p.unsafeGetTensorImpl()]);
        state.step(state.step() + 1);

        auto& prev = state.prev();
        auto& step_size = state.step_size();
        auto sign = grad.mul(prev).sign();
        // sign > 0 -> etaplus; sign < 0 -> etaminus; else 1
        sign = torch::where(
            sign.gt(0),
            torch::full_like(sign, etaplus),
            torch::where(
                sign.lt(0),
                torch::full_like(sign, etaminus),
                torch::ones_like(sign)));
        step_size.mul_(sign).clamp_(step_size_min, step_size_max);
        // for dir<0, dfdx=0
        grad = torch::where(sign.eq(etaminus), torch::zeros_like(grad), grad);
        p.addcmul_(grad.sign(), step_size, -1.0);
        prev.copy_(grad);
      }
    }
    return loss;
  }

  void save(serialize::OutputArchive& archive) const override {
    serialize(*this, archive);
  }
  void load(serialize::InputArchive& archive) override {
    serialize(*this, archive);
  }

 private:
  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(Rprop);
  }
};

} // namespace torch::optim
