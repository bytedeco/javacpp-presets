/*
  I don't think we can directly virtualize Dataset<...> because of CRTP in Dataset.

  Because of issue #723, we cannot virtualize superclasses of javacpp::*Dataset, only javacpp::*Dataset.
  We must redeclare/redefine virtual functions of parents in these classes, so that the JavaCPP peer classes implement
  the wrappers that call the Java implementations.
*/

namespace javacpp {

/**
 * Abstract class for stateless datasets to be subclassed by Java user code.
 */
 template <typename Data, typename Target>
 struct Dataset : public torch::data::datasets::Dataset<javacpp::Dataset<Data,Target>, torch::data::Example<Data, Target>> {
   virtual ~Dataset() = default;
   virtual torch::data::Example<Data, Target> get(size_t index) override = 0;
   virtual std::optional<size_t> size() const override = 0;
   virtual std::vector<torch::data::Example<Data, Target>> get_batch(c10::ArrayRef<size_t> indices) override {
     return torch::data::datasets::Dataset<javacpp::Dataset<Data, Target>, torch::data::Example<Data, Target>>::get_batch(indices);
   };
};

/**
 * Abstract class for stateless stream datasets to be subclassed by Java user code.
 */
template <typename Data, typename Target>
struct StreamDataset : public torch::data::datasets::BatchDataset<javacpp::StreamDataset<Data,Target>, std::vector<torch::data::Example<Data,Target>>, size_t> {
    virtual ~StreamDataset() = default;
    virtual std::optional<size_t> size() const override = 0;
    virtual std::vector<torch::data::Example<Data,Target>> get_batch(size_t size) override = 0;
};

/**
 * Abstract class for stateful datasets to be subclassed by Java user code.
 */
template <typename Data, typename Target>
struct StatefulDataset : public torch::data::datasets::StatefulDataset<javacpp::StatefulDataset<Data,Target>, std::vector<torch::data::Example<Data,Target>>, size_t> {
  virtual ~StatefulDataset() = default;
  virtual std::optional<size_t> size() const override = 0;
  virtual std::optional<std::vector<torch::data::Example<Data,Target>>> get_batch(size_t size) override = 0;
  virtual void reset() override = 0;
  virtual void save(torch::serialize::OutputArchive& archive) const override = 0;
  virtual void load(torch::serialize::InputArchive& archive) override = 0;
};

}