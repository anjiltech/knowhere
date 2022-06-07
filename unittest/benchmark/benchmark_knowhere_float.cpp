// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <gtest/gtest.h>

#include <vector>

#include "knowhere/index/IndexType.h"
#include "knowhere/index/VecIndexFactory.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "unittest/benchmark/benchmark_sift.h"
#include "unittest/utils.h"

#define CALC_TIME_SPAN(X)       \
    double t_start = elapsed(); \
    X;                          \
    double t_diff = elapsed() - t_start;

class Benchmark_knowhere_float : public Benchmark_sift {
 public:
    void
    write_index(const std::string& filename, const knowhere::Config& conf) {
        binary_set_.clear();

        FileIOWriter writer(filename);
        binary_set_ = index_->Serialize(conf);

        const auto& m = binary_set_.binary_map_;
        for (auto it = m.begin(); it != m.end(); ++it) {
            const std::string& name = it->first;
            size_t name_size = name.length();
            const knowhere::BinaryPtr data = it->second;
            size_t data_size = data->size;

            writer(&name_size, sizeof(name_size));
            writer(&data_size, sizeof(data_size));
            writer((void*)name.c_str(), name_size);
            writer(data->data.get(), data_size);
        }
    }

    void
    read_index(const std::string& filename) {
        binary_set_.clear();

        FileIOReader reader(filename);
        int64_t file_size = reader.size();
        if (file_size < 0) {
            throw knowhere::KnowhereException(filename + " not exist");
        }

        int64_t offset = 0;
        while (offset < file_size) {
            size_t name_size, data_size;
            reader(&name_size, sizeof(size_t));
            offset += sizeof(size_t);
            reader(&data_size, sizeof(size_t));
            offset += sizeof(size_t);

            std::string name;
            name.resize(name_size);
            reader(name.data(), name_size);
            offset += name_size;
            auto data = new uint8_t[data_size];
            reader(data, data_size);
            offset += data_size;

            std::shared_ptr<uint8_t[]> data_ptr(data);
            binary_set_.Append(name, data_ptr, data_size);
        }
    }

    std::string
    get_index_name(const std::vector<int32_t>& params) {
        std::string params_str = "";
        for (size_t i = 0; i < params.size(); i++) {
            params_str += "_" + std::to_string(params[i]);
        }
        return ann_test_name_ + "_" + std::string(index_type_) + params_str + ".index";
    }

    void
    create_cpu_index(const std::string& index_file_name, const knowhere::Config& conf) {
        printf("[%.3f s] Creating CPU index \"%s\"\n", get_time_diff(), std::string(index_type_).c_str());
        auto& factory = knowhere::VecIndexFactory::GetInstance();
        index_ = factory.CreateVecIndex(index_type_);

        try {
            printf("[%.3f s] Reading index file: %s\n", get_time_diff(), index_file_name.c_str());
            read_index(index_file_name);
        } catch (...) {
            printf("[%.3f s] Building all on %d vectors\n", get_time_diff(), nb_);
            knowhere::DatasetPtr ds_ptr = knowhere::GenDataset(nb_, dim_, xb_);
            index_->BuildAll(ds_ptr, conf);

            printf("[%.3f s] Writing index file: %s\n", get_time_diff(), index_file_name.c_str());
            write_index(index_file_name, conf);
        }
    }

    void
    test_idmap(const knowhere::Config& cfg) {
        auto conf = cfg;

        printf("\n[%0.3f s] %s | %s \n", get_time_diff(), ann_test_name_.c_str(), std::string(index_type_).c_str());
        printf("================================================================================\n");
        for (auto nq : NQs_) {
            knowhere::DatasetPtr ds_ptr = knowhere::GenDataset(nq, dim_, xq_);
            for (auto k : TOPKs_) {
                knowhere::SetMetaTopk(conf, k);
                CALC_TIME_SPAN(auto result = index_->Query(ds_ptr, conf, nullptr));
                auto ids = knowhere::GetDatasetIDs(result);
                float recall = CalcRecall(ids, nq, k);
                printf("  nq = %4d, k = %4d, elapse = %.4fs, R@ = %.4f\n", nq, k, t_diff, recall);
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(),
               std::string(index_type_).c_str());
    }

    void
    test_ivf(const knowhere::Config& cfg) {
        auto conf = cfg;
        auto nlist = knowhere::GetIndexParamNlist(conf);

        printf("\n[%0.3f s] %s | %s | nlist=%ld\n", get_time_diff(), ann_test_name_.c_str(),
               std::string(index_type_).c_str(), nlist);
        printf("================================================================================\n");
        for (auto nprobe : NPROBEs_) {
            knowhere::SetIndexParamNprobe(conf, nprobe);
            for (auto nq : NQs_) {
                knowhere::DatasetPtr ds_ptr = knowhere::GenDataset(nq, dim_, xq_);
                for (auto k : TOPKs_) {
                    knowhere::SetMetaTopk(conf, k);
                    CALC_TIME_SPAN(auto result = index_->Query(ds_ptr, conf, nullptr));
                    auto ids = knowhere::GetDatasetIDs(result);
                    float recall = CalcRecall(ids, nq, k);
                    printf("  nprobe = %4d, nq = %4d, k = %4d, elapse = %.4fs, R@ = %.4f\n", nprobe, nq, k, t_diff,
                           recall);
                }
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(),
               std::string(index_type_).c_str());
    }

    void
    test_hnsw(const knowhere::Config& cfg) {
        auto conf = cfg;
        auto M = knowhere::GetIndexParamHNSWM(conf);
        auto efConstruction = knowhere::GetIndexParamEfConstruction(conf);

        printf("\n[%0.3f s] %s | %s | M=%ld | efConstruction=%ld\n", get_time_diff(), ann_test_name_.c_str(),
               std::string(index_type_).c_str(), M, efConstruction);
        printf("================================================================================\n");
        for (auto ef : EFs_) {
            knowhere::SetIndexParamEf(conf, ef);
            for (auto nq : NQs_) {
                knowhere::DatasetPtr ds_ptr = knowhere::GenDataset(nq, dim_, xq_);
                for (auto k : TOPKs_) {
                    knowhere::SetMetaTopk(conf, k);
                    CALC_TIME_SPAN(auto result = index_->Query(ds_ptr, conf, nullptr));
                    auto ids = knowhere::GetDatasetIDs(result);
                    float recall = CalcRecall(ids, nq, k);
                    printf("  ef = %4d, nq = %4d, k = %4d, elapse = %.4fs, R@ = %.4f\n", ef, nq, k, t_diff, recall);
                }
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(),
               std::string(index_type_).c_str());
    }

    void
    test_annoy(const knowhere::Config& cfg) {
        auto conf = cfg;
        auto n_trees = knowhere::GetIndexParamNtrees(conf);

        printf("\n[%0.3f s] %s | %s | n_trees=%ld \n", get_time_diff(), ann_test_name_.c_str(),
               std::string(index_type_).c_str(), n_trees);
        printf("================================================================================\n");
        for (auto sk : SEARCH_Ks_) {
            knowhere::SetIndexParamSearchK(conf, sk);
            for (auto nq : NQs_) {
                knowhere::DatasetPtr ds_ptr = knowhere::GenDataset(nq, dim_, xq_);
                for (auto k : TOPKs_) {
                    knowhere::SetMetaTopk(conf, k);
                    CALC_TIME_SPAN(auto result = index_->Query(ds_ptr, conf, nullptr));
                    auto ids = knowhere::GetDatasetIDs(result);
                    float recall = CalcRecall(ids, nq, k);
                    printf("  search_k = %4d, nq = %4d, k = %4d, elapse = %.4fs, R@ = %.4f\n", sk, nq, k, t_diff,
                           recall);
                }
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(),
               std::string(index_type_).c_str());
    }

 protected:
    void
    SetUp() override {
        T0_ = elapsed();
        set_ann_test_name("sift-128-euclidean");
        parse_ann_test_name();
        load_hdf5_data<false>();

        assert(metric_str_ == METRIC_IP_STR || metric_str_ == METRIC_L2_STR);
        metric_type_ = (metric_str_ == METRIC_IP_STR) ? knowhere::metric::IP : knowhere::metric::L2;
        knowhere::SetMetaMetricType(cfg_, metric_type_);
        knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::AUTO);
    }

    void
    TearDown() override {
        free_all();
    }

 protected:
    knowhere::MetricType metric_type_;
    knowhere::BinarySet binary_set_;
    knowhere::IndexType index_type_;
    knowhere::VecIndexPtr index_ = nullptr;
    knowhere::Config cfg_;

    const std::vector<int32_t> NQs_ = {10000};
    const std::vector<int32_t> TOPKs_ = {10};

    // IVF index params
    const std::vector<int32_t> NLISTs_ = {1024};
    const std::vector<int32_t> NPROBEs_ = {1, 2, 4, 8, 16, 32, 64, 128, 256};

    // IVFPQ index params
    const std::vector<int32_t> Ms_ = {8, 16, 32};
    const int32_t NBITS_ = 8;

    // HNSW index params
    const std::vector<int32_t> HNSW_Ms_ = {16};
    const std::vector<int32_t> EFCONs_ = {100};
    const std::vector<int32_t> EFs_ = {16, 32, 64, 128, 256};

    // ANNOY index params
    const std::vector<int32_t> N_TREEs_ = {32};
    const std::vector<int32_t> SEARCH_Ks_ = {16, 32, 64, 128, 256};
};

TEST_F(Benchmark_knowhere_float, TEST_IDMAP) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IDMAP;

    knowhere::Config conf = cfg_;
    std::string index_file_name = get_index_name({});
    create_cpu_index(index_file_name, conf);
    index_->Load(binary_set_);
    test_idmap(conf);
}

TEST_F(Benchmark_knowhere_float, TEST_IVF_FLAT_NM) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IVFFLAT;

    knowhere::Config conf = cfg_;
    for (auto nlist : NLISTs_) {
        std::string index_file_name = get_index_name({nlist});
        knowhere::SetIndexParamNlist(conf, nlist);
        create_cpu_index(index_file_name, conf);

        // IVFFLAT_NM should load raw data
        knowhere::BinaryPtr bin = std::make_shared<knowhere::Binary>();
        bin->data = std::shared_ptr<uint8_t[]>((uint8_t*)xb_, [&](uint8_t*) {});
        bin->size = dim_ * nb_ * sizeof(float);
        binary_set_.Append(RAW_DATA, bin);

        index_->Load(binary_set_);
        test_ivf(conf);
    }
}

TEST_F(Benchmark_knowhere_float, TEST_IVF_SQ8) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IVFSQ8;

    knowhere::Config conf = cfg_;
    for (auto nlist : NLISTs_) {
        std::string index_file_name = get_index_name({nlist});
        knowhere::SetIndexParamNlist(conf, nlist);
        create_cpu_index(index_file_name, conf);
        index_->Load(binary_set_);
        test_ivf(conf);
    }
}

TEST_F(Benchmark_knowhere_float, TEST_IVF_PQ) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IVFPQ;

    knowhere::Config conf = cfg_;
    knowhere::SetIndexParamNbits(conf, NBITS_);
    for (auto m : Ms_) {
        knowhere::SetIndexParamM(conf, m);
        for (auto nlist : NLISTs_) {
            std::string index_file_name = get_index_name({nlist, m});
            knowhere::SetIndexParamNlist(conf, nlist);
            create_cpu_index(index_file_name, conf);
            index_->Load(binary_set_);
            test_ivf(conf);
        }
    }
}

TEST_F(Benchmark_knowhere_float, TEST_HNSW) {
    index_type_ = knowhere::IndexEnum::INDEX_HNSW;

    knowhere::Config conf = cfg_;
    for (auto M : HNSW_Ms_) {
        knowhere::SetIndexParamHNSWM(conf, M);
        for (auto efc : EFCONs_) {
            std::string index_file_name = get_index_name({M, efc});
            knowhere::SetIndexParamEfConstruction(conf, efc);
            create_cpu_index(index_file_name, conf);
            index_->Load(binary_set_);
            test_hnsw(conf);
        }
    }
}

TEST_F(Benchmark_knowhere_float, TEST_ANNOY) {
    index_type_ = knowhere::IndexEnum::INDEX_ANNOY;

    knowhere::Config conf = cfg_;
    for (auto n : N_TREEs_) {
        knowhere::SetIndexParamNtrees(conf, n);
        std::string index_file_name = get_index_name({n});
        create_cpu_index(index_file_name, conf);
        index_->Load(binary_set_);
        test_annoy(conf);
    }
}

TEST_F(Benchmark_knowhere_float, TEST_RHNSW_FLAT) {
    index_type_ = knowhere::IndexEnum::INDEX_RHNSWFlat;

    knowhere::Config conf = cfg_;
    for (auto M : HNSW_Ms_) {
        knowhere::SetIndexParamHNSWM(conf, M);
        for (auto efc : EFCONs_) {
            std::string index_file_name = get_index_name({M, efc});
            knowhere::SetIndexParamEfConstruction(conf, efc);
            create_cpu_index(index_file_name, conf);

            // RHNSW index should load raw data
            knowhere::BinaryPtr bin = std::make_shared<knowhere::Binary>();
            bin->data = std::shared_ptr<uint8_t[]>((uint8_t*)xb_, [&](uint8_t*) {});
            bin->size = dim_ * nb_ * sizeof(float);
            binary_set_.Append(RAW_DATA, bin);

            index_->Load(binary_set_);
            test_hnsw(conf);
        }
    }
}

TEST_F(Benchmark_knowhere_float, TEST_RHNSW_SQ) {
    index_type_ = knowhere::IndexEnum::INDEX_RHNSWSQ;

    knowhere::Config conf = cfg_;
    for (auto M : HNSW_Ms_) {
        knowhere::SetIndexParamHNSWM(conf, M);
        for (auto efc : EFCONs_) {
            std::string index_file_name = get_index_name({M, efc});
            knowhere::SetIndexParamEfConstruction(conf, efc);
            create_cpu_index(index_file_name, conf);
            index_->Load(binary_set_);
            test_hnsw(conf);
        }
    }
}

TEST_F(Benchmark_knowhere_float, TEST_RHNSW_PQ) {
    index_type_ = knowhere::IndexEnum::INDEX_RHNSWPQ;

    knowhere::Config conf = cfg_;
    for (auto M : HNSW_Ms_) {
        knowhere::SetIndexParamHNSWM(conf, M);
        for (auto efc : EFCONs_) {
            knowhere::SetIndexParamEfConstruction(conf, efc);
            for (auto m : Ms_) {
                knowhere::SetIndexParamPQM(conf, m);
                std::string index_file_name = get_index_name({M, efc, m});
                create_cpu_index(index_file_name, conf);
                index_->Load(binary_set_);
                test_hnsw(conf);
            }
        }
    }
}