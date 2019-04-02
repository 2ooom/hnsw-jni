#include <iostream>
#include "hnswlib.h"

template<typename dist_t, typename data_t=float>
class Index {
public:
    Index(hnswlib::SpaceInterface<float> *space, const int dim, bool normalize = false) :
            space(space), dim(dim), normalize(normalize) {
        appr_alg = NULL;
        index_inited = false;
    }

    void init_new_index(const size_t maxElements, const size_t M, const size_t efConstruction, const size_t random_seed) {
        if (appr_alg) {
            throw new std::runtime_error("The index is already initiated.");
        }
        cur_l = 0;
        appr_alg = new hnswlib::HierarchicalNSW<dist_t>(space, maxElements, M, efConstruction, random_seed);
        index_inited = true;
    }

    void set_ef(size_t ef) {
        appr_alg->ef_ = ef;
    }

    void saveIndex(const std::string &path_to_index) {
        appr_alg->saveIndex(path_to_index);
    }

    void loadIndex(const std::string &path_to_index, size_t max_elements) {
        if (appr_alg) {
            std::cerr<<"Warning: Calling load_index for an already inited index. Old index is being deallocated.";
            delete appr_alg;
        }
        appr_alg = new hnswlib::HierarchicalNSW<dist_t>(space, path_to_index, false, max_elements);
        cur_l = appr_alg->cur_element_count;
    }

    void normalize_vector(float *data, float *norm_array){
        float norm=0.0f;
        for(int i=0;i<dim;i++)
            norm+=data[i]*data[i];
        norm= 1.0f / (sqrtf(norm) + 1e-30f);
        for(int i=0;i<dim;i++)
            norm_array[i]=data[i]*norm;
    }

    float * get_normalized(float * vector) {
        std::vector<float> norm_array(dim);
        normalize_vector(vector, norm_array.data());
        return norm_array.data();
    }

    void addItem(float * vector) {
        size_t id = cur_l;
        float *vector_data = normalize? get_normalized(vector) : vector;
        appr_alg->addPoint((void *) vector_data, (size_t) id);
        cur_l += 1;
    }

    std::vector<unsigned int> getIdsList() {
        std::vector<unsigned int> ids;

        for(auto kv : appr_alg->label_lookup_) {
            ids.push_back(kv.first);
        }
        return ids;
    }

    void knnQuery(float * vector, hnswlib::labeltype * items, dist_t * distances, size_t k) {
        float *vector_data = normalize? get_normalized(vector) : vector;

        std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->searchKnn(
                (void *) vector_data, k);
        if (result.size() != k)
            throw std::runtime_error(
                    "Cannot return the results in a contigious 2D array. Probably ef or M is to small");
        for (int i = k - 1; i >= 0; i--) {
            auto &result_tuple = result.top();
            distances[k + i] = result_tuple.first;
            items[k + i] = result_tuple.second;
            result.pop();
        }
    }

    hnswlib::SpaceInterface<float> *space;
    int dim;
    bool normalize;
    bool index_inited;
    bool ep_added;
    int num_threads_default;
    hnswlib::labeltype cur_l;
    hnswlib::HierarchicalNSW<dist_t> *appr_alg;

    ~Index() {
        delete space;
        if (appr_alg)
            delete appr_alg;
    }
};

extern "C" {
    long createAngular(int dim) {
        hnswlib::SpaceInterface<float> *space = new hnswlib::InnerProductSpace(dim);
        bool normalize = true;
        return (long)new Index<float>(space, dim, normalize);
    }

    long createEuclidean(int dim) {
        hnswlib::SpaceInterface<float> *space = new hnswlib::L2Space(dim);
        bool normalize=false;
        return (long)new Index<float>(space, dim, normalize);
    }

    long createDotProduct(int dim) {
        hnswlib::SpaceInterface<float> *space = new hnswlib::InnerProductSpace(dim);
        bool normalize=false;
        return (long)new Index<float>(space, dim, normalize);
    }

    void init_new_index(long index, const size_t maxElements, const size_t M, const size_t efConstruction, const size_t random_seed) {
        ((Index<float> *)index)->init_new_index(maxElements, M, efConstruction, random_seed);
    }

    void set_ef(long index, size_t ef) {
        ((Index<float> *)index)->set_ef(ef);
    }

    void saveIndex(long index, const std::string &path_to_index) {
        ((Index<float> *)index)->saveIndex(path_to_index);
    }

    void loadIndex(long index, const std::string &path_to_index, size_t max_elements) {
        ((Index<float> *)index)->loadIndex(path_to_index, max_elements);
    }

    void addItem(long index, float * vector) {
        ((Index<float> *)index)->addItem(vector);
    }

    std::vector<unsigned int> getIdsList(long index) {
        return ((Index<float> *)index)->getIdsList();
    }

    void knnQuery(long index, float * vector, void * items, float * distances, size_t k) {
        ((Index<float> *)index)->knnQuery(vector, (hnswlib::labeltype *) items, distances, k);
    }
}
