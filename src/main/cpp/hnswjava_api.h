#include "hnswindex.h"
#include "hnswlib.h"

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

    void destroyIndex(long index) {
        delete ((Index<float> *)index);
    }

    void initNewIndex(long index, const size_t maxElements, const size_t M, const size_t efConstruction, const size_t random_seed) {
        ((Index<float> *)index)->initNewIndex(maxElements, M, efConstruction, random_seed);
    }

    void setEf(long index, size_t ef) {
        ((Index<float> *)index)->appr_alg->ef_ = ef;
    }

    void saveIndex(long index, const std::string &path_to_index) {
        ((Index<float> *)index)->saveIndex(path_to_index);
    }

    void loadIndex(long index, const std::string &path_to_index) {
        ((Index<float> *)index)->loadIndex(path_to_index);
    }

    void addItem(long index, float * vector, size_t id) {
        ((Index<float> *)index)->addItem(vector, id);
    }

    std::vector<float> getItem(long index, size_t id) {
        return ((Index<float> *)index)->appr_alg->template getDataByLabel<float>(id);
    }

    std::vector<size_t> getIdsList(long index) {
        return ((Index<float> *)index)->getIdsList();
    }

    size_t getNbItems(long index) {
        return ((Index<float> *)index)->appr_alg->cur_element_count;
    }

    size_t knnQuery(long index, float * vector, size_t * items, float * distances, size_t k) {
        return ((Index<float> *)index)->knnQuery(vector, items, distances, k);
    }
}
