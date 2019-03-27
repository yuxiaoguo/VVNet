#ifndef COMMON_UTILS_H_
#define COMMON_UTILS_H_

#include <tensorflow/core/framework/tensor.h>

namespace tensorflow {

template <typename T>
const T* SliceTensorPtr(const Tensor* tensor, int64 batch_idx)
{
    return tensor->Slice(batch_idx, batch_idx + 1).unaligned_flat<T>().data();
}

template <typename T>
T* SliceTensorPtr(Tensor* tensor, int64 batch_idx)
{
    return tensor->Slice(batch_idx, batch_idx + 1).unaligned_flat<T>().data();
}

}

#endif