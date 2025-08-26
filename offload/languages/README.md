# LLVM Language Offloading

Compile existing HIP or CUDA sources so that the entire program uses LLVM offloading. Kernel launches are not supported yet.

## Status

* ✅ `malloc` / `free` via `hipMalloc`/`hipFree` and `cudaMalloc`/`cudaFree`
* ✅ `memcpy` via `hipMemcpy` and `cudaMemcpy`
* ⏳ Kernel launches  not implemented yet

---

## HIP Example

### Code (`hip_no_kernel.hip`)

```cpp
#include <stdio.h>

// Currently needed for LLVM language offload
#include <offload/hip/hip_runtime.h>

int main() {
    int *A;
    int R = 5, P = 7;

    hipMalloc((void**)&A, sizeof(int));
    hipMemcpy(A, &P, sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(&R, A, sizeof(int), hipMemcpyDeviceToHost);

    printf("Hello, World from the CPU: %i!\n", R);

    hipFree(A);

    return 0;
}

```

### Compile

```bash
clang++ hip_no_kernel.hip \
  -foffload-via-llvm \
  -o hip_no_kernel
```

**Run:** `./hip_no_kernel`

---

## CUDA Example

### Code (`cuda_no_kernel.cu`)

```cpp
#include <stdio.h>

// Currently needed for LLVM language offload
#include <offload/hip/hip_runtime.h>

int main() {
    int *A;
    int R = 5, P = 7;

    cudaMalloc(&A, 4);
    cudaMemcpy(A, &P, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(&R, A, 4, cudaMemcpyDeviceToHost);

    printf("Hello, World from the CPU: %i!\n", R);

    cudaFree(A);

    return 0;
}
```

### Compile

```bash
clang++ cuda_no_kernel.cu \
  -foffload-via-llvm \
  -o cuda_no_kernel
```

**Run:** `./cuda_no_kernel`
