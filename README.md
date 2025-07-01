# Project Optimization Report: Accelerating 1.58-bit quantized MLP Inference

This report details the optimization efforts for a Multi-Layer Perceptron (MLP) project, focusing on the significant speedup achieved through **AVX2 vectorization** and the use of **Look-up Tables (LUT)**.

-----
Of course. Here is a more detailed breakdown of the concrete optimization tricks used in this project.

## Advanced Optimization Techniques

The performance gains in this project are not the result of a single change, but a combination of several advanced optimization techniques working in concert. Let's dive into the specifics of each one.

-----

### 1\. Quantization-Aware Training (QAT) and Ternary Representation

The foundation of the optimization is **Quantization-Aware Training (QAT)**. Instead of training a full-precision model and then quantizing it (which often leads to a significant accuracy drop), we simulate the quantization effects *during* the training process.

* **Ternary Weights**: During training, the model's weights are clamped to a `[-1, 1]` range and then quantized to one of three values: **-1, 0, or 1**. This is based on a `TERNARY_THRESHOLD` of `0.001`. The `TernaryQuantizeSTE` function in `train_qat_mlp.py` handles this, using a Straight-Through Estimator (STE) to allow gradients to flow during backpropagation despite the non-differentiable quantization step.

* **Learned Activation Scaling**: Activations are also quantized. Instead of a fixed scale, the model *learns* the optimal scale during training. The `QuantizedTernaryActivation` module calculates the running average of the maximum absolute activation values and uses that to determine the scaling factor. This learned scale (`hidden_activation_scale`) is then exported and used in the C++ inference code.

This QAT process produces a model that is already optimized for low-precision arithmetic, minimizing the accuracy loss while preparing it for the next stages of optimization.

-----

### 2\. AVX2 Vectorization: Parallel Processing Power 🚀

Modern CPUs can perform operations on multiple data points simultaneously using **Single Instruction, Multiple Data (SIMD)** instruction sets. This project uses **Advanced Vector Extensions 2 (AVX2)**, which can process 256 bits of data at once.

* **Enabling AVX2**: The `CMakeLists.txt` file explicitly tells the compiler to use AVX2 instructions and apply high-level optimizations (`-O3`).

  ```cmake
  target_compile_options(myQATModel PRIVATE -mavx2 -O3)
  ```

* **Intrinsic Functions**: In `src/kernels.cpp`, instead of standard C++ loops, we use **AVX2 intrinsics**. These are special functions that map directly to CPU instructions. For example, in the `weights_only_linear_forward` function, we see intrinsics for loading data (`_mm256_loadu_si256`), performing fused multiply-add operations (`_mm256_madd_epi16`), and adding the results (`_mm256_add_epi32`).

  ```cpp
  // Example from src/kernels.cpp
  __m256i mad1 = _mm256_madd_epi16(input_i16_1, weights_i16_1);
  __m256i mad2 = _mm256_madd_epi16(input_i16_2, weights_i16_2);

  accumulated_sum_vec_32 = _mm256_add_epi32(accumulated_sum_vec_32, _mm256_add_epi32(mad1, mad2));
  ```

  This allows us to process 32 bytes (or 16 16-bit integers) in a single block of instructions, providing a massive speedup over processing them one by one.

-----

### 3\. Look-up Table (LUT) and Bit-Slice Multiplication ⚡

This is the most innovative optimization in the project. The core idea is to replace expensive arithmetic (multiplication) with cheap memory lookups.

* **The Problem**: A dot product involves many multiplications. Even with quantization, this is computationally intensive.

* **The LUT Solution**: Since our quantized values are only -1, 0, or 1, the result of any multiplication can only be -1, 0, or 1. We can pre-calculate the result of multiplying packed blocks of these ternary values and store them in a **Look-up Table (LUT)**.

* **Building the LUT**: The `build_bit_slice_lut_5x3` function in `src/utils.cpp` creates a large table (`precomputed_lut`). It iterates through every possible combination of a packed 5-value activation and a packed 5-value weight, computes their dot product, and stores the `int32_t` result in the table. The index into the table is cleverly constructed by combining the packed activation and weight bytes.

* **Bit-Slice GEMM**: The `avx2_bit_slice_gemm_kernel` in `src/kernels.cpp` is the key. Instead of multiplying, it:

  1.  Loads packed activation and weight bytes.
  2.  Uses them to form an index into the `precomputed_lut_ptr`.
  3.  Uses the AVX2 `_mm256_i32gather_epi32` instruction to fetch 8 pre-computed results from the LUT at once.
  4.  Accumulates these results to get the final dot product value.

This "bit-slicing" technique completely avoids runtime multiplication for the main GEMM (General Matrix Multiply) operation, replacing it with highly efficient, parallelized memory lookups.

-----

### 4\. Memory Optimization via Value Packing

To make the LUT approach viable and to minimize memory bandwidth usage, we need to pack our data as tightly as possible.

* **From Ternary to 3-bit**: The ternary values (-1, 0, 1) are first encoded into unsigned 3-bit values (0, 1, 2).
* **5x3 Packing**: Since `3^5 = 243`, which is less than 256, we can pack **five** 3-bit encoded values into a **single byte**. The `pack_five_ternary` function in `src/utils.cpp` implements this by treating the 5 values as digits in a base-3 number system.

This packing scheme reduces the memory footprint of the weights and activations by a factor of **5**, leading to smaller model sizes, less memory usage, and faster data transfer from memory to the CPU.

-----

## Performance Analysis (FashionMNIST)

The initial performance evaluation was conducted on the FashionMNIST dataset. We compared three versions of the MLP: a full-precision floating-point model, a weight-only quantized model, and a fully quantized model accelerated with a Look-up Table.

The results clearly demonstrate the effectiveness of the quantization and LUT-based approach.

\--- Performance and Accuracy Comparison (10000 images, Batch Size: 64) ---
| Metric | All Quantized MLP (LUT) | Wt-Only Quant MLP | Full Prec. Float MLP |
| :--- | :--- | :--- | :--- |
| **Total Time (ms)** | 968.9827 | 1107.6604 | 5666.0382 |
| **Avg. Time / iter(ms)** | 0.0969 | 0.1108 | 0.5666 |
| **Accuracy (%)** | 69.2700 | 77.2700 | 88.1200 |

As shown in the table, the **All Quantized MLP (LUT)** is the fastest implementation, approximately **5.85 times faster** than the **Full Precision Float MLP**. While there is a drop in accuracy, which is a typical trade-off in quantization, the performance gain is substantial.

-----

## Memory Optimization

Quantization also leads to a significant reduction in the memory footprint of the model. By representing weights and activations with fewer bits, we can drastically decrease the model size.

\--- Memory Cost Comparison ---
| Model Type | Memory Cost (KB) |
| :--- | :--- |
| Full Prec. Float MLP| 1213.79 |
| Wt-Only Quant MLP | 304.41 |
| All Quantized (LUT) | 61.91 |

The **All Quantized (LUT)** model is approximately **19.6 times smaller** than the full precision model. This makes it highly suitable for deployment on resource-constrained devices. It's also worth noting that the memory cost of the LUT itself is negligible, especially as the matrix sizes grow.

-----

## Scalability and Speed Evaluation

To test the scalability of our optimizations, we evaluated the models with larger dimensions (Input=3200, Hidden=3200, Output=10).

\--- Speed Evaluation Results ---
Number of iterations: 1000

* **Standard Float MLP**: 17122.8 ms
* **Weights-Only Quantized MLP**: 2278.05 ms
* **LUT-based Full Quantized MLP**: 1517.34 ms

The results from `speed_test.cpp` confirm the trend seen with the smaller model. The **LUT-based Full Quantized MLP** remains the top performer, being about **11.3 times faster** than the standard floating-point implementation and **1.5 times faster** than the weights-only quantized version. This demonstrates that the benefits of the LUT-based approach scale effectively with model size.

-----

## Core Optimization Techniques

The remarkable speedup is primarily due to two key optimization techniques:

### 1\. AVX2 Vectorization 🚀

We leveraged **Advanced Vector Extensions 2 (AVX2)** to perform parallel computations. Instead of processing single data points, AVX2 allows us to handle multiple data points in a single instruction. This is particularly effective in the context of neural networks, where operations like dot products are inherently parallelizable.

The `CMakeLists.txt` file is configured to enable AVX2 support during compilation:

```cmake
target_compile_options(myQATModel PRIVATE -mavx2 -O3)
```

The core of the AVX2 implementation can be found in `src/kernels.cpp`. For instance, the `weights_only_linear_forward` function uses AVX2 intrinsics (`_mm256_...`) to accelerate the matrix multiplication process. The `avx2_bit_slice_gemm_kernel` is another key function that leverages AVX2 for the LUT-based approach.

### 2\. Look-up Table (LUT) Acceleration ⚡

The most significant optimization comes from the use of a **Look-up Table (LUT)**. In the fully quantized model, both weights and activations are represented as ternary values (-1, 0, 1). We pre-compute the results of all possible multiplication combinations between these ternary values and store them in a LUT.

The LUT is generated by the `build_bit_slice_lut_5x3` function in `src/utils.cpp`. During inference, instead of performing multiplications, the model can simply look up the result from the table. This transforms the computationally expensive multiplication operations into much faster memory access operations. The `lut_linear_forward` function in `src/kernels.cpp` orchestrates this process.

-----

## Future Work: Larger Look-Up Tables

There is potential for even greater acceleration. If the matrix sizes are known and fixed, we could pre-compute a **larger, more specialized Look-up Table**.

The current implementation uses a LUT for 5x3-bit packed values. A larger LUT could, for example, pre-compute the results for bigger blocks of the matrix. This would further reduce the number of operations at runtime. The trade-off would be a larger memory footprint for the LUT itself, but for specific hardware and fixed model architectures, this could be a viable strategy for pushing the performance boundaries even further.

-----

## How to Run the Project

Here is a step-by-step guide to reproduce the results:

### Step 1: Environment Setup

Ensure you have a C++ compiler (like g++), CMake, and Python with PyTorch installed.

### Step 2: Train the Models

First, you need to train both the full-precision and the quantization-aware models.

* **Train the full-precision MLP**:

  ```bash
  python train_full_mlp.py
  ```

  This will generate `small_mlp_float.pth`.

* **Train the quantization-aware MLP**:

  ```bash
  python train_qat_mlp.py
  ```

  This script trains the model with ternary quantization and saves it as `small_mlp_ternary_act_qat.pth`.

### Step 3: Convert Models to Binary Format

The C++ application uses a custom binary format for the models.

* **Convert the full-precision model**:

  ```bash
  python convert_full.py
  ```

  This will create `mlp_model_float.bin` from `small_mlp_float.pth`.

* **Convert the quantized model**:

  ```bash
  python convert_qat.py
  ```

  This will create `mlp_model_aq.bin` from `small_mlp_ternary_act_qat.pth`.

### Step 4: Generate Test Data

The C++ application requires the FashionMNIST test data in a specific binary format.

* **Generate the test data**:
  ```bash
  python save_all_images.py
  ```
  This will create `test_images_padded_f32.bin` and `test_labels.bin`.

### Step 5: Build and Run the C++ Application

Now you can build the C++ project using CMake.

* **Create a build directory**:

  ```bash
  mkdir build
  cd build
  ```

* **Run CMake and build the project**:

  ```bash
  cmake ..
  make
  ```

  This will create two executables in the `build/bin` directory: `myQATModel` and `speed_test`.
* **Move the models and test data&labels into build/bin**
* **Run the performance comparison**:

  ```bash
  ./bin/myQATModel
  ```

  This will run the comparison on the FashionMNIST dataset and print the performance and memory usage tables.

* **Run the speed test for larger matrices**:

  ```bash
  ./bin/speed_test 3200 3200 10
  ```

  This will run the speed evaluation with the specified dimensions.