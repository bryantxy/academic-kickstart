---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Anatomy of a High-Performance Convolution"
subtitle: ""
summary: ""
authors: [""]
tags: []
categories: []
date: 2019-08-01T21:57:19+05:30
lastmod: 2019-08-01T21:57:19+05:30
featured: false
draft: true

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---
# Naive Convolutions
Before we go around optimizing things, let's first get a handle on our baseline. Here's a naive numpy/for-loop convolution you'd write in Deep Learning 101:
```python
'''
Convolve `input` with `kernel` to generate `output`
    input.shape = [input_channels, input_height, input_width]
    kernel.shape = [num_filters, input_channels, kernel_height, kernel_width]
    output.shape = [num_filters, output_height, output_width]
'''
for filter in 0..num_filters
    for channel in 0..input_channels
        for out_h in 0..output_height
            for out_w in 0..output_width
                for k_h in 0..kernel_height
                    for k_w in 0..kernel_width
                        output[filter, channel, out_h, out_h] += 
                            kernel[filter, channel, k_h, k_w] * 
                            input[channel, out_h + k_h, out_w + k_w]
```
Yikes. That's 6 nested `for` loops for one conv (7 if you iterate over batches of multiple inputs). And we're not yet looking at stride, dilation, or any other parameters. If I plug in here the sizes for, say the first layer of MobileNet, and run this in plain C, it takes a whopping *22 seconds*! With the most aggressive compiler optimizations like `-O3` or `-Ofast`, it reduces to 2.2 seconds. But that's still terribly slow for just the first layer.

What if I run the same layer using, say, Caffe? It took just 18ms on the same PC. That's over a 100x speedup! The entire network itself runs in less than half a second on my CPU.

It's no surprise that modern deep learning libraries have very carefuly designed implementations. But just what is the black magic with which they're able to improve performance by 100x? Can you bring your own humble convolution to match the same performance?

# Some Prerequisites
## FLOP/s and throughput
Our metric for "performance" or speed will be the throughput, measured in the number of **F**loating **P**oint **O**perations per **S**econd, or FLOP/s. A heavier operation will perform more floating-point operations and hence take more seconds, so FLOP/s rate allows a more consistent way to compare performance.

We can also use this to get an idea of how close we are to the peak performance of the CPU. On my PC's Intel i7:

* there are 2 phsyical cores
* each core has a frequency of 2.5 GHz, or $2.5\times10^9$ CPU cycles per second
* in each cycle, it can process 32 FLOPs (using AVX & FMA, more on this in a bit)
  
This gives a peak performance of $2\times2.5\times10^9\frac{cycles}{second}\times32 \frac{FLOPs}{cycle}=160$ GFLOP/s. This is the theoretical peak of my CPU. Realistically, fetching the data from memory takes time so we rarely achieve this peak.

The number of math operations relative to the memory accesses -- or *arithmetic instensity* or ops:bytes ratio -- can serve as a rough proxy for how significant this memory bottleneck is. For operations like ReLU, Pooling, etc. this is 0.5-2 FLOPs/byte, i.e. they need to fetch a lot of memory and do relatively simpler arithmetic, and our hence limited by memory. On the other hand, operations like convolution, matrix multiplication, etc. perform 100-1000 FLOPs per byte and are hence compute limited. In other words, it makes sense to focus on 

## Storage orders and Row Major
While we logically view matrices/images/tensors as muti-dimensional, they're physically stored in a linear, one-dimensional computer memory. We have to define a convention which dictates how to unwrap these multiple dimensions to a linear storage, and vice versa.

Most modern DL libraries use a *row-major* storage order. This means that consecutive elements of the same row are stored next to each other. More generally with multiple dimensions, row-major means the first dimension changes the slowest as you scan the memory linearly.

What about the ordering of the dimensions themselves? Usually for 4-dimensional tensors like in CNNs, you'll hear of storage orders like NCHW, NHWC, etc. I'll assume NCHW throughout this post -- if I have N blocks of C channels of HxW images, then all images with the same N are contigous; within that block all pixels of the same channel C are contigous, and so on.

[storage order image]

As you'll see, one of our biggest concerns throughout the discussion will be how we're accessing data we're operating on, and how that relates to the the way it's stored. 

## Memory heirarchy

## Halide

# From convolution to GEMM
The naive convolution that we discussed above is slow already, and a more realistic implementation will only be further complicated by parameters like stride, dilation, padding, etc. As you'll see, extracting maximum performance out of the computer will require many tricks, exploiting fine-tuning at multiple levels and knowledge of the computer architecture at hand. In other words, this is going to be a formidable task if we're hoping to address all of the complexities.

Can we instead transform this into a problem that's easier to solve?

Convolution is, after all, a dot-product of the filter with input patches. If we layout the filter in a 2-D matrix and the input patches in another, then the multiplying these 2 matrices would give the compute the same dot product. And matrix multiplication -- unlike CNNs -- has been heavily studied and optimized over several decades, being a critical problem in many domains. 


The above laying out of the image patches into a matrix is called _**im2col**_, for image to column. We rearrange the image into columns of a matrix, so that each column corresponds to one patch where the conv filter is applied. This is what it looks like.

[im2col]

Now since there is usually some overlap between different image patches in a conv, there is actually some memory duplication going on here. The time taken to generate this im2col buffer and the inflated memory, will have to be offset by the speedup we achieve via the matrix multiplication.

Note that the matrix product gives us the output conv result directly. You just have to read the same buffer a 3D tensor instead of a 2D matrix -- there is no need for an extra "conversion" back into an image tensor.

{{% alert note %}}
Interestingly, this im2col approach actually originated from in Caffe, and some sneaky jokes on "temporary" solutions.
{{% /alert %}}

# Speeding up the GEMM
With im2col, we have now transformed the convolution operation into a matrix multiplication. We can now plug in a high-performance linear algebra library, like OpenBLAS, to take care of performing this matmul, riding on the back of decades of optimizations & careful fine-tuning.

We'll need some pretty serious speedups if we're going to justify the extra work and memory resulting from the im2col transform, so let's see how these libraries might be achieving that. This also gives a good intro to what are some general approaches when optimizing at the system level

## Naive
As before, first let's time the basic, textbook matrix multiplication:

```cpp
for i in 0..M:
    for j in 0..N:
        for k in 0..K:
            C[i, j] += A[i, k] * B[k, j]
```

The inner most line does 2 floating point ops (multiply & add) and is performed $M*N*K$ times, so the number of FLOPs for this GEMM is $2MNK$.

How does this perform?
[naive performance]

We barely reach 10% of peak performance! As you'll see, the recurring theme will be that it's not enough to just *process* the data fast if we can't *get* the data fast. Notice how the performance gradually dips for larger sizes. If you ran this on an older, less capable computer, you'll even be able to identify a specific point when the matrices become too big to fit in the cache and the throughput suddenly drops.

[mbp performance]

## Caching
Every time we fetch data from the main memory, the CPU automatically loads it and its neighboring memory into the cache, hoping to utilize locality of reference.

The first thing that you should notice here is the pattern in which we're accessing our data. We're row-wise on $A$ and column-wise on $B$. Their storage is also row-major, so once we find `A[i, k]`, the next element in the row, `A[i, k+1]` is already cached. Cool. But see what happens for $B$ :

 - the next element of the column isn't present in the cache -- we get a cache miss and the CPU stalls while the data is fetched
 - once fetched, the cache also gets filled with other elements in the same row of $B$. We won't actually use them, so they'll be evicted soon. A few iterations later when they're actually needed, we'll be working to fetch them again. We're **polluting** the cache with values we don't need.
[cache lines for columns]


We need to rework our loops to exploit this caching ability instead. If data is being read, we might as well make use of it. This brings us to the first change we'll make: **loop reordering**.

Let's reorder the loops from `i,j,k` to `i,k,j`:
```cpp
for i in 0..M:
    for k in 0..K:
        for j in 0..N:
```
Our answer is still correct because the order of multiplications/additions doesn't matter. The traversal order will now look like this
[traversal order]

This simple change of just reordering the loops gives a considerable speedup:
[loop order speedup]

## Tiling
While the loop reordering helps to an extent, there's one more issue it doesn't solve. 

We were looping over all of $B$ for each row of $A$. With each iteration over B, we'll load some new columns and evict some older ones from the cache. When we get to the next row of A, we start all over again from the first columns. We're repeatedly adding and removing the same data from the cache, or **thrashing** it.

[cache thrashing]
 
The thrashing happened because all our data couldn't fit in the cache. If only we were working with smaller matrices, they could happily live together without getting evicted repeatedly. Thankfully for us, we can break down matrix multiplication over submatrices. To compute a small $r\times c$ tile of $C$, we only need $r$ rows of $A$ and $c$ columns of $B$. Let's break $C$ into tiles of 6x16.

```cpp
Halide::Var x, y;
C(x, y) += A(k, y) * B(x, k);

C.tile(x, y, xo, yo, xi, yi, 6, 16)

/*
in pseudocode:
for xo in 0..N/16:
    for yo in 0..M/6:
        for yi in 6:
            for xi in 0..16:
                for k in 0..K:
                    C(...) = ...
*/
```

We've broken the x,y dimensions into an outer `xo,yo` and inner `xi,yi`. We'll spend our efforts optimizing a *micro-kernel* for the smaller 6x16 block, and run that micro-kernel over all the blocks iterated by `xo,yo`.

## Vectorization & FMA
Almost all modern CPUs come with a feature called **SIMD**, or **S**ingle **I**nstruction **M**ultiple **D**ata. As the name suggests, SIMD can be used to do the same operation/instruction (like add, multiply, etc.) on multiple values simultaneously, in the same CPU cycle. If we can run SIMD instructions on say 4 data points at a time, that's a 4x speedup straightaway. So when we calculated the peak speed of the processor, we _sort of_ cheated and were instead referring to this vectorized performance. This is of great use for data like vectors, where we have to apply the same instruction to every vector element. But we still have to design our kernel to exploit this properly.

[SIMD diagram]

On Intel CPUs, we can use SIMD for processing up to 8 floating-point numbers in a single instruction. Compiler optimizations will often be able to identify vectorization opportunities on their own, but we'll take things in our own hands here.

//FMA

```cpp
C.tile(x, y, xo, yo, xi, yi, 6, 16)
 .reorder(xi, yi, k, xo, yo)
 .vectorize(xi, 8)

/*
in pseudocode:
for xo in 0..N/16:
    for yo in 0..M/6:
        for k in 0..K:
            for yi in 6:
                for vectorized xi in 0..16:
                    C(...) = ...
*/
```

## Threading
Till now we've only been using one CPU core.

All modern CPUs have multiple cores available, and each core can physically execute multiple instructions at the same time. A program can divide itself into multiple threads, and each thread can run on a separate core.

```cpp
C.tile(x, y, xo, yo, xi, yi, 6, 16)
 .reorder(xi, yi, k, xo, yo)
 .vectorize(xi, 8)
 .parallel(yo)
```
You might notice that the performance actually drops for very small sizes, because with small workloads, the threads spend less time working and more time synchronizing with each other. There are a lot of other such issues with respect to threading which could warrant another deep dive on its own.

## Unrolling
```cpp
C.tile(x, y, xo, yo, xi, yi, 6, 16)
 .reorder(xi, yi, k, xo, yo)
 .vectorize(xi, 8)
 .unroll(xi)
 .unroll(yi)
```

# Putting it together
im2col + gemm