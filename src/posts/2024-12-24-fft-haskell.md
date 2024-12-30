---
author: "Julien"
desc: "FFT in Haskell and Futhark"
keywords: "haskell, blog, fft, algorithms, numerical"
lang: "en"
title: "FFT in Haskell and Futhark"
updated: "2024-12-24 11:11"
mathjax: true
---

The Fourier transform is one of the fundamental tools in analysis. 
From the perspective of approximation theory, it gives us one of the orthogonal function bases, the natural basis for periodic functions.
To use this numerically, as with any basis, we must sample from the function to be approximated, and periodic functions have a wonderful property that the optimal points to sample are uniformly spaced on the interval.
This is very different from polynomial bases like Chebyshev or Legendre polynomials, where choosing the points is somewhat involved.
For Fourier, choosing to sample at \\(N\\) points leads to the Discrete Fourier Transform (DFT), a cornerstone of signal processing.

Let \\(x_n\\) be our signal sampled at \\(N\\) points, i.e. \\(x\\) is a sequence of \\(N\\) real or complex numbers.
Let \\(\omega = \omega_N = e^{-2\pi i / N}\\) be the \\(N^\text{th}\\) root of unity, the powers of which are sometimes called "twiddle" factors in this context.
We can then define the DFT element-wise as follows.

\\[ y_t =\sum_{n=0}^{N-1} x_{n} \cdot \omega^{t n} \\]

We see that each element is the dot product of \\(x\\) with a vector of the twiddle factors, so we can represent this as a matrix multiplication \\(y = W_N x\\) where

\\[ 
W_N = \begin{bmatrix} 
\omega^0_N & \omega^0_N & \omega^0_N & \cdots & \omega^0_N \\
\omega^0_N & \omega_N^1 & \omega_N^2 & \cdots & \omega_N^{N-1} \\
\omega^0_N & \omega_N^2 & \omega_N^4 & \cdots & \omega_N^{2(N-1)} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
\omega^0_N & \omega_N^{N-1} & \omega_N^{2(N-1)} & \cdots & \omega_N^{(N-1)(N-1)}
\end{bmatrix}
\\]

We will also denote this as \\(y = F(x)\\). 
The complexity of matrix-vector multiplication is trivially \\(O(n^2)\\), as we must access every element of the matrix. 
In the 1960s, during the great boom of numerical research, James Cooley and John Tukey published their paper exploiting the structure of the matrix for a log-linear solution, ushering in the age of real time digital signal processing.
This family of algorithms is called the Fast Fourier Transform (FFT).

The key insight of the Cooley-Tukey FFT is that one can split the signal in half and recursively compute the FFT.
This is one of the early examples of divide and conquer algorithms, along with merge sort, which shares a similar \\(n \log n\\) time complexity.
For simplicity we assume \\(N\\) is a power of two, so that we can divide in half until we get to a single element, though this is not necessary with real FFT implementations.
Recursive algorithms are often most naturally expressed in functional languages, so we derive a recursive form to implement in Haskell.

First we identify the base case, which is simply the identity \\(y = \omega^0 x = x\\).
For \\(N = 2, N=4\\) it is instructive to do examples out by hand in full using the matrix multiplication \\(W_N x\\).
For \\(N = 2\\) we get the following.

\\[ \begin{align}  y_0 &= x_0 + x_1 \\ y_1 &= x_0 - x_1 \end{align} \\]

When drawn out as a data flow diagram, as you would see in more hardware-adjacent expositions, this forms a cross-over, leading to the name [butterfly](https://en.wikipedia.org/wiki/Butterfly_diagram) for the combining stage of the FFT.

The trick to the recursion is that splitting \\(x\\) into even \\( x^e = \{x_0, x_2, ...\}\\) and odd \\( x^o = \{x_1, x_3, ...\}\\) components. 
This can be seen by rewriting the \\(N=4\\) case out by hand, I will leave out the derivation here, but the result should look like the following.

\\[ 
u = F(x^e) = W_2 \begin{bmatrix} x_0 \\ x_2  \end{bmatrix}, \quad
v = F(x^o) = W_2 \begin{bmatrix} x_1 \\ x_3  \end{bmatrix}
\\]
We then combine the two sub-problems.
\\[ 
y = F(x) = \begin{bmatrix}
u_0 + \omega^0 v_0 \\
u_1 + \omega^1 v_1 \\
u_0 + \omega^2 v_0 \\
u_1 + \omega^3 v_1
\end{bmatrix} = \begin{bmatrix}
u_0 + \omega^0 v_0 \\
u_1 + \omega^1 v_1 \\
u_0 - \omega^0 v_0 \\
u_1 - \omega^1 v_1
\end{bmatrix}
\\]
This is not entirely intuitive and I encourage you to look in an introductory numerical analysis textbook if you would like to be guided through the derivation.
Note that the last equality is just using \\(\omega_N^{N/2} = -1\\) to simplify, this is very helpful computationally, as the bottom half and top half of the vector are now much more similar.
From this we have the motivation for the recursive definition we will implement.
Let \\(u = F(x^e), v = F(x^o)\\) and \\(T = [\omega^0, ..., \omega^{N/2-1}]\\) be a vector of twiddle factors, with \\(\odot\\) being element-wise "broadcasting" multiplication.
\\[
y = \begin{bmatrix}
u + T \odot v \\
u - T \odot v
\end{bmatrix}
\\]

A minimal Haskell implementation of this recursive form is quite elegant.


```haskell
split :: [a] -> ([a], [a])
split [] = ([], [])
split [_] = error "input size must be power of two"
split (x:y:xs) =
  let (es, os) = split xs
  in (x:es, y:os)

mergeRadix2 :: [Complex Double] -> [Complex Double] -> Int -> [Complex Double]
mergeRadix2 u v n = (++) (zipWith (+) u q) (zipWith (-) u q)
  where q = zipWith (*) v w
        n2 = length u - 1
        w = [exp (0 :+ (-2 * pi * fromIntegral k / fromIntegral n )) | k <- [0..n2]]

fft :: [Complex Double] -> [Complex Double]
fft [] = []
fft [z] = [z]
fft zs = mergeRadix2 (fft evens) (fft odds) (length zs)
  where (evens, odds) = split zs
```

One might immediately ask about performance, and yes, this implementation is meant only to be instructive, but explicitly recursive implementations can be competitive.
The first place to look is FFTW, the state of the art software FFT library, which takes a "bag of algorithms + planner" approach.
It is implemented with OCaml for code generation with many passes of optimization to create a portable C library, and many of the variants are recursive.

The obvious suspects in a numerical optimization such as this are
- Avoiding memory reallocation and optimizing cache locality (with algorithm variants such as Stockham)
- Using lookup tables or otherwise avoiding trigonometric calculation 

I wanted to try Futhark, the pure functional array based language implemented in Haskell that compiles to C or Cuda/OpenCL, and thought this algorithm would be a good fit.
There is a Stockham variant in the Futhark packages for reference, but I implemented Cooley-Tukey Radix-2. 
Unfortunately Futhark does not support explicit recursion, and it is not clear (to me at least) if it ever will.
My understanding is that it may be possible in the future, though there are fundamental difficulties, as the stack cannot be used willy-nilly on a GPU, so any recursion would be limited in nature, and currently you just have to unroll it into a loop manually.
This means we cannot implement a recursive FFT, but must do the more complicated iterative approach.

I attempted to use Claude for this, to see how it would do with a relative obscure programming language, surprisingly it mostly worked, though it consistently would get indexing wrong and mostly would not use the array combinators correctly.
The main points of the iterative approach are that successive applications of the even/odd splits can be viewed as a rearrangement by "bit reversal permutation" and that we must do much tedious indexing to keep track of the arithmetic combinations, these are the "butterflies" previously mentioned.
Not going into depth, here is my implementation.

```haskell
def twiddle (k: i64) (n: i64): complex =
let angle = -2.0 * f64.pi * f64.i64 k / f64.i64 n
in (f64.cos angle, f64.sin angle)

def bit_reversal [n] 't (input: [n]t): [n]t =
  let bits = i64.f64 (f64.log2 (f64.i64 n))
  let indices = map (\i ->
    let rev = loop rev = 0 for j < bits do
      (rev << 1) | ((i >> j) & 1)
    in rev
  ) (iota n)
  in spread n (input[0]) indices input

-- Type to hold butterfly operation parameters
type butterfly_params = {
  upper_idx: i64,    -- Index of upper butterfly input
  lower_idx: i64,    -- Index of lower butterfly input
  twiddle: complex   -- Twiddle factor for this butterfly
}

-- Calculate butterfly parameters for a given stage
def get_butterfly_params (stage: i64) (n: i64) (i: i64): butterfly_params =
  let butterfly_size = 1 << (stage + 1)        -- Size of entire butterfly
  let half_size = butterfly_size >> 1          -- Size of half butterfly
  let group = i / butterfly_size               -- Which group of butterflies
  let k = i % half_size                        -- Position within half
  let group_start = group * butterfly_size     -- Start index of this group
  let twiddle_idx = k * (n / butterfly_size)   -- Index for twiddle factor
  in {
    upper_idx = group_start + k,
    lower_idx = group_start + k + half_size,
    twiddle = twiddle twiddle_idx n
  }

-- Perform single butterfly operation
def butterfly_op (data: []complex) (p: butterfly_params) (is_upper: bool): complex =
  if is_upper
  then complex_add data[p.upper_idx]
                  (complex_mul data[p.lower_idx] p.twiddle)
  else complex_sub data[p.upper_idx]
                  (complex_mul data[p.lower_idx] p.twiddle)

-- Main FFT function
def fft [n] (input: [n]complex): [n]complex =
  let bits = i64.f64 (f64.log2 (f64.i64 n))
  -- This method can only handle arrays of length 2^n
  in assert (n == 1 << bits) (
    -- First apply bit reversal permutation
    let reordered = bit_reversal input

    -- Perform log2(n) stages of butterfly operations
    in loop data = reordered for stage < bits do
      -- For each stage, compute butterfly parameters and perform operations
      let butterfly_size = 1 << (stage + 1)
      let half_size = butterfly_size >> 1
      let params = map (get_butterfly_params stage n) (iota n)
      in map2 (\p i ->
        let is_upper = (i % butterfly_size) < half_size
        in butterfly_op data p is_upper
      ) params (iota n)
  )
```

This is not particularly optimized. Futhark allows for fused memory operations and has a semantics for tracking when it is safe to overwrite memory while remaining pure, I did not use this here. I did make sure to use `spread` and `map2` array combinators when traversing, which theoretically should allow for some automatic parallelism, though I did not test this, as I don't have CUDA running on my laptop. 

Futhark is slowly emerging from being an academic project into a serious tool, and the ecosystem is still in its infancy.
I wanted to try implementing some of my research in eigensolvers, but the linear algebra module is at the level of undergraduate research project, and does not appear to support complex matrices at the moment.
Personally, I probably will not use it further at the moment, but it is very much the direction I would like numerical algorithms to go, with functional DSLs (or full languages) that compile to highly portable, highly optimized code.
