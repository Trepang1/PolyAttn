#ifndef POLYEVAL_CUH
#define POLYEVAL_CUH

#include "fr-tensor.cuh"
#include "polynomial.cuh"

// PolyEval: Replaces tlookup for middle segments
// Uses Horner evaluation of a Chebyshev polynomial to compute exp(-lambda*x)
// on a bounded domain [0, domain_size-1].
// Eliminates field inversion (no finv() calls needed).

class PolyEvalSegment {
public:
    // coeffs: polynomial coefficients (d+1 values in F_p, pre-computed via Chebyshev)
    // domain_size: range of input values (typically b_k = 256)
    // degree: polynomial degree (typically d=9)
    PolyEvalSegment(const vector<Fr_t>& coeffs, uint domain_size, uint degree);
    ~PolyEvalSegment();

    // Compute Y = p(X) for each element of X via Horner
    // Returns: {Y, m} where Y is the output tensor and m is the counts tensor
    pair<FrTensor, FrTensor> compute(const FrTensor& X);

    // Prove the polynomial evaluation
    Fr_t prove(const FrTensor& X, const FrTensor& Y, const FrTensor& m,
        const Fr_t& r_seg, const Fr_t& alpha_seg, const Fr_t& beta_seg,
        const vector<Fr_t>& u_Y, const vector<Fr_t>& v_Y,
        vector<Polynomial>& proof);

    uint domain_size;
    uint degree;
    Fr_t* d_coeffs; // GPU coefficients
};

#endif
