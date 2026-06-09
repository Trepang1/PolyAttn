#!/bin/bash
cd /root/autodl-tmp/zkllm-ccs2024
rm -f *.o main self-attn ffn rmsnorm skip-connection ppgen commit-param

NVCC=/root/miniconda3/bin/nvcc
FLAGS="-arch=sm_120 -std=c++17 -I/root/miniconda3/include"

echo "Compiling object files..."
for src in bls12-381.cu ioutils.cu commitment.cu fr-tensor.cu g1-tensor.cu proof.cu zkrelu.cu zkfc.cu tlookup.cu polynomial.cu zksoftmax.cu rescaling.cu polyeval.cu; do
    obj=${src%.cu}.o
    echo -n "  $src ... "
    if $NVCC $FLAGS -dc -dlto $src -o $obj 2>/tmp/nvcc_err_$$; then
        echo "OK"
        rm -f /tmp/nvcc_err_$$
    else
        echo "FAILED"
        cat /tmp/nvcc_err_$$
        exit 1
    fi
done

echo ""
echo "All objects compiled. Linking targets (with PolyEval)..."
$NVCC $FLAGS -L/root/miniconda3/lib self-attn.cu *.o -o self-attn -dlto 2>&1 && echo self-attn_OK || echo self-attn_FAILED
$NVCC $FLAGS -L/root/miniconda3/lib ffn.cu *.o -o ffn -dlto 2>&1 && echo ffn_OK || echo ffn_FAILED

echo ""
echo "Build complete. Targets:"
ls -la self-attn ffn rmsnorm 2>/dev/null
