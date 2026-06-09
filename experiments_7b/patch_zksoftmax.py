"""Patch zksoftmax.cu to load PolyAttn polynomial tables from files"""
import re

with open('/root/autodl-tmp/zkllm-ccs2024/zksoftmax.cu', 'r') as f:
    content = f.read()

# Add fstream include
if '#include <fstream>' not in content:
    content = '#include <fstream>\n#include <cstdio>\n' + content

# Add static workdir variable before constructor
old_ctor = 'zkSoftmax::zkSoftmax(const vector<uint>& bs, uint L, uint M, unsigned long scaling_factor_in, const vector<double>& thetas, uint m, uint n, uint d, uint E):'
new_ctor = '''static string g_poly_workdir = "";
void set_poly_workdir(const string& wd) { g_poly_workdir = wd; }

zkSoftmax::zkSoftmax(const vector<uint>& bs, uint L, uint M, unsigned long scaling_factor_in, const vector<double>& thetas, uint m, uint n, uint d, uint E):'''
content = content.replace(old_ctor, new_ctor)

# Replace table generation with conditional poly load
old_table = '''        FrTensor mapped_values(bs[i]);
        uint threads_per_block = 256;
        uint blocks_per_grid = (bs[i] + threads_per_block - 1) / threads_per_block;
        zksoftmax_calculate_table<<<blocks_per_grid, threads_per_block>>>(mapped_values.gpu_data, thetas[i - L], scaling_factor_in, d, Bs[i], bs[i]);'''

new_table = '''        FrTensor mapped_values(bs[i]);
        // PolyAttn: try loading pre-computed polynomial table
        string poly_file = g_poly_workdir + "/polyeval_table_" + to_string(i) + ".bin";
        ifstream poly_f(poly_file, ios::binary);
        if (poly_f.good()) {
            poly_f.close();
            uint table_words = bs[i] * 8;  // 8 x uint32 per Fr_t element
            uint* host_table = new uint[table_words];
            FILE* fp = fopen(poly_file.c_str(), "rb");
            size_t read = fread(host_table, sizeof(uint), table_words, fp);
            fclose(fp);
            cudaMemcpy(mapped_values.gpu_data, host_table, table_words * sizeof(uint), cudaMemcpyHostToDevice);
            delete[] host_table;
            printf("[PolyAttn] Loaded polynomial table seg=%d, entries=%d\\n", i, bs[i]);
        } else {
            uint threads_per_block = 256;
            uint blocks_per_grid = (bs[i] + threads_per_block - 1) / threads_per_block;
            zksoftmax_calculate_table<<<blocks_per_grid, threads_per_block>>>(mapped_values.gpu_data, thetas[i - L], scaling_factor_in, d, Bs[i], bs[i]);
        }'''

content = content.replace(old_table, new_table)

# Add workdir parameter to self-attn.cu's constructor call
# The self-attn binary creates zkSoftmax with hardcoded params
# We need to set the workdir before that call
# Let's add it to self-attn.cu instead

with open('/root/autodl-tmp/zkllm-ccs2024/zksoftmax.cu', 'w') as f:
    f.write(content)

print("zksoftmax.cu patched successfully")

# Now patch self-attn.cu to call set_poly_workdir
with open('/root/autodl-tmp/zkllm-ccs2024/self-attn.cu', 'r') as f:
    sa = f.read()

# Add set_poly_workdir call before zkSoftmax construction
old_sm = 'zkSoftmax softmax({1<<8, 1<<20, 1<<20}, 1, 0, 1UL<<32, {1<<18, 1<<22}, seq_len, seq_len, d, 1);'
new_sm = '''zkSoftmax softmax({1<<8, 1<<20, 1<<20}, 1, 0, 1UL<<32, {1<<18, 1<<22}, seq_len, seq_len, d, 1);
    set_poly_workdir(workdir);  // PolyAttn: set table file directory'''
sa = sa.replace(old_sm, new_sm)

with open('/root/autodl-tmp/zkllm-ccs2024/self-attn.cu', 'w') as f:
    f.write(sa)

print("self-attn.cu patched successfully")
