#!/bin/bash
cd /root/autodl-tmp/PolyAttn

# Copy all our experiment files
cp -r /root/autodl-tmp/experiments_7b . 2>/dev/null
cp -r /root/autodl-tmp/experiments_7b_results . 2>/dev/null

# Copy our Python prototypes
cp /root/autodl-tmp/polyeval_prototype.py . 2>/dev/null
cp /root/autodl-tmp/polyattn_validate.py . 2>/dev/null
cp /root/autodl-tmp/polyattn_ppl_test.py . 2>/dev/null
cp /root/autodl-tmp/gen_poly_tables.py . 2>/dev/null
cp /root/autodl-tmp/full_e2e_compare.py . 2>/dev/null
cp /root/autodl-tmp/pipeline_a100.py . 2>/dev/null
cp /root/autodl-tmp/patch_zksoftmax.py . 2>/dev/null

# Remove temp files
rm -f *.log *.tar.gz 2>/dev/null

# Verify
echo "Files copied:"
ls *.py experiments_7b/*.py experiments_7b_results/*.json 2>/dev/null | head -20

# Git add all
git add -A .
echo ""
echo "Committing..."
git commit -m "Add PolyAttn experiments, PPL results, ZK prototypes, CUDA integration" --quiet

echo "Pushing..."
GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" git push

echo ""
echo "Total tracked files:"
git ls-files | wc -l
