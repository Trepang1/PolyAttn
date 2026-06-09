#!/bin/bash
cd /root/autodl-tmp/PolyAttn

# Remove nested git repo
rm -rf zkllm-ccs2024/.git 2>/dev/null
find . -name ".git" -type d -mindepth 2 -exec rm -rf {} \; 2>/dev/null

# Remove submodule
git rm --cached zkllm-ccs2024 2>/dev/null

# Add everything
git add -A .

# Commit
git commit -m "PolyAttn: full codebase - zkLLM source, experiments, PolyAttn CUDA, results" --quiet 2>&1

# Push
GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" git push 2>&1

echo "---"
git ls-files | wc -l
echo "total files pushed"
