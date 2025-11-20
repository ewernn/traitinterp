#!/bin/bash
# Batch refactor all paths to new structure:
# experiments/{exp}/extraction/{category}/{trait}/
# experiments/{exp}/validation/ (for cross-dist results)

set -e

echo "🔧 Refactoring paths to new structure..."
echo ""

# Analysis/validation scripts - update cross-dist paths
echo "📝 Updating analysis/validation scripts..."

# analysis/cross_distribution_scanner.py
sed -i '' 's|results/cross_distribution_analysis|experiments/{experiment_name}/validation|g' analysis/cross_distribution_scanner.py
sed -i '' 's|behavioral/\|cognitive/\|stylistic/\|alignment/|extraction/behavioral/|extraction/cognitive/|extraction/stylistic/|extraction/alignment/|g' analysis/cross_distribution_scanner.py

# scripts/run_cross_distribution.py
sed -i '' 's|results/cross_distribution_analysis|experiments/{exp}/validation|g' scripts/run_cross_distribution.py

# scripts/run_extraction_scores.py
sed -i '' 's|results/cross_distribution_analysis|experiments/{exp}/validation|g' scripts/run_extraction_scores.py

echo "  ✓ Updated 3 analysis scripts"

# Visualization files
echo "📝 Updating visualization files..."

# visualization/serve.py
sed -i '' 's|results/cross_distribution_analysis|validation|g' visualization/serve.py

# visualization/core/data-loader.js
sed -i '' 's|results/cross_distribution_analysis|validation|g' visualization/core/data-loader.js

echo "  ✓ Updated 2 visualization files"

echo ""
echo "✅ Path refactor complete!"
echo ""
echo "Note: Shell scripts need manual review for complex path logic"
