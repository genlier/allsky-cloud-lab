# Archived algorithms

**English** | [简体中文](README.zh-CN.md)

These two source files preserve handcrafted features, decision trees, random forests, SVMs, peridata time matching and the original CNN training approach for reference.

The original evaluation added supplementary samples after a random split and selected the best epoch using the test set. For reproducible experiments, use the root-level training entry point with pixel deduplication, a separate validation set and final test evaluation.

The original `--use-peridata` returns different feature dimensions when records are missing. This historical implementation needs repair before use. Field units and time zones also require confirmation from instrument documentation.
