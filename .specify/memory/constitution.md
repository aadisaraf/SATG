<!--
Sync Impact Report
- Version change: 0.0.0 → 1.0.0 (MAJOR: initial ratification)
- Modified principles: N/A (initial creation)
- Added sections: 6 (Research Integrity, Code Quality, Experimental Design, Documentation, Multi-Agent Coordination, Conflict Resolution)
- Removed sections: N/A
- Templates requiring updates:
  - .specify/templates/plan-template.md: ✅ Constitution Check section already flexible
  - .specify/templates/spec-template.md: ✅ No changes needed
  - .specify/templates/tasks-template.md: ✅ No changes needed
  - .specify/templates/checklist-template.md: ✅ No changes needed
- Follow-up TODOs: None
-->

# SATG Constitution

## Section 1: Research Integrity and Reproducibility

### 1.1 Seed Management (CRITICAL)

ALL experiments MUST use fixed random seeds set at the start of every script for
numpy (`np.random.seed`), torch (`torch.manual_seed`), torch.cuda
(`torch.cuda.manual_seed_all`), and Python's `random` module. The default seed
is 42, but it MUST be configurable via the YAML config file. For full
determinism, set `torch.backends.cudnn.deterministic = True` and
`torch.backends.cudnn.benchmark = False`.

**Rationale**: Random seed control is the foundation of reproducible deep
learning research. PyTorch's official reproducibility guide requires seeding
all four RNG sources. The mmsegmentation framework (used by DAFormer/MIC/HRDA)
follows this convention.

### 1.2 Multi-Seed Evaluation (CRITICAL)

ALL reported mIoU values MUST be averaged over exactly 3 independent runs with
seeds {42, 1337, 2024}. Report mean ± standard deviation. A single-seed result
MAY be reported as preliminary but MUST be clearly labeled "(single seed, not
final)".

**Rationale**: The DAFormer/MIC/HRDA lineage established the 3-seed convention
as standard for UDA segmentation. Mean ± std is the universally adopted format
in this community, ensuring comparability on Papers With Code leaderboards.

### 1.3 Config Documentation (CRITICAL)

ALL hyperparameters (tau_conf, tau_struct, EMA momentum, learning rate, batch
size, iterations, crop size, augmentation parameters) MUST be logged to a YAML
config file alongside every experimental result. An experiment without its config
file is considered incomplete.

**Rationale**: Config snapshots are essential for reproducibility. Hydra +
OmegaConf automatically saves resolved configs, but manual logging provides a
backup and ensures transparency.

### 1.4 No Cherry-Picking (CRITICAL)

NO cherry-picked results. Every ablation run MUST be documented in
EXPERIMENTS.md regardless of whether it improved performance.

**Rationale**: Selective reporting undermines scientific integrity. All results
must be visible to enable honest assessment of what works and what does not.

### 1.5 Structural Prior Validity (CRITICAL)

The structural prior MUST be computed solely from raw RGB pixel values using
classical, deterministic computer vision operations. It MUST NOT involve any
parameters learned from the target domain or any labels. This is a hard
constraint that protects the validity of the research claim.

**Rationale**: If the structural prior learns from target data, it violates the
UDA assumption that no target labels are available. This would invalidate the
entire research contribution.

### 1.6 Target Label Isolation (CRITICAL)

The model MUST NEVER access target domain ground truth labels at training time.
Violations invalidate the UDA claim entirely.

**Rationale**: The fundamental premise of UDA is learning from unlabeled target
data. Any leakage of target labels destroys the experimental validity.

### 1.7 Compute Documentation (CRITICAL)

ALL GPU compute used (GPU type, training hours, GPU memory peak) MUST be
documented in each experiment record in EXPERIMENTS.md.

**Rationale**: Compute requirements affect reproducibility and enable fair
comparison of methods. MIC's config explicitly includes `gpu_model` for this
reason.

### 1.8 Statistical Significance (HIGH)

Statistical significance of improvements: differences of less than 0.5 mIoU
MUST be discussed as marginal and not presented as significant.

**Rationale**: With only 3 seeds, small differences may be noise. The DAFormer
reports mean ± std of 68.3 ± 0.6, indicating that sub-0.5 differences require
cautious interpretation.

### 1.9 Per-Class IoU Reporting (HIGH)

Per-class IoU MUST be reported alongside mIoU for all final results.

**Rationale**: Per-class breakdowns reveal which classes benefit from adaptation
and which remain challenging. All major UDA papers (DAFormer, MIC, HRDA, ProDA)
provide per-class IoU in their comparison tables.

### 1.10 Trust Mask Coverage (HIGH)

Trust mask coverage ratio (percentage of target pixels trusted per batch) MUST
be logged during training to verify the mechanism is active.

**Rationale**: Logging coverage ratios ensures the trust gating mechanism is
actually functioning as intended, not collapsing to accept all or reject all
pixels.

## Section 2: Code Quality and Architecture

### 2.1 Library-First (CRITICAL)

Every functional component (structural prior computation, trust gating, EMA
update, loss function, data loading) MUST be implemented as a standalone,
independently importable and testable module before it is integrated into the
training loop. No monolithic training scripts.

**Rationale**: Modular design enables isolated testing, reuse, and
maintainability. This follows the mmsegmentation framework's approach of
separate model components.

### 2.2 Test-First (CRITICAL)

Unit tests for each module MUST be written and verified to fail (Red phase)
before any implementation code is written. Tests define the interface and the
contract. Implementation makes tests pass.

**Rationale**: TDD ensures clear interfaces and catches regressions early. For
ML code, this is especially important for data processing, loss functions, and
metrics computation.

### 2.3 Test Coverage (CRITICAL)

Every module MUST have ≥80% line coverage as measured by pytest-cov. This
threshold is enforced before any module is considered complete.

**Rationale**: High coverage ensures critical paths are tested. 80% is a
practical threshold that catches most bugs without requiring exhaustive testing
of trivial code.

### 2.4 Type Annotations and Docstrings (CRITICAL)

All functions and classes MUST have type annotations and Google-style docstrings
including Args, Returns, Raises, and Example sections.

**Rationale**: Type annotations enable static analysis with mypy. Google-style
docstrings are the standard in PyTorch and HuggingFace codebases, ensuring
consistency and readability.

### 2.5 Config-Driven Hyperparameters (CRITICAL)

ALL hyperparameters MUST be sourced from config files (OmegaConf YAML). Zero
hardcoded numeric constants in training or evaluation scripts.

**Rationale**: Hardcoded values break reproducibility and make experiments
difficult to compare. OmegaConf provides hierarchical YAML configs with
variable interpolation and type safety.

### 2.6 Tensor Shape Comments (CRITICAL)

Tensor operations MUST be accompanied by inline shape comments where the shape
is non-obvious (e.g., `# [B, C, H, W]` or `# [B, H, W]`).

**Rationale**: Shape mismatches are the most common source of bugs in deep
learning code. Inline comments make the expected shapes explicit.

### 2.7 Linting and Formatting (HIGH)

Code MUST pass flake8 linting (max line length 100) and black formatting before
being committed.

**Rationale**: Consistent formatting reduces cognitive load and prevents style
debates. The 100-character line length is the Python community standard.

### 2.8 Swappable Backbones (HIGH)

The backbone architecture MUST be swappable via config without code changes.

**Rationale**: Swappable backbones enable fair comparison of architectures and
facilitate ablation studies over encoder choices.

### 2.9 Heatmap Naming Convention (HIGH)

Precomputed heatmap files MUST follow a consistent naming convention:
`{image_stem}_satg_heatmap.npy`, stored alongside source images.

**Rationale**: Consistent naming enables reliable file lookup and prevents
accidental overwrites.

## Section 3: Experimental Design

### 3.1 Mandatory Ablations (CRITICAL)

Ablation studies are MANDATORY and MUST include:
(a) Confidence-only baseline (standard Mean Teacher, no structural prior)
(b) Edge-density-only structural prior
(c) Local-variance-only structural prior
(d) Combined structural prior (edge + variance)
(e) Hard rejection vs. soft weighting
(f) tau_conf sensitivity: at least 3 values
(g) tau_struct sensitivity: at least 3 values

**Rationale**: Ablations isolate the contribution of each component. Without
them, it is impossible to determine whether the structural prior or other design
choices drive performance.

### 3.2 Source Only Baseline (CRITICAL)

A Source Only baseline (no pseudo-labeling, source supervised only) MUST always
be evaluated and reported as the lower bound.

**Rationale**: Source-only quantifies the domain gap and measures adaptation
gain. Every UDA paper since ADVENT (2017) includes this baseline. Without it,
improvements cannot be attributed to the adaptation method.

### 3.3 Dry Run Validation (CRITICAL)

A Dry Run (10 images, 10 iterations) MUST pass without errors before any
full-scale training run begins.

**Rationale**: Early error detection prevents wasted compute on broken
experiments. This is a practical checkpoint that catches data loading,
shape mismatches, and configuration errors.

## Section 4: Documentation

### 4.1 README Completeness (CRITICAL)

README.md MUST include: installation, dataset download, precomputation,
training, and evaluation commands with exact CLI syntax.

**Rationale**: A complete README enables others to reproduce results without
guessing. CLI syntax examples reduce ambiguity.

### 4.2 Experiment Comparison Table (CRITICAL)

EXPERIMENTS.md MUST include a comparison table with all methods, their configs,
mIoU (mean ± std), and per-class IoU for key classes.

**Rationale**: A centralized comparison table enables quick assessment of
progress and facilitates paper writing.

### 4.3 Visualization Outputs (CRITICAL)

Visualization outputs showing trust mask overlays MUST be generated for at
least 10 diverse target images per configuration.

**Rationale**: Visualizations provide qualitative evidence of what the trust
mask is doing, complementing quantitative metrics.

## Section 5: Multi-Agent Coordination

### 5.1 Research Delegation (SHOULD)

Research tasks (literature search, hyperparameter survey, dataset
investigation) MAY be delegated to parallel subagents.

**Rationale**: Parallel research accelerates progress. Subagents can explore
different aspects simultaneously without blocking each other.

### 5.2 Implementation Delegation (SHOULD)

Independent implementation tasks (structural prior module, trust gate module,
data loader modules) MAY be assigned to parallel subagents.

**Rationale**: Independent modules can be developed in parallel, reducing
overall development time.

### 5.3 Output Verification (SHOULD)

All subagent outputs MUST be verified by the primary agent before merging into
the codebase. No unreviewed code is accepted.

**Rationale**: Quality control requires human or primary agent review. Subagent
outputs may contain errors, style inconsistencies, or design misalignments.

## Section 6: Conflict Resolution

When principles conflict:

- Research validity (Sections 1 and 3) takes precedence over everything.
- Code quality (Section 2) takes precedence over development speed.
- Method interpretability takes precedence over raw performance numbers.
- If compute is limited, reduce resolution or dataset size, but never skip
  required ablations or the Source Only baseline.

**Rationale**: Scientific rigor is the foundation of this project. Code quality
ensures long-term maintainability. Interpretability matters more than squeezing
out marginal performance gains.

## Governance

This constitution supersedes all other practices when conflicts arise. All PRs
and code reviews MUST verify compliance with these principles. Complexity must
be justified with reference to specific principles. Use `.specify/memory/constitution.md`
for runtime development guidance.

Amendments require:
1. Documentation of the proposed change
2. Justification referencing research or engineering rationale
3. Version increment following semantic versioning (MAJOR for principle
   removal/redefinition, MINOR for additions, PATCH for clarifications)

Compliance review expectations:
- Every experiment must be auditable against Section 1 principles
- Every module must pass test coverage thresholds from Section 2
- Every ablation must be documented per Section 3

**Version**: 1.0.0 | **Ratified**: 2026-06-24 | **Last Amended**: 2026-06-24
