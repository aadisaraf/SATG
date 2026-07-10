# Graph Report - .  (2026-07-08)

## Corpus Check
- 51 files · ~61,597 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 141 nodes · 318 edges · 24 communities (15 shown, 9 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 12 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Speckit SDD Skills
- Shell Utility Scripts
- UDA Data Loading Research
- SATG Implementation & Risks
- Experimental Setup & Pipelines
- Training & Configuration
- Trust Gate & Loss Core
- Agent Context Extension
- Audit & Validation Gaps
- Structural Prior Pipeline
- Feature Scaffolding Scripts
- Research Integrity
- Speckit Templates & Workflow
- Code Principles
- Experimental Design Basics
- Shell Utility: Update Agent Context
- Shell Utility: Check Prerequisites
- Shell Utility: Setup Plan
- Shell Utility: Setup Tasks
- Data Processing Gap
- Dataloader Pattern
- Spec Kit Workflow
- UDA Framework

## God Nodes (most connected - your core abstractions)
1. `SATG Feature Specification` - 55 edges
2. `SATG Implementation Plan` - 26 edges
3. `SATG Task List` - 24 edges
4. `SATGTrainer Class` - 18 edges
5. `SATG Constitution` - 17 edges
6. `Extensions Configuration` - 15 edges
7. `SATG Research Summary` - 15 edges
8. `Module API Contracts` - 14 edges
9. `Extension Hook System` - 11 edges
10. `Dot-to-Hyphen Command Convention` - 11 edges

## Surprising Connections (you probably didn't know these)
- `AMP Integration Gap` --conceptually_related_to--> `SATG Research Summary`  [INFERRED]
  AUDIT_REPORT.md → specs/001-structure-aware-trust-gating/research.md
- `Config Key Naming Inconsistency` --conceptually_related_to--> `SATG Implementation Plan`  [INFERRED]
  AUDIT_REPORT.md → specs/001-structure-aware-trust-gating/plan.md
- `Config Key Naming Inconsistency` --conceptually_related_to--> `SATG Feature Specification`  [INFERRED]
  AUDIT_REPORT.md → specs/001-structure-aware-trust-gating/spec.md
- `Auxiliary Loss Handling Gap` --conceptually_related_to--> `DeepLabV3+ ResNet50`  [INFERRED]
  AUDIT_REPORT.md → specs/001-structure-aware-trust-gating/research.md
- `speckit-agent-context-update Skill` --references--> `Extensions Configuration`  [EXTRACTED]
  .claude/skills/speckit-agent-context-update/SKILL.md → .specify/extensions.yml

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Spec Kit Pipeline** — claude_skills_speckit_specify_skill, claude_skills_speckit_clarify_skill, claude_skills_speckit_plan_skill, claude_skills_speckit_tasks_skill, claude_skills_speckit_analyze_skill, claude_skills_speckit_implement_skill, claude_skills_speckit_converge_skill [EXTRACTED 1.00]
- **Extension Hook Lifecycle** — specify_extensions_yml, extension_hook_system, hook_condition_deferral, dot_to_hyphen_command_convention, agent_context_extension [EXTRACTED 1.00]
- **SATG Constitution Principles** — research_integrity_reproducibility, code_quality_architecture, experimental_design, multi_agent_coordination, conflict_resolution, specify_memory_constitution_authority_principle [EXTRACTED 1.00]
- **SATG Core Module Architecture** — structural_prior, hard_trust_gate, soft_trust_gate, temperature_soft_label, satg_loss, soft_label_kl_loss, ema_model, satg_trainer, evaluator [EXTRACTED 1.00]
- **Spec Kit SDD Cycle (Specify → Plan → Tasks → Implement)** — specify_templates_spec_template, specify_templates_plan_template, specify_templates_tasks_template, specify_workflows_speckit_workflow, specify_workflows_speckit_workflow [EXTRACTED 1.00]
- **SATG Data Pipeline (GTA5 + Cityscapes + Heatmaps)** — gta5_dataset, cityscapes_dataset, structural_heatmap, dual_domain_dataloader, itertools_cycle, source_augmentation_pipeline, target_augmentation_pipeline, gta5_label_collisions, cityscapes_19_class_mapping, heatmap_precomputation_pipeline [EXTRACTED 1.00]

## Communities (24 total, 9 thin omitted)

### Community 0 - "Speckit SDD Skills"
Cohesion: 0.25
Nodes (23): Checklists as Unit Tests for Requirements, speckit-analyze Skill, speckit-checklist Skill, speckit-clarify Skill, speckit-constitution Skill, speckit-converge Skill, speckit-implement Skill, speckit-plan Skill (+15 more)

### Community 1 - "Shell Utility Scripts"
Cohesion: 0.13
Nodes (5): get_feature_paths(), get_repo_root(), _persist_feature_json(), resolve_specify_init_dir(), common.sh script

### Community 2 - "UDA Data Loading Research"
Cohesion: 0.30
Nodes (12): Cityscapes 19-Class Mapping, Cityscapes Dataset, Confirmation Bias in Pseudo-Label UDA, GTA5 to Cityscapes Data Loading Research, GTA5 to Cityscapes Domain Gap, DualDomainDataLoader Pattern, GTA5 Dataset, GTA5 Label RGB Collisions (+4 more)

### Community 3 - "SATG Implementation & Risks"
Cohesion: 0.27
Nodes (10): CLAUDE.md Agent Context, Combination Experiments (SATG + DAFormer), Risk: Structural Complexity / Class Frequency Confounding, Risk: Missing Combination Experiments, Risk: Statistical Power with 3 Seeds, SoftLabelKLLoss Class, Soft Modulation as Primary Contribution, SATG Implementation Plan (+2 more)

### Community 4 - "Experimental Setup & Pipelines"
Cohesion: 0.25
Nodes (9): Ablation Studies (A-G), Cloud Setup and Training Scripts, Dry Run Verification, Heatmap Precomputation Pipeline, Hypothesis Validation Framework, Multi-Subagent Coordination Pattern, Rare Class Sampling, SATG Quickstart Guide (+1 more)

### Community 5 - "Training & Configuration"
Cohesion: 0.39
Nodes (8): Auxiliary Loss Handling Gap, DeepLabV3+ ResNet50, EMAModel Class, OmegaConf Configuration, PolynomialLR Scheduler, SATGTrainer Class, SATG Research Summary, WandB Experiment Tracking

### Community 6 - "Trust Gate & Loss Core"
Cohesion: 0.36
Nodes (8): HardTrustGate Class, Pseudo-Label Map, SATGLoss Class, SoftTrustGate Class, SATG Data Model, Structural Confirmation Bias, Teacher Confidence Map, Trust Mask / Weights

### Community 7 - "Agent Context Extension"
Cohesion: 0.71
Nodes (7): Coding Agent Context Extension, speckit-agent-context-update Skill, Agent Context Config, speckit.agent-context.update Command, Agent Context Extension Definition, Agent Context Extension README, SPECKIT START/END Markers

### Community 8 - "Audit & Validation Gaps"
Cohesion: 0.38
Nodes (7): AMP Integration Gap, SATG Audit Report, Config Key Naming Inconsistency, Evaluator Class, Validation Checklist, Module API Contracts, Stale Validation Checklist

### Community 9 - "Structural Prior Pipeline"
Cohesion: 0.60
Nodes (6): Canny Edge Detection, Structural Prior CV Pipeline Reference, Structural Cues for SATG Research, Edge Density Feature, Local Variance Feature, StructuralPrior Class

### Community 11 - "Research Integrity"
Cohesion: 0.50
Nodes (4): Research Integrity and Reproducibility, Seed Management, Structural Prior Validity, Target Label Isolation

### Community 12 - "Speckit Templates & Workflow"
Cohesion: 1.00
Nodes (4): Implementation Plan Template, Feature Specification Template, Task List Template, Speckit SDD Cycle

### Community 13 - "Code Principles"
Cohesion: 0.67
Nodes (3): Code Quality and Architecture, Library-First Principle, Test-First Principle

### Community 14 - "Experimental Design Basics"
Cohesion: 0.67
Nodes (3): Experimental Design, Mandatory Ablations, Source Only Baseline

## Knowledge Gaps
- **12 isolated node(s):** `update-agent-context.sh script`, `check-prerequisites.sh script`, `common.sh script`, `create-new-feature.sh script`, `setup-plan.sh script` (+7 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SATG Feature Specification` connect `UDA Data Loading Research` to `SATG Implementation & Risks`, `Experimental Setup & Pipelines`, `Training & Configuration`, `Trust Gate & Loss Core`, `Audit & Validation Gaps`, `Structural Prior Pipeline`, `Speckit Templates & Workflow`?**
  _High betweenness centrality (0.128) - this node is a cross-community bridge._
- **Why does `SATG Constitution` connect `Speckit SDD Skills` to `Research Integrity`, `Code Principles`, `Experimental Design Basics`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Why does `SATG Implementation Plan` connect `SATG Implementation & Risks` to `UDA Data Loading Research`, `Experimental Setup & Pipelines`, `Training & Configuration`, `Trust Gate & Loss Core`, `Audit & Validation Gaps`, `Structural Prior Pipeline`, `Speckit Templates & Workflow`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._
- **What connects `update-agent-context.sh script`, `check-prerequisites.sh script`, `common.sh script` to the rest of the system?**
  _26 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Shell Utility Scripts` be split into smaller, more focused modules?**
  _Cohesion score 0.1323529411764706 - nodes in this community are weakly interconnected._