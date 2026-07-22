# Cold Email Drafts - SATG Research Mentorship

Compiled on 2026-07-13. All 10 drafts staged in Outlook (Drafts folder). ✅

---

## 1. Ming Shao — UMass Lowell
**To:** ming_shao@uml.edu
**Subject:** High School Student: Question about structure-preserved domain adaptation for semantic segmentation
**Paper:** Graph Adaptive Knowledge Transfer for Unsupervised Domain Adaptation (ECCV 2018)
**Insight:** Graph structure regularization for knowledge transfer in multi-source UDA

Dear Professor Shao,

I'm a high school student working on UDA semantic segmentation, and I came across your ECCV 2018 paper on graph adaptive knowledge transfer. Using graph structure to regularize knowledge transfer across source domains is a perspective I hadn't considered before starting my own project.

I've been building SATG, a framework for GTA5→Cityscapes adaptation on DeepLabV3+ ResNet50. Instead of binary pseudo-label filtering, it modulates supervision through temperature-scaled soft labels guided by edge density and local variance — the idea being that structural complexity predicts where pseudo-labels are unreliable.

Your graph-based structure regularization operates at the representation level, while SATG gates at the prediction level. I'm curious whether you think these could be complementary — does preserving structure in latent space reduce the need for structural gating on outputs, or do they solve different problems?

I've attached a one-page overview of the project. I'd be grateful for any perspective, and if you have a few minutes, I'd love to hear your thoughts.

Thanks,
Aadi Saraf

---

## 2. Judy Hoffman — UC Irvine
**To:** judy.hoffman@uci.edu
**Subject:** High School Student: Question about spatial structure across domain shifts
**Paper:** FCNs in the Wild (arXiv 2016), EgoBridge (NeurIPS 2025)
**Insight:** Spatial layout constraints improve adaptation; Optimal Transport-based latent alignment for cross-domain transfer

Dear Professor Hoffman,

I'm a high school student working on unsupervised domain adaptation for semantic segmentation, and I've been reading your work — both the earlier FCNs in the Wild paper and your more recent EgoBridge paper from NeurIPS 2025. Your finding that category-specific spatial layout constraints improve adaptation, and EgoBridge's approach using Optimal Transport to align latent spaces across extreme viewpoint gaps, both seem to touch on a question I keep coming back to: what about spatial structure stays stable when everything else shifts?

I've been working on a project called SATG, which replaces hard pseudo-label filtering with temperature-scaled soft labels guided by structural priors (edge density, local variance) for GTA5→Cityscapes on DeepLabV3+ ResNet50. The hypothesis I'm testing is that structural complexity might correlate with pseudo-label error, so supervision should degrade gracefully in those regions.

EgoBridge's OT-based latent alignment works at the feature level, while SATG approaches things at the prediction level. I was wondering how these might interact — would aligned features reduce the need for structural gating, or do they address different failure modes?

I've attached a one-page overview of the project. Any perspective you have would mean a lot — I'm trying to figure out which direction to take the next experiments.

Thanks,
Aadi Saraf

---

## 3. Andrew Owens — Cornell Tech
**To:** andrew.owens@cornell.edu
**Subject:** High School Student: Question about dense correspondence for pseudo-label quality
**Paper:** Learning Pixel Trajectories with Multiscale Contrastive Random Walks (CVPR 2022)
**Insight:** Extending contrastive learning to dense pixel-space space-time graphs with coarse-to-fine transition matrices

Dear Professor Owens,

I'm a high school student working on UDA semantic segmentation, and I've been working through your CVPR 2022 paper on pixel trajectories with multiscale contrastive random walks. The way you extend contrastive learning to dense pixel-space time graphs is something I haven't seen elsewhere, and it made me wonder: could the same multiscale contrastive principle help improve pseudo-label quality in UDA? If dense correspondences can be learned across video frames without labels, maybe correspondences across domains (synthetic→real) could also be learned at the pixel level.

The project I've been working on, SATG, uses temperature-scaled soft labels modulated by structural priors on DeepLabV3+ for GTA5→Cityscapes. Instead of hard confidence thresholds, it uses continuous weighting. But your contrastive random walks suggest an alternative path — learning cross-domain correspondences directly instead of modulating by image structure.

I was curious whether you think a contrastive objective between source and target feature maps could produce more reliable pseudo-labels than confidence-based selection.

I've attached a one-page overview of my project for context. I'd appreciate any thoughts — I suspect edge density and confidence scores capture different kinds of signal, and I'd love to hear whether you agree.

Thanks,
Aadi Saraf

---

## 4. Zhengming Ding — Tulane University
**To:** zding1@tulane.edu
**Subject:** High School Student: Question about class-informed mixup for UDA semantic segmentation
**Paper:** IDA: Informed Domain Adaptive Semantic Segmentation (2023)
**Insight:** Expected Confidence Score for class-level performance-guided mixup instead of random sampling

Dear Professor Ding,

I'm a high school student working on UDA semantic segmentation, and I've been reading your IDA paper. I noticed your observation that small-region semantics — poles, traffic signs, fences — get systematically washed out during adaptation. That matches what I've seen in my own experiments.

Your Expected Confidence Score approach guides mixup by per-class performance rather than random sampling. This resonates with what I've been working on — the idea that not all pixels are equally trustworthy during self-training.

My project SATG uses temperature-scaled soft pseudo-labels modulated by edge density and local variance for GTA5→Cityscapes on DeepLabV3+ ResNet50, instead of binary rejection. The shared motivation is that supervision quality varies across the image, and binary accept-reject decisions waste useful signal in uncertain regions.

I was wondering whether you think structural priors like edge density could complement confidence-based sampling for deciding which regions to trust during adaptation.

I've attached a one-page overview of the project. I'm trying to understand whether structural priors and class-informed sampling capture different signals — would appreciate your perspective.

Thanks,
Aadi Saraf

---

## 5. Anand Bhattad — Johns Hopkins University
**To:** bhattad@jhu.edu
**Subject:** High School Student: Question about intrinsic representations as structural priors
**Paper:** StyleGAN knows Normal, Depth, Albedo, and More (NeurIPS 2023)
**Insight:** Generative models encode intrinsic scene properties without explicit supervision — structure emerges naturally

Dear Professor Bhattad,

I'm a high school student working on UDA semantic segmentation, and I found your NeurIPS 2023 paper fascinating — the finding that StyleGAN encodes depth, normals, and albedo without explicit supervision suggests that intrinsic scene properties emerge naturally from generative training.

In my own project, SATG, I've been testing a related hypothesis — that structural signals (edge density, local variance) can predict pseudo-label reliability during domain adaptation on DeepLabV3+ ResNet50 for GTA5→Cityscapes. The idea is that pixels on edges and texture boundaries are where the teacher is most likely wrong, so those regions might need softened targets.

Your results made me wonder: if emergent intrinsic representations from generative models capture depth and normals, could they serve as stronger structural priors than classical edge detection? Has your work on intrinsic properties ever intersected with domain adaptation?

I've attached a one-page overview of my project for context. I'd appreciate any thoughts — I'm trying to decide whether to push the structural prior further or pivot toward contrastive correspondence learning.

Thanks,
Aadi Saraf

---

## 6. Yunhui Guo — UT Dallas
**To:** yunhui.guo@utdallas.edu
**Subject:** High School Student: Question about threshold-free pseudo-label selection
**Paper:** Segment Every Out-of-Distribution Object (CVPR 2024)
**Insight:** Converting anomaly scores into SAM prompts to avoid threshold selection entirely

Dear Professor Guo,

I'm a high school student working on UDA semantic segmentation, and I've been thinking about a design tension that your S2M paper (CVPR 2024) addresses from a different angle: where to set the confidence threshold. Pick 0.9 and you lose useful supervision; pick 0.7 and noise corrupts training.

Your approach sidesteps this by converting anomaly scores into SAM prompts — delegating hard segmentation boundaries to a promptable model. OoD detection and pseudo-label refinement seem to share the same root challenge: model predictions become unreliable outside the training distribution.

In my project SATG, I've been exploring a different way to avoid thresholding — replacing binary confidence thresholds with temperature-scaled soft labels guided by edge density and local variance for GTA5→Cityscapes adaptation on DeepLabV3+ ResNet50. The idea is that supervision degrades gracefully instead of being accepted or rejected.

Your work made me wonder: should we even be setting thresholds at all, or could a foundation model handle the hard segmentation decisions during adaptation?

I've attached a one-page overview of my project. I'm curious whether your approach of routing anomaly scores through SAM prompts could generalize to pseudo-label filtering — it feels like a threshold-free alternative to the confidence calibration problem.

Thanks,
Aadi Saraf

---

## 7. Mohammad Rostami — USC
**To:** rostamim@usc.edu
**Subject:** High School Student: Question about source-free domain adaptation with structural priors
**Paper:** Source-free domain adaptation for semantic image segmentation using internal representations (Frontiers in Big Data, 2024)
**Insight:** Approximating source latent feature distribution via GMM surrogate to eliminate source data access

Dear Professor Rostami,

I'm a high school student working on UDA semantic segmentation, and I found your Frontiers in Big Data paper (with Serban Stan) on source-free domain adaptation interesting for a practical reason: most UDA papers assume source data is always available during adaptation, but in deployment it often isn't.

Your approach of approximating the source latent feature distribution via a GMM surrogate during adaptation seems like a practical way to address this. My project SATG also operates in a source-free setting — it uses temperature-scaled soft pseudo-labels modulated by structural priors (edge density, local variance) for GTA5→Cityscapes on DeepLabV3+ ResNet50. The structural prior doesn't need source data since it's computed per target image.

I was wondering whether you think a source feature distribution surrogate could be combined with structure-based trust gating for stronger pseudo-label filtering in the source-free setting.

I've attached a one-page overview of the project. I think your approach of approximating source distributions via a learned surrogate suggests an interesting path for source-free UDA — one SATG could potentially incorporate.

Thanks,
Aadi Saraf

---

## 8. Carl Vondrick — Columbia University
**To:** vondrick@cs.columbia.edu
**Subject:** High School Student: Question about self-supervised representations for UDA
**Paper:** Generating Videos with Scene Dynamics (NeurIPS 2016)
**Insight:** Scene structure and dynamics emerge from generative video modeling without human annotation; self-supervised learning from unlabeled video

Dear Professor Vondrick,

I'm a high school student working on UDA semantic segmentation, and I've been reading your work on self-supervised learning from unlabeled video. In teacher-student self-training for UDA, pseudo-labels come from a source-trained teacher — but those pseudo-labels inherit the domain gap since the teacher was never trained on target data.

Your body of work suggests a different path. Your NeurIPS 2016 paper on generating videos with scene dynamics shows that models can learn about scene structure purely from unlabeled video — without any human annotation or depth supervision. This made me wonder: could self-supervised features replace the source-trained teacher entirely for pseudo-label generation in domain adaptation?

My project SATG uses temperature-scaled soft labels gated by structural priors (edge density, local variance) on DeepLabV3+ ResNet50 for GTA5→Cityscapes. But if structure emerges from unlabeled data, maybe self-supervised features could serve as a domain-agnostic backbone for pseudo-labeling.

I've attached a one-page overview of my project for context. Your perspective on whether self-supervised features could fully replace a source-trained teacher would be really valuable — I'm trying to understand if SSL removes the domain gap problem rather than patching around it.

Thanks,
Aadi Saraf

---

## 9. Zhaozheng Yin — Stony Brook University
**To:** zyin@cs.stonybrook.edu
**Subject:** High School Student: Question about disentangling domain-specific features with structural gating
**Paper:** Forget More to Learn More: Domain-specific Feature Unlearning for Semi-supervised and Unsupervised Domain Adaptation (ECCV 2024)
**Insight:** Domain-specific features persist alongside domain-agnostic ones; Gaussian-guided latent alignment disentangles them

Dear Professor Yin,

I'm a high school student working on UDA semantic segmentation, and your ECCV 2024 paper "Forget More to Learn More" described a challenge I've run into myself — domain-specific and domain-agnostic features coexist in the same model, so combining DA with SSL can amplify domain bias instead of reducing it.

I've been approaching this at the prediction level. My project SATG uses structural priors (edge density, local variance) to modulate pseudo-label trust during GTA5→Cityscapes adaptation on DeepLabV3+ ResNet50. The idea behind it is that structural complexity might be domain-agnostic — edges are edges whether in GTA5 or Cityscapes — so it could serve as a reliable signal for gating supervision regardless of domain.

I was wondering whether structural priors at the prediction level could complement Gaussian-guided feature unlearning at the feature level.

I've attached a one-page overview of the project. Your perspective on whether prediction-level structural gating could complement feature-level unlearning would be really valuable — I'm trying to figure out if they solve different parts of the same problem.

Thanks,
Aadi Saraf

---

## 10. Boqing Gong — Boston University
**To:** bgong@bu.edu
**Subject:** High School Student: Question about curriculum-inspired soft weighting for UDA
**Paper:** Curriculum Domain Adaptation for Semantic Segmentation of Urban Scenes (ICCV 2017)
**Insight:** Adaptation difficulty varies across the image; curriculum learning schedules easy→hard regions

Dear Professor Gong,

I'm a high school student working on UDA semantic segmentation, and your ICCV 2017 paper on curriculum domain adaptation for urban scenes has been really helpful to think through. Not all target regions are equally adaptable — some transfer well (uniform road, broad buildings) while others like fences and poles consistently fail. Your paper was one of the first to formalize this as a curriculum: start with easy-to-adapt regions and progressively incorporate harder ones.

That insight — that adaptation difficulty varies across the image — directly motivates my project. I've been working on SATG, a UDA semantic segmentation framework for GTA5→Cityscapes on DeepLabV3+ ResNet50. It modulates pseudo-label supervision using structural priors (edge density, local variance) as a proxy for adaptation difficulty, with structurally complex regions getting softened supervision.

I think this mirrors the curriculum you proposed, applied continuously rather than in discrete stages. I was wondering whether you think a soft weighting scheme informed by image structure might be more effective than staged curriculum learning.

I've attached a one-page overview of the project. I'm trying to understand whether a continuous curriculum (soft weighting) is more principled than a staged one — would appreciate your take.

Thanks,
Aadi Saraf
