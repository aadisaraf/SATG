# LLM Recording Rules — SATG Research Information Archive

## Mandate

Every time an LLM/agent learns something new about SATG — whether from
reading code, running experiments, analyzing results, discussing design
decisions, or reviewing literature — **it MUST record that information**
in the appropriate file(s) within this archive before doing anything else.

## Rules

### Rule 1: Context-Aware Update

Read `INDEX.md` to find the right file for the new information. If the
information fits none of the existing files, create a new one and update
`INDEX.md`.

### Rule 2: Always Update KNOWLEDGE-LOG.md

Every significant finding MUST also be recorded in `07-KNOWLEDGE-LOG.md`
with a timestamp and clear description. This is the chronological backbone
of the archive.

### Rule 3: Maintain Cross-References

When adding information, update cross-references in affected files. If
a finding contradicts an earlier claim, note the contradiction explicitly
and explain how understanding evolved.

### Rule 4: Be Precise and Verifiable

- Include exact numbers, error rates, pixel counts, thresholds
- Reference exact file paths and line numbers where relevant
- Distinguish between **observed facts** and **inferred interpretations**
- Note confidence level (e.g., "confirmed", "strongly indicated", "suggested", "speculative")

### Rule 5: Capture Negative Results

Negative findings, failed hypotheses, and surprising results are as
valuable as positive ones. Record them all.

### Rule 6: Record Decisions and Their Rationale

When a design decision is made (e.g., "use soft-weighting not hard gating"),
record:
- What was decided
- Why (evidence / reasoning)
- What alternatives were rejected
- Who made the decision

### Rule 7: Paper-Relevant Synthesis

Where possible, frame findings in language suitable for a research paper:
motivation, method, experiment, result, discussion, limitation.

### Rule 8: Update After Every Diagnostic Run

Every time a new test or experiment completes, the results, analysis, and
any updated understanding must be recorded here.

---

## Violation Consequences

Failing to record information is treated as leaving evidence on the table.
The research paper will be weaker for it. Treat this archive as the most
important deliverable alongside the code itself.
