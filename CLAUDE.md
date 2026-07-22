<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan
at specs/001-structure-aware-trust-gating/plan.md
<!-- SPECKIT END -->

## Cold Email Outreach

For drafting research mentorship cold emails, use `skill(name="cold-email-satg")`. This loads the established format, professor targeting criteria, and Composio Outlook draft flow used for the initial SATG outreach batch.

Key reference files:
- `cold-email-tracking/cold-email-prompt.md` — self-contained prompt for generating new emails
- `cold-email-tracking/TRACKING.md` — professor tracker
- `cold-email-tracking/emails-compiled.md` — example emails following the pattern

**Before sending any email, always attach the research proposal PDF** at `cold-email-tracking/research-proposal.pdf`.

## graphify

This project has a knowledge graph at `graphify-out/` with community structure, god nodes, and cross-file relationships.

When the user asks about codebase architecture or relationships, use graphify before grepping.

Rules:
- For codebase questions, first run `graphify query "<question>"` when `graphify-out/graph.json` exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts.
- If `graphify-out/wiki/index.md` exists, use it for broad navigation instead of raw source browsing.
- Read `graphify-out/GRAPH_REPORT.md` for broad architecture review or when query/path/explain don't surface enough.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
