# Cold Email Generation — Prompt Reference

> Use this prompt when asking OpenCode to draft cold emails for research mentorship. It encodes the patterns, structure, and quality heuristics extracted from the SATG outreach batch.

---

## One-Shot Prompt

Copy the block below when asking OpenCode to draft cold emails:

```
Draft cold emails to professors for research mentorship in [YOUR_TOPIC].

My project: [2-3 sentence description of your project, model, dataset, and key idea]

Professor targets (format per row): Name — University — Email — Key Paper (Year) — Connection to my project

For each professor, write a personalized email following these EXACT specifications:

=== EMAIL STRUCTURE ===
1. **Subject**: 6-8 words, title case, references their subfield. No "Question about" or "High School Student:" preface.
2. **Opening sentence**: "I'm a high school student working on [your field], and I've been reading your work on [their specific topic/intervention]."
3. **Paper reference**: Name a specific paper AND a specific finding or insight from it. Show you actually read it.
4. **Connection**: 2-3 sentences describing your project. Connect it to THEIR finding in a specific, technical way.
5. **Question**: A genuine research question that connects their approach to yours. Not "can you mentor me" — a real intellectual inquiry.
6. **Attachment mention**: "I've attached a one-page overview of the project."
7. **Close**: "I'd appreciate any thoughts/perspective." SIMPLE. No "I would be deeply grateful if you could find the time..."
8. **Signature**: "Thanks,\n<Your Name>"

=== HARD RULES (violate any and the email is rejected) ===
- Maximum ~150-180 words
- Each email must reference a DIFFERENT paper/finding — no templates
- NO AI markers: never use "I am writing to express", "furthermore", "moreover", "delve", "explore", "passionate", "I've always been fascinated", "stood out", "struck me"
- DO use contractions: "I've", "I'm", "doesn't", "it's", "don't"
- Shorter sentences. Vary sentence length. Some sentences can be 8 words.
- Don't use perfect grammar structure — write like a real person wrote it
- The question at the end must be specific enough that they can answer in 2-3 sentences
- Never ask for a meeting, a call, or mentorship directly. Ask for their perspective/thoughts.
- Signature is just "Thanks,\nName" — no "Best regards", "Sincerely", "Warmly"

=== PROFESSOR SELECTION CRITERIA ===
- Assistant or Associate Professors (not full professors — lower response rate)
- Active publications in the last 2-3 years in the target field
- Lab websites that mention mentorship, students, or outreach
- Direct topical alignment to the project (not tangential)
- Mix of prestige institutions and accessible ones

=== DRAFTING PROCESS ===
1. For each professor, first search for their recent key paper
2. Identify ONE specific finding/insight from that paper
3. Write the email connecting THAT finding to my project
4. Verify against HARD RULES list
5. Verify word count
6. Remove any AI markers
```

---

## Analysis: What Makes These Emails Effective

### Structure Breakdown

```
Sentence 1: Identity + field context
    "I'm a high school student working on UDA semantic segmentation..."

Sentence 2-3: Specific paper engagement
    "...and I've been reading your work on [paper name]. Your finding that [specific insight]..."
    ↳ Shows genuine reading, not template

Sentence 4-6: Your project + connection
    "[2-3 sentences about your project]"
    ↳ Technical but accessible. Names architecture, dataset, key idea.

Sentence 7-8: Genuine question
    "I was wondering whether you think [specific question connecting their work to yours]."
    ↳ The question must have a real answer. Not rhetorical.

Sentence 9: Attachment hook
    "I've attached a one-page overview for context."
    ↳ Signals preparation without demanding anything.

Sentence 10: Close
    "I'd appreciate your perspective."
```

### Quality Heuristics

| Dimension | What Makes It Good |
|-----------|-------------------|
| **Personalization** | Not just naming a paper, but naming a *finding* within it. "Your CVPR 2022 paper on pixel trajectories" is good. "Your finding that contrastive learning extends to dense pixel-space time graphs" is better. |
| **Connection** | Explicitly links their finding to your specific approach. "Your Expected Confidence Score approach guides mixup by per-class performance — this resonates with my idea that not all pixels are equally trustworthy during self-training." |
| **Question quality** | Must be answerable in 2-3 sentences. "I was wondering whether structural priors at the prediction level could complement Gaussian-guided feature unlearning at the feature level." — invites a substantive but brief response. |
| **Tone** | Curious peer, not supplicant. You're doing interesting work and want their perspective, not their permission. |
| **Length** | 150-180 words. Every sentence earns its place. No life story, no grades, no generic praise. |

### Anti-Patterns (Common AI Slop)

| Don't Say | Instead Say |
|-----------|-------------|
| "I am writing to express my keen interest in..." | "I'm a high school student working on..." |
| "Your groundbreaking work on..." | (just name the work neutrally) |
| "This has always been a passion of mine" | "This is the challenge I keep coming back to" |
| "I would be deeply grateful if you could find the time..." | "I'd appreciate your perspective." |
| "Furthermore, your approach to..." | (delete — start a new sentence) |
| Perfect sentences with no contractions | "I've been reading your work on..." |
| "delve into", "explore the possibility" | "I've been testing whether..." |

### Subject Line Patterns

| Type | Formula | Example |
|------|---------|---------|
| Specific paper | `[Their Intervention] for [Your Domain]` | "Dense Correspondence for Pseudo-Label Quality in UDA" |
| Shared problem | `[Problem] in [Domain]` | "Class-Informed Supervision for UDA Semantic Segmentation" |
| Method transfer | `[Their Method] as [Your Approach]` | "Intrinsic Scene Properties as Structural Priors for UDA" |
| Contrast | `[Alternative Path] for [Shared Goal]` | "Threshold-Free Pseudo-Label Selection for Domain Adaptation" |

### Follow-Up Email Pattern

If no response after 7-10 days:
- Subject: `Re: [Original Subject]` (threaded)
- 3-4 sentences max
- Reference your original email briefly
- Add ONE new data point (experiment result, paper you read since)
- Close: "If you have any thoughts, I'd appreciate it."

---

## Attachment Reminder

**Always attach the one-page overview PDF** before sending. The attachment path is:
```
/Users/aadisaraf/Documents/research/SATG/cold-email-tracking/research-proposal.pdf
```

The email body should reference it: "I've attached a one-page overview of the project."

---

## Quick Reference: Full Email Flow

1. Select target professors (Assistant/Associate, recent pubs, topical alignment)
2. Read 1 key paper per professor — extract ONE specific finding
3. Draft email using structure above
4. Verify: word count, AI markers, contraction usage, question quality
5. Create draft in Outlook via Composio: `OUTLOOK_CREATE_MAIL_FOLDER_MESSAGE` (folder: "drafts")
6. Attach PDF: `OUTLOOK_ADD_MAIL_ATTACHMENT` with the local file path
7. Mark in TRACKING.md as `drafted`
8. After sending, update to `sent` with date
