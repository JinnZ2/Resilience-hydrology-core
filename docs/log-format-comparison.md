# Two Logs, Tested

This repository carries two falsification records:
[`method-log.md`](method-log.md) (M-nn) and [`research-log.md`](research-log.md)
(H-nn, O-nn). They exist because two sessions worked the same repository in
parallel, never saw each other's output, and were merged afterwards.

The obvious response was to pick one. Instead they were **tested**, using the
same standard the logs impose on the models: measure it, don't argue it. The
tool is [`../tools/log_audit.py`](../tools/log_audit.py); rerun it any time with

```bash
python tools/log_audit.py --verify-commands
```

## What was measured

Both logs promise the same three things, and each is mechanically checkable:

1. **Traceability** — a claim cites a command you can run today
2. **Completeness** — an entry carries the fields its own format requires
3. **Reachability** — code that is knowingly wrong points at the entry saying so

## Results

Measured 2026-08-16, before any corrections were made:

| | method-log | research-log |
|---|---|---|
| Lines | 435 | 593 |
| Entries | 8 | 9 (+13 open questions) |
| Entries citing a runnable command | **2 / 8** | **5 / 9** |
| Runs described in prose only | 5 | 2 |
| Entries missing a required field | **0 / 8** | **2 / 9** |
| Cited commands that actually execute | 2 / 2 | 4 / 4 |
| Distinct IDs cited from source code | 6 | 8 |
| Dangling references | 0 | 0 |

After fixing what the audit found, research-log reached 7/9 runnable and 0
incomplete. The `method-log` prose entries were left as they are — see below.

## What each format is actually good at

**method-log is more disciplined.** It hit 8/8 on its own required fields with
nobody checking. Its format is a per-claim ledger — Claim, Run, Result, Status —
with a fixed status vocabulary and a rule that code which is knowingly wrong
cites its `M-nn` ID in a comment. That rule works: M-01 through M-06 appear in
five source files. Someone editing `01_basic_dew.py` cannot miss that its
coefficient is unsourced.

**research-log is more reproducible.** It cites runnable commands nearly twice as
often, because its entry format has an explicit `Run:` line that wants a command
rather than a description. "Same runs as M-01" is a perfectly honest sentence
that a reader cannot execute.

**They answer different questions.** A ledger answers *"is this number safe to
quote?"* — look it up by ID, read its status. A round-based log with carried-
forward open questions answers *"what should we do next?"* — O1 through O13 are
a work queue, and the method log has no equivalent structure. Neither question is
more important; a project needs both answered.

**Their coverage is complementary, and the overlap is the point.** Each log found
things the other missed:

| Only in method-log | Only in research-log |
|---|---|
| M-06 GPIO5 pin collision — a real hardware defect | H7 climate presets make dew thermodynamically impossible |
| M-01 provenance of the retired 0.034/0.14 figures | H8 3x needs ~19x the available power |
| M-07 the paired-control requirement | H9 first model-vs-measurement comparison |
| | O1–O13 the open-question queue |

Where they overlap — the headline figure, the 3x factor, the degenerate seed
optimiser, the inverted crop tolerance — they agree, having been derived
independently. **That is a replication, and it is the strongest evidence in this
repository.** Everything else here is one person running one model once. These
four findings are the only ones two independent efforts reached separately.

Deleting either log to tidy up would destroy that, which is the argument for
keeping both that no amount of reasoning about file organisation would have
produced.

## The audit was wrong twice before it was right

Recorded because it is the same failure mode the logs exist to catch.

**First run**: method-log appeared to cite almost no commands — 1 of 8. That
looked like a real difference between the formats. It was not: method-log states
its runs in fenced ` ```bash ` blocks, and the extractor only matched inline
backticks. The instrument was miscalibrated, not the log.

**Second run**: M-01 was reported as citing a command that *failed to execute*.
Also false. The extractor had split a `for c in ...; do python ... --climate $c;
done` loop into lines and tried to run a fragment with an unbound variable.

Both errors ran in the same direction — making the other session's log look
worse than it is. Neither was caught by the audit; both were caught by looking at
a surprising result and going back to the source. An automated check is a claim
like any other, and it gets audited the same way.

## Recommendation

**Keep both, with a stated division of labour.**

- `method-log.md` holds precedence on everything up to M-08. It landed first, and
  where the two agree its entry is the original record.
- `research-log.md` continues from Round 2 and carries the open-question queue.
- New entries should take **method-log's field discipline** and **research-log's
  insistence on a literal command**. Those are the two measured strengths, and
  they are not in tension.

The 5 prose-only entries in method-log were deliberately not rewritten. Editing
another session's record to score better on a test written afterwards would be
the documentation equivalent of tuning constants until the output matches — the
exact failure M-03 is left broken to illustrate.
