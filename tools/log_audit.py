#!/usr/bin/env python3
"""
Audit the falsification logs: do they do the job they claim to do?

This repository now carries two independent records of the same work —
docs/method-log.md (M-nn) and docs/research-log.md (H-nn, O-nn). They were
written by separate sessions that never saw each other's output, then merged.

Rather than argue about which format is better, this tests them. Both logs make
the same three promises, and each promise is mechanically checkable:

  1. TRACEABILITY  Every claim cites a command you can run today, or a recorded
                   measurement with its conditions.
  2. RESOLUTION    Every ID referenced from code or docs resolves to a real
                   entry, and every entry has the fields its own format requires.
  3. REACHABILITY  Code that is knowingly wrong points at the entry saying so,
                   so someone reading the code finds the caveat.

A log that fails these is decoration. The point of running this is that the
answer is measured, not asserted — which is the same standard the logs
themselves impose on the models.

The comparison this produces is written up in docs/log-format-comparison.md.

Usage:
    python tools/log_audit.py
    python tools/log_audit.py --verify-commands   # actually execute cited commands
    python tools/log_audit.py --format json
"""

import argparse
import json
import os
import re
import subprocess
import sys


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOGS = {
    'method-log': {
        'path': 'docs/method-log.md',
        'id_pattern': r'\bM-(\d{2})\b',
        'entry_heading': re.compile(r'^## (M-\d{2})\s*[—-]\s*(.+)$', re.M),
        'required_fields': ['Claim', 'Run', 'Result', 'Status'],
        'field_pattern': r'\*\*{field}',
    },
    'research-log': {
        'path': 'docs/research-log.md',
        'id_pattern': r'\b(?:H|O)-?(\d{1,2})\b',
        'entry_heading': re.compile(r'^### (H\d+)\s*[—-]\s*(.+)$', re.M),
        'required_fields': ['Prediction', 'Run', 'Result', 'Verdict'],
        'field_pattern': r'\*\*{field}',
    },
}

# Where a claim's evidence can live. A command is strongest: anyone can rerun it.
# Both fenced blocks and inline backticks count — an early version of this script
# only matched inline backticks and reported method-log as citing almost no
# commands, because that log states its runs in fenced bash blocks. The
# instrument was miscalibrated, not the log. Checking a surprising result against
# the source is the same move the logs themselves are for.
FENCE_RE = re.compile(r'```(?:bash|sh|console)?\n(.*?)```', re.S)
INLINE_RE = re.compile(r'`((?:python|git|for )[^`\n]+)`')
RUN_FIELD_RE = re.compile(r'\*\*Run\*\*')


SHELL_CONTROL_RE = re.compile(r'^\s*(for|while|if)\b', re.M)


def extract_commands(body):
    """
    Commands cited in an entry, from fenced blocks or inline backticks.

    A fenced block containing shell control flow is kept whole: splitting a
    `for c in ...; do python x --climate $c; done` loop into lines yields a
    fragment with an unbound variable, which then fails to run and looks like
    the log's fault. It is not. Extract the block as one script instead.
    """
    found = []
    for block in FENCE_RE.findall(body):
        block = block.strip()
        if not block:
            continue
        if SHELL_CONTROL_RE.search(block):
            found.append(block)
            continue
        for line in block.splitlines():
            line = line.strip()
            if line and not line.startswith('#') and (
                    'python' in line or line.startswith(('for ', 'git '))):
                found.append(line)
    found += INLINE_RE.findall(body)
    return found
CODE_DIRS = ('simulations', 'firmware', 'tools')


def read(path):
    full = os.path.join(REPO, path)
    if not os.path.exists(full):
        return None
    with open(full, encoding='utf-8') as f:
        return f.read()


def split_entries(text, spec):
    """Split a log into (id, title, body) triples."""
    matches = list(spec['entry_heading'].finditer(text))
    entries = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        entries.append((m.group(1), m.group(2).strip(), text[m.start():end]))
    return entries


def audit_traceability(entries, spec):
    """Does each entry cite a runnable command?"""
    rows = []
    for eid, title, body in entries:
        commands = extract_commands(body)
        # A static read of the source is legitimate evidence, but weaker: it
        # cannot be re-run to check the claim still holds.
        static = bool(re.search(r'[Ss]tatic read|[Ss]earched the repository|'
                                r'inspection of|read `', body))
        has_run = bool(RUN_FIELD_RE.search(body)) or '**Run**' in body
        if commands:
            evidence = 'command'
        elif static:
            evidence = 'static'
        elif has_run:
            evidence = 'prose'      # describes a run, but not reproducibly
        else:
            evidence = 'none'
        rows.append({'id': eid, 'title': title, 'commands': commands,
                     'evidence': evidence})
    return rows


def audit_fields(entries, spec):
    """Does each entry carry the fields its own format promises?"""
    rows = []
    for eid, title, body in entries:
        missing = [f for f in spec['required_fields']
                   if not re.search(spec['field_pattern'].format(field=f), body)]
        rows.append({'id': eid, 'title': title, 'missing': missing})
    return rows


def collect_references():
    """Every M-nn / H-nn / O-nn cited anywhere outside the logs themselves."""
    refs = {}
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', 'legacy')]
        for fn in files:
            if not fn.endswith(('.md', '.py')):
                continue
            rel = os.path.relpath(os.path.join(root, fn), REPO)
            if rel in (spec['path'] for spec in LOGS.values()):
                continue
            text = read(rel) or ''
            for m in re.finditer(r'\b(M-\d{2}|H\d+|O\d+)\b', text):
                refs.setdefault(m.group(1), []).append(rel)
    return refs


def audit_reachability(refs, known_ids):
    """Do references resolve, and does code point at the log at all?"""
    dangling, resolved = {}, {}
    for ref, sites in refs.items():
        (resolved if ref in known_ids else dangling)[ref] = sorted(set(sites))
    code_refs = {r: sorted(set(s for s in sites if s.endswith('.py')))
                 for r, sites in refs.items()}
    code_refs = {r: s for r, s in code_refs.items() if s}
    return dangling, resolved, code_refs


def verify_commands(rows, limit=None):
    """Actually run the cited commands. The strongest test a log can pass."""
    results = []
    seen = set()
    for row in rows:
        for cmd in row['commands']:
            if cmd in seen:
                continue
            seen.add(cmd)
            if 'python simulations/' not in cmd:
                continue          # only simulation commands are safe to run here
            if limit and len(results) >= limit:
                return results
            # Strip argument placeholders like {a,b,c} — not literally runnable.
            runnable = re.sub(r'\{[^}]*\}', lambda m: m.group(0)[1:].split(',')[0],
                              cmd)
            runnable += ' --no-plot' if 'variable_search' in runnable or \
                                        'transition_paths' in runnable else ''
            env = dict(os.environ, MPLBACKEND='Agg')
            try:
                proc = subprocess.run(runnable, shell=True, cwd=REPO, env=env,
                                      capture_output=True, timeout=180)
                ok = proc.returncode == 0
            except subprocess.TimeoutExpired:
                ok = False
            label = cmd if len(cmd) < 70 else cmd.splitlines()[0][:66] + ' ...'
            results.append({'entry': row['id'], 'command': label, 'ok': ok})
    return results


def audit_log(name, spec, refs, run_commands):
    text = read(spec['path'])
    if text is None:
        return None
    entries = split_entries(text, spec)
    trace = audit_traceability(entries, spec)
    fields = audit_fields(entries, spec)
    ids = {e[0] for e in entries}
    # Open questions are a separate species of entry.
    open_ids = set(re.findall(r'^- \*\*(O\d+)', text, re.M)) | \
               set(re.findall(r'^- ~~\*\*(O\d+)', text, re.M))
    result = {
        'name': name, 'path': spec['path'],
        'lines': text.count('\n') + 1,
        'entries': len(entries),
        'ids': sorted(ids),
        'open_questions': sorted(open_ids),
        'traceability': trace,
        'fields': fields,
        'with_command': sum(1 for t in trace if t['evidence'] == 'command'),
        'static_only': sum(1 for t in trace if t['evidence'] == 'static'),
        'prose_only': sum(1 for t in trace if t['evidence'] == 'prose'),
        'no_evidence': sum(1 for t in trace if t['evidence'] == 'none'),
        'incomplete': [f for f in fields if f['missing']],
    }
    if run_commands:
        result['command_runs'] = verify_commands(trace)
    return result


def print_report(audits, refs):
    all_ids = set()
    for a in audits:
        all_ids |= set(a['ids']) | set(a['open_questions'])
    dangling, resolved, code_refs = audit_reachability(refs, all_ids)

    print("=" * 72)
    print("Falsification log audit")
    print("=" * 72)
    print("Two logs make the same three promises. This checks them.\n")

    print(f"{'log':<16}{'lines':>7}{'entries':>9}{'runnable':>10}"
          f"{'static':>8}{'prose':>7}{'none':>6}{'incomplete':>12}")
    for a in audits:
        print(f"{a['name']:<16}{a['lines']:>7}{a['entries']:>9}"
              f"{a['with_command']:>10}{a['static_only']:>8}"
              f"{a['prose_only']:>7}{a['no_evidence']:>6}"
              f"{len(a['incomplete']):>12}")

    print("\n" + "-" * 72)
    print("1. TRACEABILITY — can a reader re-run the evidence?")
    print("-" * 72)
    for a in audits:
        total = a['entries'] or 1
        pct = 100 * a['with_command'] / total
        print(f"\n  {a['name']}: {a['with_command']}/{a['entries']} "
              f"entries cite a runnable command ({pct:.0f}%)")
        for t in a['traceability']:
            if t['evidence'] != 'command':
                mark = {'static': 'static read (not re-runnable)',
                        'prose': 'run described in prose, no command',
                        'none': 'NO RUN RECORDED'}[t['evidence']]
                print(f"      {t['id']}: {mark} — {t['title'][:40]}")

    print("\n" + "-" * 72)
    print("2. COMPLETENESS — does each entry carry its own required fields?")
    print("-" * 72)
    for a in audits:
        req = ', '.join(LOGS[a['name']]['required_fields'])
        print(f"\n  {a['name']} requires: {req}")
        if not a['incomplete']:
            print(f"      all {a['entries']} entries complete")
        for f in a['incomplete']:
            print(f"      {f['id']}: missing {', '.join(f['missing'])}")

    print("\n" + "-" * 72)
    print("3. REACHABILITY — do citations resolve, and does code point back?")
    print("-" * 72)
    print(f"\n  {len(resolved)} distinct IDs cited outside the logs, all resolving")
    if dangling:
        print(f"  ! {len(dangling)} DANGLING references to entries that do not exist:")
        for ref, sites in sorted(dangling.items()):
            print(f"      {ref} cited in {', '.join(sites)}")
    else:
        print("  no dangling references")

    print(f"\n  Citations from source code (the ones that matter — a caveat in a")
    print(f"  doc is easy to miss, a caveat in the file you are editing is not):")
    if code_refs:
        for ref, sites in sorted(code_refs.items()):
            print(f"      {ref:<6} {', '.join(sites)}")
    else:
        print("      none — no code file cites a log entry")

    for a in audits:
        if 'command_runs' in a:
            print("\n" + "-" * 72)
            print(f"4. EXECUTION — {a['name']} cited commands, actually run")
            print("-" * 72)
            for r in a['command_runs']:
                print(f"  [{'ok ' if r['ok'] else 'FAIL'}] {r['entry']}: {r['command']}")

    print("\n" + "-" * 72)
    print("COVERAGE OVERLAP")
    print("-" * 72)
    print("  Which findings exist in one log but not the other. Duplicated")
    print("  findings are replication; unique ones are why both are kept.\n")
    for a in audits:
        print(f"  {a['name']:<15} {len(a['ids'])} entries: {', '.join(a['ids'])}")
        if a['open_questions']:
            print(f"  {'':<15} {len(a['open_questions'])} open questions: "
                  f"{', '.join(a['open_questions'])}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--verify-commands', action='store_true',
                        help='actually execute the commands each entry cites')
    parser.add_argument('--format', choices=['text', 'json'], default='text')
    args = parser.parse_args()

    refs = collect_references()
    audits = [a for a in (audit_log(name, spec, refs, args.verify_commands)
                          for name, spec in LOGS.items()) if a]
    if not audits:
        print("No logs found. Expected docs/method-log.md or docs/research-log.md.")
        return 1

    if args.format == 'json':
        print(json.dumps(audits, indent=2))
    else:
        print_report(audits, refs)

    failures = sum(a['no_evidence'] + len(a['incomplete']) for a in audits)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
