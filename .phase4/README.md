# Phase 4 Orchestrator

Multi-agent build-test-triage loop for Phase 4 issues.

## Usage

```bash
# See current status
.phase4/orchestrate.sh --status

# Start working on a specific issue
.phase4/orchestrate.sh --issue 83

# After implementing the fix, continue to test
.phase4/orchestrate.sh --continue

# If tests fail on 2nd attempt, triage file written to .phase4/triage/
```

## Workflow per issue

1. **Architecture spec** → `.phase4/architecture.md`
2. **Build** → AI implements fix
3. **Test** → runs pytest on issue's tests
4. If fail → max 2 attempts
5. If still fail → `.phase4/triage/` file written

## Issues

| #  | Name                    | Files                                    |
|----|-------------------------|------------------------------------------|
| 83 | relative-path-chdir     | database.py, database.rs                 |
| 82 | stale-handle-delete     | collection.rs                            |
| 93 | future-schema-version   | collection.rs                            |
| 107| http-body-bounds        | service.py                               |
