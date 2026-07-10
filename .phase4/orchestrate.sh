#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# Phase 4 Orchestrator — (build → test) × 2 → triage
#
# Usage:
#   .phase4/orchestrate.sh --status         # Show progress
#   .phase4/orchestrate.sh --issue <NUM>    # Kick off (build → test) × 2 for one issue
#   .phase4/orchestrate.sh --continue       # After build, run tests; if fail, rebuild
#   .phase4/orchestrate.sh --reset <NUM>    # Reset issue back to pending
#
# State machine per issue:
#   pending → [--issue] → building → [AI fixes] → [--continue]
#   → testing → PASS → pass (remove xfail) ✓
#            → FAIL → rebuilding → [AI fixes with test output] → [--continue]
#            → testing → PASS → pass (remove xfail) ✓
#                     → FAIL → triage (write .phase4/triage/N.md) ✗
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATE="$ROOT/.phase4/state.json"
TRIAGE_DIR="$ROOT/.phase4/triage"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[p4]${NC} $*"; }
ok()    { echo -e "${GREEN}[p4]${NC} $*"; }
warn()  { echo -e "${YELLOW}[p4]${NC} $*"; }
fail()  { echo -e "${RED}[p4]${NC} $*"; }

ISSUES=(
  '{"num":83,"name":"relative path retargets after chdir","tests":"test_relative_path_retargets_after_chdir","files":["src/turbovecdb/database.py","crates/turbovecdb-core/src/database.rs"]}'
  '{"num":82,"name":"stale handle writes after delete","tests":"test_stale_handle","files":["crates/turbovecdb-core/src/collection.rs"]}'
  '{"num":93,"name":"future schema version accepted","tests":"test_future_schema_version","files":["crates/turbovecdb-core/src/collection.rs"]}'
  '{"num":107,"name":"HTTP body unbounded","tests":"test_http_","files":["src/turbovecdb/service.py"]}'
)

# ── helpers ──────────────────────────────────────────────────────────────────

ensure_branch() {
  local branch=$(python3 -c "
import json
d=json.load(open('$STATE'))
print(d.get('branch','phase-4'))
" 2>/dev/null || echo "phase-4")
  if [ "$(git branch --show-current 2>/dev/null)" != "$branch" ]; then
    if git rev-parse --verify "$branch" 2>/dev/null; then
      git checkout "$branch" 2>/dev/null
    else
      git checkout -b "$branch"
    fi
  fi
}

init_state() {
  mkdir -p "$TRIAGE_DIR"
  if [ ! -f "$STATE" ]; then
    cat > "$STATE" <<'EOF'
{
  "branch": "phase-4",
  "issues": {
    "83": {"status": "pending", "attempts": 0},
    "82": {"status": "pending", "attempts": 0},
    "93": {"status": "pending", "attempts": 0},
    "107": {"status": "pending", "attempts": 0}
  }
}
EOF
  fi
  ensure_branch
}

set_st() { python3 -c "
import json
d=json.load(open('$STATE'))
d['issues']['$1']['status']='$2'
json.dump(d,open('$STATE','w'),indent=2)
"; }
get_st() { python3 -c "
import json
d=json.load(open('$STATE'))
print(d['issues'].get('$1',{}).get('status','unknown'))
"; }
get_att() { python3 -c "
import json
d=json.load(open('$STATE'))
print(d['issues'].get('$1',{}).get('attempts',0))
"; }
inc_att() { python3 -c "
import json
d=json.load(open('$STATE'))
d['issues']['$1']['attempts']=d['issues']['$1'].get('attempts',0)+1
json.dump(d,open('$STATE','w'),indent=2)
print(d['issues']['$1']['attempts'])
"; }
save_fail_log() { python3 -c "
import json
d=json.load(open('$STATE'))
d['issues']['$1']['fail_log']='''$2'''
json.dump(d,open('$STATE','w'),indent=2)
"; }

build_rust() { cargo test --lib -p turbovecdb-core 2>&1 | tail -1; }
install_py() { python3 -m pip install -e . -q 2>&1 | tail -1; }

run_tests() {
  local num=$1 test_filter=$2
  info "Building Rust..."
  build_rust
  info "Installing Python..."
  install_py
  info "Testing #$num (filter: $test_filter)..."
  python3 -m pytest tests/test_phase4_bugs.py -k "$test_filter" -v 2>&1
}

remove_xfail() {
  local test_filter=$1
  python3 - "$test_filter" <<'PYEOF'
import re, sys
filter = sys.argv[1] if len(sys.argv) > 1 else ""
with open("tests/test_phase4_bugs.py", "r") as f:
    content = f.read()
lines = content.split("\n")
result = []
skip = False
for i, line in enumerate(lines):
    if "@pytest.mark.xfail" in line and not skip:
        for j in range(i+1, min(i+4, len(lines))):
            if "def test_" in lines[j] and filter in lines[j]:
                skip = True
                break
    if skip:
        skip = False
        continue
    result.append(line)
with open("tests/test_phase4_bugs.py", "w") as f:
    f.write("\n".join(result))
PYEOF
}

write_triage() {
  local num=$1 name=$2
  local tf="$TRIAGE_DIR/$num.md"
  local fl=$(python3 -c "
import json
d=json.load(open('$STATE'))
print(d['issues'].get('$num',{}).get('fail_log','none'))
" 2>/dev/null || echo "unknown")
  cat > "$tf" <<EOF
# Triage: Issue #$num — $name

Auto-generated after 2 failed (build → test) rounds.

## Failure output
$fl

## Next steps
1. Review \`tests/test_phase4_bugs.py\` for expected behaviour
2. Review \`.phase4/architecture.md\` for fix design
3. Implement manually
4. Test: \`python3 -m pytest tests/test_phase4_bugs.py -k "test_$num" -v\`
5. Remove xfail marker, commit
EOF
  warn "Triage written: $tf"
}

show_status() {
  init_state
  echo ""
  echo "╔═══════════════════════════════════════════════════════════════╗"
  echo "║               Phase 4 — Status                              ║"
  echo "╠═══════════════════════════════════════════════════════════════╣"
  for d in "${ISSUES[@]}"; do
    num=$(echo "$d" | python3 -c "import json,sys; print(json.load(sys.stdin)['num'])")
    name=$(echo "$d" | python3 -c "import json,sys; print(json.load(sys.stdin)['name'])")
    st=$(get_st "$num")
    st_emoji=$(python3 -c "
e = {'pending':'⏳','building':'🔧','testing':'🔬','rebuilding':'🔧','pass':'✅','triage':'❌'}
print(e.get('$st','❓'))
")
    attempts=$(get_att "$num")
    printf "║  #%-3s %-42s %s %d/2         ║\n" "$num" "$name" "$st_emoji" "$attempts"
  done
  echo "╠═══════════════════════════════════════════════════════════════╣"
  python3 -c "
import json
d=json.load(open('$STATE'))
print(f'║  Branch: {d.get(\"branch\",\"?\")}')
  " 2>/dev/null
  echo "╚═══════════════════════════════════════════════════════════════╝"
}

# ── issue loop ──────────────────────────────────────────────────────────────

process_issue() {
  local num=$1
  local issue_data=""
  for d in "${ISSUES[@]}"; do
    n=$(echo "$d" | python3 -c "import json,sys; print(json.load(sys.stdin)['num'])")
    [ "$n" = "$num" ] && issue_data="$d" && break
  done
  [ -z "$issue_data" ] && fail "Unknown issue #$num" && exit 1

  name=$(echo "$issue_data" | python3 -c "import json,sys; print(json.load(sys.stdin)['name'])")
  test_filter=$(echo "$issue_data" | python3 -c "import json,sys; print(json.load(sys.stdin)['tests'])")
  files=$(echo "$issue_data" | python3 -c "import json,sys; print(' '.join(json.load(sys.stdin)['files']))")

  st=$(get_st "$num")
  attempts=$(get_att "$num")
  info "Issue #$num — $name  (state=$st attempts=$attempts/2)"

  # ── already done ────────────────────────────────────────────────────
  [ "$st" = "pass" ]   && ok "Already passed!" && return 0
  [ "$st" = "triage" ] && warn "In triage — check $TRIAGE_DIR/$num.md" && return 0

  # ── Kick off: pending → building (first build) ─────────────────────
  if [ "$st" = "pending" ]; then
    set_st "$num" "building"
    echo ""
    echo "┌─────────────────────────────────────────────────────────────────┐"
    echo "│ BUILD Issue #$num — $name (attempt 1/2)"
    echo "│"
    for f in $files; do echo "│   $f"; done
    echo "│"
    echo "│ Spec:  .phase4/architecture.md"
    echo "│ Tests: tests/test_phase4_bugs.py (filter: $test_filter)"
    echo "│"
    echo "│ After implementing, run:"
    echo "│   .phase4/orchestrate.sh --continue"
    echo "└─────────────────────────────────────────────────────────────────┘"
    exit 0
  fi

  # ── Building → Testing (round 1 or 2) ──────────────────────────────
  if [ "$st" = "building" ] || [ "$st" = "rebuilding" ]; then
    inc_att "$num"
    attempts=$(get_att "$num")
    set_st "$num" "testing"

    if [ "$attempts" -gt 2 ]; then
      fail "Max attempts (2) for #$num — writing triage"
      write_triage "$num" "$name"
      set_st "$num" "triage"
      return 1
    fi

    info "Removing xfail markers for '$test_filter'..."
    remove_xfail "$test_filter"
    test_output=$(run_tests "$num" "$test_filter" 2>&1) && {
      ok "═══ Issue #$num PASSED! ═══"
      set_st "$num" "pass"

      # Check if all done
      local all_done=true
      for d2 in "${ISSUES[@]}"; do
        n2=$(echo "$d2" | python3 -c "import json,sys; print(json.load(sys.stdin)['num'])")
        [ "$(get_st "$n2")" != "pass" ] && all_done=false
      done
      if $all_done; then
        echo ""
        ok "═══════════════════════════════════════════════════════════════════"
        ok "  ALL 4 PHASE 4 ISSUES PASSED"
        ok "═══════════════════════════════════════════════════════════════════"
        echo ""
        echo "To finish:"
        echo "  git add -A && git commit -m \"feat(phase-4): fix #83 #82 #93 #107\""
        echo "  git push origin phase-4"
        echo "  gh pr create --fill"
      fi
      return 0
    } || {
      fail "═══ Issue #$num FAILED (attempt $attempts/2) ═══"
      save_fail_log "$num" "$test_output"
      # Show test output
      echo ""
      echo "── Test output ──────────────────────────────────────────"
      echo "$test_output"
      echo "────────────────────────────────────────────────────────"
      echo ""

      if [ "$attempts" -ge 2 ]; then
        fail "2 attempts used — writing triage"
        write_triage "$num" "$name"
        set_st "$num" "triage"
        return 1
      else
        set_st "$num" "rebuilding"
        echo "┌─────────────────────────────────────────────────────────────────┐"
        echo "│ REBUILD Issue #$num — $name (attempt $((attempts+1))/2)"
        echo "│"
        echo "│ Study the test failure above and fix the code."
        echo "│"
        echo "│ After fixing, run:"
        echo "│   .phase4/orchestrate.sh --continue"
        echo "└─────────────────────────────────────────────────────────────────┘"
        exit 0
      fi
    }
  fi

  fail "Unexpected state '$st' for #$num"
  return 1
}

# ═════════════════════════════════════════════════════════════════════════════
main() {
  init_state

  case "${1:---status}" in
    --status)
      show_status
      ;;
    --issue)
      [ -z "${2:-}" ] && fail "Usage: --issue <NUM>" && exit 1
      process_issue "$2"
      ;;
    --continue)
      local found=""
      for d in "${ISSUES[@]}"; do
        num=$(echo "$d" | python3 -c "import json,sys; print(json.load(sys.stdin)['num'])")
        st=$(get_st "$num")
        if [ "$st" = "building" ] || [ "$st" = "rebuilding" ]; then
          found="$num"
          process_issue "$num"
        fi
      done
      if [ -z "$found" ]; then
        warn "No issue in building/rebuilding state."
        echo "Start one: .phase4/orchestrate.sh --issue 83"
      fi
      ;;
    --reset)
      [ -z "${2:-}" ] && fail "Usage: --reset <NUM>" && exit 1
      python3 -c "
import json
d=json.load(open('$STATE'))
d['issues']['$2']={'status':'pending','attempts':0}
json.dump(d,open('$STATE','w'),indent=2)
" && ok "Reset #$2 to pending"
      ;;
    *)
      echo "Usage: .phase4/orchestrate.sh [--status|--issue NUM|--continue|--reset NUM]"
      exit 1
      ;;
  esac
}

main "$@"
