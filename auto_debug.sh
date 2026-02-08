#!/bin/bash
set -e

SUBJECT=${1:-sub-0001}
SESSION=${2:-ses-0002}
MAX_ITERATIONS=${3:-10}
SUBJECT_CLEAN=${SUBJECT//-/}
TEST_FILE="tests/integration/test_${SUBJECT_CLEAN}.py"

PROJECT_DIR=~/Developer/pmriprep
PERF_PLUS=~/Developer/perf-plus/bidsroot
LOG_DIR=$PROJECT_DIR/debug_logs/$(date +%Y%m%d_%H%M%S)_${SUBJECT}
mkdir -p $LOG_DIR

cd $PROJECT_DIR
conda activate pmriprep-env

for i in $(seq 1 $MAX_ITERATIONS); do
    echo ""
    echo "========================================"
    echo "  ITERATION $i of $MAX_ITERATIONS"
    echo "  Subject: $SUBJECT / $SESSION"
    echo "========================================"
    
    # =========================================
    # STEP 1: Run integration test
    # =========================================
    echo "[Step 1] Running integration tests..."
    pytest $TEST_FILE -v 2>&1 | tee $LOG_DIR/iter${i}_test.txt
    
    # Check if all passed
    if grep -q " passed" $LOG_DIR/iter${i}_test.txt && \
       ! grep -qE "(FAILED|ERROR)" $LOG_DIR/iter${i}_test.txt; then
        echo ""
        echo "========================================" 
        echo "  ALL TESTS PASSED!"
        echo "========================================"
        exit 0
    fi
    
    # =========================================
    # STEP 2: Determine focus (DSC or DCE)
    # =========================================
    if grep -qE "FAILED.*dsc" $LOG_DIR/iter${i}_test.txt; then
        FOCUS="dsc"
    elif grep -qE "FAILED.*dce" $LOG_DIR/iter${i}_test.txt; then
        FOCUS="dce"
    else
        FOCUS="other"
    fi
    echo "[Step 2] Focus: $FOCUS"
    
    # =========================================
    # STEP 3: Analyze failures + plan fixes
    # =========================================
    echo "[Step 3] Analyzing failures..."
    
    cat > $LOG_DIR/iter${i}_analyze_prompt.txt << PROMPT
You are debugging PMRIPrep integration tests.

## Test Results
$(cat $LOG_DIR/iter${i}_test.txt)

## Focus
Analyze ${FOCUS^^} failures only.

## Task
Compare PMRIPrep code to perf-plus reference:
- PMRIPrep: pmriprep/utils/${FOCUS}_*.py, pmriprep/models/${FOCUS}/
- perf-plus: ~/Developer/perf-plus/bidsroot/code/

For each failure:
1. Root cause
2. perf-plus implementation (show code)
3. PMRIPrep implementation (show code)
4. Specific difference causing the failure

Output a JSON plan:
\`\`\`json
{
  "failures": [
    {
      "test": "test_name",
      "root_cause": "...",
      "perf_plus_approach": "...",
      "pmriprep_approach": "...",
      "recommended_fix": "..."
    }
  ]
}
\`\`\`

CRITICAL: Do not say outputs are stale. Find the code bug.
PROMPT

    claude -p "$PROJECT_DIR" --output-format text \
        "$(cat $LOG_DIR/iter${i}_analyze_prompt.txt)" \
        2>&1 | tee $LOG_DIR/iter${i}_analysis.txt
    
    # =========================================
    # STEP 4: Physics expert evaluation
    # =========================================
    echo "[Step 4] Physics expert review..."
    
    cat > $LOG_DIR/iter${i}_physics_prompt.txt << PROMPT
You are a perfusion MRI physics expert reviewing proposed code changes.

## Analysis from previous step
$(cat $LOG_DIR/iter${i}_analysis.txt)

## Your task
For each proposed fix, evaluate:

1. Is the perf-plus method CORRECT according to established perfusion MRI literature?
   - SVD deconvolution: Østergaard 1996, Wu 2003
   - Tofts model: Tofts 1999
   - AIF detection: best practices

2. Decision for each fix:
   - ADOPT: perf-plus method is correct, use it
   - ADOPT_WITH_DOCS: perf-plus works but is legacy/suboptimal, add to FUTURE_WORK.md
   - REJECT: propose alternative based on literature

Output JSON:
\`\`\`json
{
  "decisions": [
    {
      "fix": "description",
      "decision": "ADOPT|ADOPT_WITH_DOCS|REJECT",
      "rationale": "...",
      "future_work_note": "..." // if ADOPT_WITH_DOCS
    }
  ]
}
\`\`\`
PROMPT

    claude -p "$PROJECT_DIR" --output-format text \
        "$(cat $LOG_DIR/iter${i}_physics_prompt.txt)" \
        2>&1 | tee $LOG_DIR/iter${i}_physics.txt
    
    # =========================================
    # STEP 5: Apply fixes + code review
    # =========================================
    echo "[Step 5] Applying fixes with nipreps review..."
    
    cat > $LOG_DIR/iter${i}_fix_prompt.txt << PROMPT
Apply approved fixes from physics review.

## Physics Expert Decisions
$(cat $LOG_DIR/iter${i}_physics.txt)

## Tasks
1. Apply all ADOPT and ADOPT_WITH_DOCS fixes
2. For ADOPT_WITH_DOCS: add note to docs/FUTURE_WORK.md
3. After changes, review for nipreps compliance:
   - Type hints on all functions
   - Google-style docstrings
   - init_*_wf() naming for workflows
   - Proper inputnode/outputnode

Make the changes now. Do not ask for confirmation.
PROMPT

    claude -p "$PROJECT_DIR" --output-format text --allowedTools "Edit,Write,Bash" \
        "$(cat $LOG_DIR/iter${i}_fix_prompt.txt)" \
        2>&1 | tee $LOG_DIR/iter${i}_fixes.txt
    
    # =========================================
    # STEP 6: Clean and re-run pipeline
    # =========================================
    echo "[Step 6] Cleaning outputs and re-running pipeline..."
    
    # Clean specific subject outputs only
    rm -rf work/${FOCUS}_pipeline/${SUBJECT}/${SESSION}/ 2>/dev/null || true
    rm -rf test_outputs/${FOCUS}_pipeline/${SUBJECT}/ 2>/dev/null || true
    
    # Get CLI options for this subject
    source $PROJECT_DIR/subject_options.sh
    get_cli_options $SUBJECT $SESSION
    
    # Run pipeline
    echo "Running: pmriprep $PERF_PLUS ./test_outputs participant \\"
    echo "  --participant-label ${SUBJECT#sub-} --session-label ${SESSION#ses-} \\"
    echo "  $DSC_OPTS $DCE_OPTS -w ./work"
    
    pmriprep $PERF_PLUS ./test_outputs participant \
        --participant-label ${SUBJECT#sub-} \
        --session-label ${SESSION#ses-} \
        $DSC_OPTS $DCE_OPTS \
        -w ./work \
        2>&1 | tee $LOG_DIR/iter${i}_pipeline.txt
    
    echo "[Iteration $i complete]"
    echo ""
done

echo "Max iterations ($MAX_ITERATIONS) reached without passing all tests."
exit 1
