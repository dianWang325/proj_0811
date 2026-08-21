#!/usr/bin/env bash

set -u
set -o pipefail

REPO_ROOT="${REPO_ROOT:-/home/w00985415/proj_0811/deps/vllm-ascend}"
LOCAL_CI_ROOT="${LOCAL_CI_ROOT:-/home/w00985415/proj_0811/local_CI}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${LOCAL_CI_ROOT}/results/${RUN_ID}"
ALIAS_ROOT="${LOCAL_CI_ROOT}/model_aliases"
MAIN_MODEL_PATH="${MAIN_MODEL_PATH:-/mnt/weight/Qwen3-8B}"
DSPARK_MODEL_PATH="${DSPARK_MODEL_PATH:-}"
NPU_DEVICE="${NPU_DEVICE:-4}"

TEST_FILE="${REPO_ROOT}/tests/e2e/pull_request/one_card/model_runner_v2/test_basic.py"
TEST_FUNCTION="${TEST_FILE}::test_dspark_spec_decoding"
TARGET_PREFIX="::test_dspark_spec_decoding[default_full_and_piecewise-"
TARGET_MODELS="deepseek-ai/dspark_qwen3_8b_block7-Qwen/Qwen3-8B]"

mkdir -p \
  "${LOCAL_CI_ROOT}/scripts" \
  "${LOCAL_CI_ROOT}/model_aliases/Qwen" \
  "${LOCAL_CI_ROOT}/model_aliases/deepseek-ai" \
  "${RUN_DIR}/cache" \
  "${RUN_DIR}/tmp" \
  "${RUN_DIR}/torchinductor"

echo "RUN_DIR=${RUN_DIR}"

finalize() {
  git -C "${REPO_ROOT}" status --porcelain=v1 >"${RUN_DIR}/git_status_after.txt" 2>&1 || true
  npu-smi info >"${RUN_DIR}/npu_after.txt" 2>&1 || true
  if diff -u "${RUN_DIR}/git_status_before.txt" "${RUN_DIR}/git_status_after.txt" \
    >"${RUN_DIR}/git_status_diff.txt"; then
    echo "SOURCE_WORKTREE_UNCHANGED=1" >>"${RUN_DIR}/summary.txt"
  else
    echo "SOURCE_WORKTREE_UNCHANGED=0" >>"${RUN_DIR}/summary.txt"
  fi
}
trap finalize EXIT

git -C "${REPO_ROOT}" status --porcelain=v1 >"${RUN_DIR}/git_status_before.txt"
npu-smi info >"${RUN_DIR}/npu_before.txt" 2>&1 || true

{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "repository=${REPO_ROOT}"
  echo "branch=$(git -C "${REPO_ROOT}" branch --show-current)"
  echo "head=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  echo "npu_device=${NPU_DEVICE}"
  echo "main_model_id=Qwen/Qwen3-8B"
  echo "main_model_path=${MAIN_MODEL_PATH}"
  echo "dspark_model_id=deepseek-ai/dspark_qwen3_8b_block7"
  echo "dspark_model_path=${DSPARK_MODEL_PATH:-UNRESOLVED}"
  echo "max_tokens=32"
  echo "enforce_eager=False"
  echo "compilation_config={}"
  python --version
} >"${RUN_DIR}/environment.txt" 2>&1

if [[ -s "${RUN_DIR}/git_status_before.txt" ]]; then
  echo "status=blocked_dirty_source_worktree" >"${RUN_DIR}/summary.txt"
  echo "exit_code=2" >"${RUN_DIR}/exit_code.txt"
  exit 2
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="${RUN_DIR}/cache"
export TORCHINDUCTOR_CACHE_DIR="${RUN_DIR}/torchinductor"
export TMPDIR="${RUN_DIR}/tmp"

cd "${ALIAS_ROOT}"
python -m pytest --collect-only -q -p no:cacheprovider "${TEST_FUNCTION}" \
  >"${RUN_DIR}/collect.log" 2>&1
collect_rc=$?
echo "collect_exit_code=${collect_rc}" >>"${RUN_DIR}/environment.txt"
if [[ ${collect_rc} -ne 0 ]]; then
  echo "status=blocked_collection_failed" >"${RUN_DIR}/summary.txt"
  echo "exit_code=${collect_rc}" >"${RUN_DIR}/exit_code.txt"
  exit "${collect_rc}"
fi

mapfile -t target_nodes < <(
  grep -F "${TARGET_PREFIX}" "${RUN_DIR}/collect.log" \
    | grep -F "${TARGET_MODELS}"
)
if [[ ${#target_nodes[@]} -ne 1 ]]; then
  echo "status=blocked_target_selection_count_${#target_nodes[@]}" >"${RUN_DIR}/summary.txt"
  echo "exit_code=4" >"${RUN_DIR}/exit_code.txt"
  exit 4
fi

collected_node="${target_nodes[0]}"
parameter_suffix="${collected_node#*::test_dspark_spec_decoding}"
target_node="${TEST_FUNCTION}${parameter_suffix}"
echo "target_node=${target_node}" >>"${RUN_DIR}/environment.txt"

if [[ ! -d "${MAIN_MODEL_PATH}" ]]; then
  echo "status=blocked_missing_main_model" >"${RUN_DIR}/summary.txt"
  echo "missing_path=${MAIN_MODEL_PATH}" >>"${RUN_DIR}/summary.txt"
  echo "exit_code=3" >"${RUN_DIR}/exit_code.txt"
  exit 3
fi

if [[ -z "${DSPARK_MODEL_PATH}" ]]; then
  for candidate in \
    /mnt/weight/dspark_qwen3_8b_block7-Qwen \
    /mnt/weight/dspark_qwen3_8b_block7 \
    /mnt/weight/deepseek-ai/dspark_qwen3_8b_block7; do
    if [[ -d "${candidate}" ]]; then
      DSPARK_MODEL_PATH="${candidate}"
      break
    fi
  done
fi

if [[ -z "${DSPARK_MODEL_PATH}" || ! -d "${DSPARK_MODEL_PATH}" ]]; then
  echo "status=blocked_missing_dspark_model" >"${RUN_DIR}/summary.txt"
  echo "expected_model_id=deepseek-ai/dspark_qwen3_8b_block7" >>"${RUN_DIR}/summary.txt"
  echo "exit_code=3" >"${RUN_DIR}/exit_code.txt"
  exit 3
fi

main_alias="${ALIAS_ROOT}/Qwen/Qwen3-8B"
dspark_alias="${ALIAS_ROOT}/deepseek-ai/dspark_qwen3_8b_block7"

if [[ ! -e "${main_alias}" ]]; then
  ln -s "${MAIN_MODEL_PATH}" "${main_alias}"
fi
if [[ ! -e "${dspark_alias}" ]]; then
  ln -s "${DSPARK_MODEL_PATH}" "${dspark_alias}"
fi

echo "resolved_dspark_model_path=${DSPARK_MODEL_PATH}" >>"${RUN_DIR}/environment.txt"

export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICE}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

set +e
python -m pytest -sv -p no:cacheprovider "${target_node}" \
  2>&1 | tee "${RUN_DIR}/pytest.log"
test_rc=${PIPESTATUS[0]}
set -e

echo "exit_code=${test_rc}" >"${RUN_DIR}/exit_code.txt"
if [[ ${test_rc} -eq 0 ]]; then
  echo "status=passed" >"${RUN_DIR}/summary.txt"
else
  echo "status=failed" >"${RUN_DIR}/summary.txt"
fi
echo "pytest_exit_code=${test_rc}" >>"${RUN_DIR}/summary.txt"

exit "${test_rc}"
