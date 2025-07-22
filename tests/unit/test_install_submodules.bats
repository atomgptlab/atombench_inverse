#!/usr/bin/env bats

load 'test_helper/bats-support/load'
load 'test_helper/bats-assert/load'

REPO_ROOT="${BATS_TEST_DIRNAME}/../.."

@test "submodules can be initialized and updated" {
  run git -C "$REPO_ROOT" submodule update --init --recursive
  assert_success

  run git -C "$REPO_ROOT" submodule status
  echo "$output" | grep --quietE '^[+-]' && \
    fail "Submodule update did not produce a clean state:\n$output"
}

@test "expected submodule directories exist" {
  # replace these with the actual paths your repo uses
  dirs=(
    "models/atomgpt"
    "models/cdvae"
    "models/flowmm"
  )

  for d in "${dirs[@]}"; do
    [ -d "$REPO_ROOT/$d" ]
    assert_success "Directory $d not found; submodule may not have been initialized"
  done
}
