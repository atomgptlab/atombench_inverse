#!/usr/bin/env bats

load 'test_helper/bats-support/load'
load 'test_helper/bats-assert/load'

REPO_ROOT="${BATS_TEST_DIRNAME}/../.."

@test "git submodules are initialized and up to date" {
  run git -C "$REPO_ROOT" submodule status
  # submodule status prints a leading " " for clean,
  # "-" if not initialized, "+" if out of sync.
  # We assert that no lines begin with "-" or "+".
  echo "$output" | grep --quietE '^[+-]' && \
    fail "Some submodules are missing or out of sync:\n$output"
}

@test "expected submodule directories exist" {
  # replace these with the actual paths your repo uses
  dirs=(
    "third_party/foo"
    "modules/bar"
    "external/baz"
  )

  for d in "${dirs[@]}"; do
    [ -d "$REPO_ROOT/$d" ]
    assert_success "Directory $d not found; submodule may not have been initialized"
  done
}

@test "build script runs without errors" {
  run bash "$REPO_ROOT/install.sh"
  assert_success
}

