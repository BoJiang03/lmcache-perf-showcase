#!/bin/bash
# Build the PRT images.
#   build.sh serving                         -> prt-serving:<vllm sha9>
#   build.sh lmcache <LMCache dir> <serving> -> prt-lmcache:<lmcache sha9>-v<vllm sha9>
#   build.sh mooncake <serving>              -> prt-mooncake:<mooncake sha9>-v<vllm sha9>
#   build.sh aiperf                          -> prt-aiperf:<sha9 of aiperf inputs>
# serving: vLLM is the current nightly unless VLLM_COMMIT (full sha) is set,
# e.g. to rebuild an older night. lmcache: built FROM the given serving image
# (tonight's or the last-good one); <LMCache dir> must be a clean git checkout
# at the commit under test, the tag is its HEAD. mooncake: the wheel of the
# newest Mooncake nightly.yml run (needs `gh` logged in), or of run
# MOONCAKE_RUN_ID if set. Prints the tag, then the image id.
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)

# Nightly wheel name carries "+g<sha9>"; the per-commit index needs the full sha.
resolve_nightly() {
  local short
  short=$(curl -fsS https://wheels.vllm.ai/nightly/cu130/vllm/ \
    | grep -oE 'vllm-[^"<>]*\+g[0-9a-f]+-[^"<>]*x86_64\.whl' | head -1 \
    | sed -E 's/.*\+g([0-9a-f]+)-.*/\1/')
  [ -n "$short" ] || { echo "cannot resolve vLLM nightly" >&2; exit 1; }
  curl -fsS "https://api.github.com/repos/vllm-project/vllm/commits/$short" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])'
}

# Newest run of Mooncake's nightly.yml that has the wheel artifact; prints
# "<run id> <commit> <created>". The wheel is built every night even when the
# nightly gate fails.
MC_REPO=kvcache-ai/Mooncake
MC_ARTIFACT=nightly-cuda13-x86-py312
resolve_mooncake() {
  local id sha t
  while read -r id sha t; do
    if gh api "repos/$MC_REPO/actions/runs/$id/artifacts" \
        --jq ".artifacts[] | select(.name == \"$MC_ARTIFACT\" and .expired == false) | .name" \
        | grep -q .; then
      echo "$id $sha $t"; return
    fi
  done < <(gh run list -R "$MC_REPO" -w nightly.yml -L 20 \
             --json databaseId,headSha,createdAt --jq '.[] | "\(.databaseId) \(.headSha) \(.createdAt)"')
  echo "no Mooncake nightly wheel in the last 20 runs" >&2; exit 1
}

serving_vllm_commit() {
  local c
  c=$(docker image inspect -f '{{index .Config.Labels "prt.vllm.commit"}}' "$1")
  [[ $c =~ ^[0-9a-f]{40}$ ]] || { echo "$1 has no prt.vllm.commit label" >&2; exit 1; }
  echo "$c"
}

case "${1:-}" in
  serving)
    VLLM_COMMIT=${VLLM_COMMIT:-$(resolve_nightly)}
    [[ $VLLM_COMMIT =~ ^[0-9a-f]{40}$ ]] || { echo "bad VLLM_COMMIT $VLLM_COMMIT" >&2; exit 1; }
    TAG=prt-serving:${VLLM_COMMIT:0:9}
    docker build --target serving --build-arg "VLLM_COMMIT=$VLLM_COMMIT" -t "$TAG" "$D"
    ;;
  lmcache)
    USAGE="usage: build.sh lmcache <LMCache dir> <serving image>"
    SRC=$(cd "${2:?$USAGE}" && pwd)
    SERVING=${3:?$USAGE}
    [ -z "$(git -C "$SRC" status --porcelain)" ] || { echo "dirty checkout: $SRC" >&2; exit 1; }
    SHA=$(git -C "$SRC" rev-parse HEAD)
    VLLM_COMMIT=$(serving_vllm_commit "$SERVING")
    TAG=prt-lmcache:${SHA:0:9}-v${VLLM_COMMIT:0:9}
    docker build --target lmcache --build-arg "SERVING_IMAGE=$SERVING" \
      --build-context "lmcache_src=$SRC" --label "prt.lmcache.commit=$SHA" -t "$TAG" "$D"
    ;;
  mooncake)
    SERVING=${2:?usage: build.sh mooncake <serving image>}
    VLLM_COMMIT=$(serving_vllm_commit "$SERVING")
    if [ -n "${MOONCAKE_RUN_ID:-}" ]; then
      read -r MC_SHA MC_TIME < <(gh run view -R "$MC_REPO" "$MOONCAKE_RUN_ID" \
        --json headSha,createdAt --jq '"\(.headSha) \(.createdAt)"')
    else
      read -r MOONCAKE_RUN_ID MC_SHA MC_TIME < <(resolve_mooncake)
    fi
    WHEEL_DIR=$(mktemp -d)
    trap 'rm -rf "$WHEEL_DIR"' EXIT
    gh run download "$MOONCAKE_RUN_ID" -R "$MC_REPO" -n "$MC_ARTIFACT" -D "$WHEEL_DIR"
    WHEEL=$(basename "$(ls "$WHEEL_DIR"/*.whl)")
    MC_VERSION=$(echo "$WHEEL" | cut -d- -f2)
    echo "Mooncake $MC_VERSION commit $MC_SHA (nightly run $MOONCAKE_RUN_ID, $MC_TIME)" >&2
    TAG=prt-mooncake:${MC_SHA:0:9}-v${VLLM_COMMIT:0:9}
    docker build --target mooncake --build-arg "SERVING_IMAGE=$SERVING" \
      --build-context "mooncake_wheel=$WHEEL_DIR" \
      --label "prt.mooncake.commit=$MC_SHA" --label "prt.mooncake.version=$MC_VERSION" \
      --label "prt.mooncake.built=$MC_TIME" -t "$TAG" "$D"
    ;;
  aiperf)
    P=$D/../aiperf_plugins
    SUM=$(cat "$D/aiperf-constraints.txt" $(find "$P" -type f -not -name '*.pyc' | sort) | sha256sum)
    TAG=prt-aiperf:${SUM:0:9}
    docker build --target aiperf --build-context "aiperf_plugins=$P" -t "$TAG" "$D"
    ;;
  *)
    echo "usage: build.sh serving | lmcache <LMCache dir> <serving image> | mooncake <serving image> | aiperf" >&2; exit 2;;
esac
echo "$TAG"
docker image inspect -f '{{.Id}}' "$TAG"
