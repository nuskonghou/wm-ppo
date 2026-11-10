#!/usr/bin/env bash
# Collect the released checkpoints into a staging directory (nothing is uploaded).
#   scripts/export_weights.sh OUT [LAYOUT]
# LAYOUT (default scripts/weights_layout.tsv): one line per component, "<destination under OUT> <source directory>";
# relative sources are relative to the repository. A third column "warmup-meta" marks a warm-up run directory: its
# state.json (iteration, task cursor, tokens, episodes at iteration 12) and the metrics.jsonl records of iterations
# 1-12 are written, which the fork scripts read. Otherwise only the LoRA adapters (adapter_config.json,
# adapter_model.safetensors) and the critic's value head (value_head.pt) are copied, keeping their sub-directories;
# base_model_name_or_path in every adapter_config.json is set to the Hugging Face id of the base model. Files the current
# user cannot read (the trainer containers write them as root, possibly with mode 600) are copied by a root container
# from a read-only bind mount and handed to the current user; the sources are never modified. Afterwards OUT holds
# SHA256SUMS and MANIFEST.tsv (path, bytes, sha256).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT=${1:?usage: export_weights.sh OUT [LAYOUT]}; LAYOUT=${2:-$ROOT/scripts/weights_layout.tsv}
BASE_ID=${WMPPO_BASE_ID:-Qwen/Qwen3-4B-Instruct-2507}
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
copy() {   # copy <src file> <dst file>
  local s=$1 d=$2; mkdir -p "$(dirname "$d")"
  if [ -r "$s" ]; then cp "$s" "$d"; else
    docker run --rm -v "$(dirname "$s"):/src:ro" -v "$(dirname "$d"):/dst" alpine:3.20 \
      sh -c "cp /src/$(basename "$s") /dst/$(basename "$d") && chown $(id -u):$(id -g) /dst/$(basename "$d")"
  fi
  chmod 644 "$d"
}
while read -r dst src kind; do
  [ -z "${dst:-}" ] || [ "${dst:0:1}" = "#" ] && continue
  [ "${src:0:1}" = "/" ] || src="$ROOT/$src"
  [ -d "$src" ] || { echo "missing source: $src" >&2; exit 1; }
  if [ "${kind:-}" = "warmup-meta" ]; then
    mkdir -p "$OUT/$dst"
    python3 - "$src" "$OUT/$dst" <<'PY'
import json, os, sys
src, dst = sys.argv[1:3]
recs = [json.loads(l) for l in open(os.path.join(src, 'metrics.jsonl'))]
recs = [r for r in recs if r['it'] <= 12]
st = json.load(open(os.path.join(src, 'state.json')))
if st['it'] == 12:
    state = {k: st[k] for k in ('it', 'cursor', 'tokens', 'episodes')}
else:
    r12 = [r for r in recs if r['it'] == 12][-1]
    state = {'it': 12, 'cursor': r12['episodes'], 'tokens': r12['tokens'], 'episodes': r12['episodes']}
json.dump(state, open(os.path.join(dst, 'state.json'), 'w'))
with open(os.path.join(dst, 'metrics.jsonl'), 'w') as f:
    for r in recs:
        f.write(json.dumps(r) + '\n')
PY
    echo "$dst <- $src (warm-up state and metrics)"; continue
  fi
  ( cd "$src" && find . -type f \( -name adapter_config.json -o -name adapter_model.safetensors -o -name value_head.pt \) ) | sort |
  while read -r f; do copy "$src/${f#./}" "$OUT/$dst/${f#./}"; done
  echo "$dst <- $src"
done < "$LAYOUT"
find "$OUT" -name adapter_config.json | while read -r f; do
  python3 - "$f" "$BASE_ID" <<'PY'
import json, sys
p, base = sys.argv[1], sys.argv[2]
c = json.load(open(p)); c['base_model_name_or_path'] = base
json.dump(c, open(p, 'w'), indent=2); open(p, 'a').write('\n')
PY
done
( cd "$OUT" && find . -type f ! -name SHA256SUMS ! -name MANIFEST.tsv | sed 's|^\./||' | sort | xargs -d '\n' sha256sum > SHA256SUMS
  { printf 'path\tbytes\tsha256\n'; while read -r h p; do printf '%s\t%s\t%s\n' "$p" "$(stat -c %s "$p")" "$h"; done < SHA256SUMS; } > MANIFEST.tsv )
echo "staged $(wc -l < "$OUT/SHA256SUMS") files, $(du -sh "$OUT" | cut -f1) in $OUT"
