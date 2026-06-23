#!/usr/bin/env bash
#SBATCH --job-name=aihub537-merge-call
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/users/voice/zoey/todak/impl/EdgeSpot}"
PART_ROOT="${PART_ROOT:-/data/datasets/voice/aihub_537/full}"
OUT_ROOT="${OUT_ROOT:-/data/datasets/voice/aihub_537/call_only}"

cd "${PROJECT_DIR}"
mkdir -p logs

merge_one() {
  local rel_dir="$1"
  local name="$2"
  local src_dir="${PART_ROOT}/${rel_dir}"
  local dst_dir="${OUT_ROOT}/${rel_dir}"
  local dst="${dst_dir}/${name}"

  mkdir -p "${dst_dir}"
  if [[ -s "${dst}" ]]; then
    echo "skip existing ${dst}"
    return
  fi

  echo "merge ${src_dir}/${name}.part* -> ${dst}"
  python - "$src_dir" "$name" "$dst" <<'PY'
from pathlib import Path
import sys

src_dir = Path(sys.argv[1])
name = sys.argv[2]
dst = Path(sys.argv[3])
parts = sorted(
    src_dir.glob(name + ".part*"),
    key=lambda p: int(p.name.rsplit(".part", 1)[1]),
)
if not parts:
    raise SystemExit(f"no parts found for {src_dir / (name + '.part*')}")
expected = 0
with dst.open("wb") as out:
    for part in parts:
        offset = int(part.name.rsplit(".part", 1)[1])
        if offset != expected:
            raise SystemExit(f"missing part before {part}: expected offset {expected}, got {offset}")
        data = part.read_bytes()
        out.write(data)
        expected += len(data)
print(f"wrote {dst} bytes={expected} parts={len(parts)}")
PY
}

TRAIN_SRC="003.화자_인식용_음성_데이터/01.데이터/1.Training/원천데이터"
VAL_SRC="003.화자_인식용_음성_데이터/01.데이터/2.Validation/원천데이터"
VAL_LABEL="003.화자_인식용_음성_데이터/01.데이터/2.Validation/라벨링데이터"

merge_one "${TRAIN_SRC}" "TS_call_02.zip"
merge_one "${TRAIN_SRC}" "TS_call_03.zip"
merge_one "${VAL_LABEL}" "VL_call_01.zip"
merge_one "${VAL_SRC}" "VS_call_01.zip"

du -sh "${OUT_ROOT}"
