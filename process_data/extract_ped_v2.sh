#!/usr/bin/env bash
# Put all ped_v2 scenes (batch 1 folder + batch 2 tars) into one folder: ped_v2/raw/
# Safe to re-run: finished tars are skipped (.done markers), tars are never deleted.
#
# Usage: bash process_data/extract_ped_v2.sh
set -euo pipefail

ROOT=/datahdd2/karthik/ped_v2
B1=$ROOT/pedestrian_two_ped_v4_500
TARS=$ROOT/pedestrian_two_ped_v4_batch2_tars
RAW=$ROOT/raw
META=$RAW/_batch_meta
DONE=$RAW/_batch_meta/extract_done

mkdir -p "$RAW" "$META/v4_500" "$META/v4_batch2" "$DONE"

# --- Batch 2: index tar (top-level files) -> _batch_meta/v4_batch2 ---
if [ ! -f "$DONE/index.done" ]; then
    tar -xf "$TARS/pedestrian_two_ped_v4_batch2_index.tar" -C "$META/v4_batch2" --strip-components=1
    touch "$DONE/index.done"
fi

# --- Batch 2: scene tars -> raw/ ---
for t in "$TARS"/pedestrian_two_ped_v4_batch2_part_*.tar; do
    name=$(basename "$t" .tar)
    part=${name##*_part_}
    [ -f "$DONE/part_$part.done" ] && { echo "skip $name"; continue; }
    echo "$(date +%T) extracting $name"
    tar -xf "$t" -C "$RAW" --strip-components=1
    # verify every scene in this part's list landed
    missing=0
    while read -r scene; do
        [ -z "$scene" ] && continue
        [ -d "$RAW/$scene" ] || { echo "  MISSING $scene"; missing=$((missing+1)); }
    done < "$TARS/lists/part_$part.txt"
    [ "$missing" -eq 0 ] && touch "$DONE/part_$part.done" || echo "  $missing missing in $name"
done

# --- Batch 1: move scene folders + keep top-level files ---
if [ -d "$B1" ]; then
    find "$B1" -maxdepth 1 -type f -exec cp -n {} "$META/v4_500/" \;
    n=0
    for d in "$B1"/*/; do
        s=$(basename "$d")
        [ -e "$RAW/$s" ] && { echo "  EXISTS already: $s (left in place)"; continue; }
        mv "$d" "$RAW/$s"; n=$((n+1))
    done
    echo "moved $n batch-1 scenes"
fi

# --- Final check ---
total=$(find "$RAW" -mindepth 1 -maxdepth 1 -type d ! -name _batch_meta | wc -l)
nodone=$(find "$RAW" -mindepth 1 -maxdepth 1 -type d ! -name _batch_meta \
         ! -exec test -f '{}/rendered/_DONE' \; -print | wc -l)
echo "scene folders: $total (expected 9890), missing rendered/_DONE: $nodone"
for b in v4_500 v4_batch2; do
    tail -n +2 "$META/$b/manifest.csv" | cut -d, -f1 | while read -r s; do
        [ -d "$RAW/$s" ] || echo "  $b manifest scene not in raw: $s"
    done | head -20
done
echo "$(date +%T) ALL DONE"
