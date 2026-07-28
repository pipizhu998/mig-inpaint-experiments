#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT=/home/pipizhu/workspace/experiment/7.23_new_experiment
CODE_ROOT=$EXPERIMENT_ROOT/code
TRANSFER_ROOT=$EXPERIMENT_ROOT/transfer
RUN_NAME=mig_worst_scale_vs_original_100_512
ARCHIVE=$TRANSFER_ROOT/$RUN_NAME.tar.gz
DOWNLOAD_PID_FILE=$TRANSFER_ROOT/download_512_gdrive.pid
DOWNLOAD_LOG=$TRANSFER_ROOT/download_512_gdrive.log
EXPECTED_BYTES=5767386296
EXPECTED_SHA256=b6be939114e39456abd5a71ca0fdab18b60b408a41992187cda6b8aafdc4bafc
EXPECTED_FILES=24916
STAGING_ROOT=$TRANSFER_ROOT/verified_extract_512
DESTINATION=$CODE_ROOT/runs/$RUN_NAME
REPORT=$TRANSFER_ROOT/integrity_512.txt

download_pid=$(cat "$DOWNLOAD_PID_FILE")
printf '[finalizer] waiting for Drive download PID %s\n' "$download_pid"
while kill -0 "$download_pid" 2>/dev/null; do
    sleep 20
done

if [[ ! -f "$ARCHIVE" ]]; then
    printf 'Download did not produce %s\n' "$ARCHIVE" >&2
    tail -n 50 "$DOWNLOAD_LOG" >&2 || true
    exit 1
fi

actual_bytes=$(stat -c '%s' "$ARCHIVE")
if [[ "$actual_bytes" != "$EXPECTED_BYTES" ]]; then
    printf 'Size mismatch: expected %s, got %s\n' \
        "$EXPECTED_BYTES" "$actual_bytes" >&2
    exit 1
fi

printf '%s  %s\n' "$EXPECTED_SHA256" "$ARCHIVE" | sha256sum -c -
gzip -t "$ARCHIVE"

mkdir -p "$STAGING_ROOT"
tar -xzf "$ARCHIVE" -C "$STAGING_ROOT"
archive_root=$(tar -tzf "$ARCHIVE" | awk -F/ 'NR == 1 {print $1}')
if [[ -z "$archive_root" || "$archive_root" == "." || "$archive_root" == ".." ]]; then
    printf 'Unsafe or empty archive root: %q\n' "$archive_root" >&2
    exit 1
fi
STAGING_RUN=$STAGING_ROOT/$archive_root
if [[ ! -d "$STAGING_RUN" ]]; then
    printf 'Archive root was not extracted: %s\n' "$STAGING_RUN" >&2
    exit 1
fi

actual_files=$(find "$STAGING_RUN" -type f | wc -l)
if [[ "$actual_files" != "$EXPECTED_FILES" ]]; then
    printf 'File-count mismatch: expected %s, got %s\n' \
        "$EXPECTED_FILES" "$actual_files" >&2
    exit 1
fi

mkdir -p "$DESTINATION"
rsync -a --checksum "$STAGING_RUN/" "$DESTINATION/"

differences=$(rsync -ani --checksum "$STAGING_RUN/" "$DESTINATION/")
if [[ -n "$differences" ]]; then
    printf 'Post-merge rsync verification failed:\n%s\n' "$differences" >&2
    exit 1
fi

{
    printf 'status=verified\n'
    printf 'run_name=%s\n' "$RUN_NAME"
    printf 'archive_bytes=%s\n' "$actual_bytes"
    printf 'archive_sha256=%s\n' "$EXPECTED_SHA256"
    printf 'archive_files=%s\n' "$actual_files"
    printf 'destination=%s\n' "$DESTINATION"
    printf 'verified_at=%s\n' "$(date --iso-8601=seconds)"
} > "$REPORT"

# Remove only the explicitly-created extraction tree and temporary OAuth config.
find "$STAGING_ROOT" -depth -delete
secret_dir=$(cat "$EXPERIMENT_ROOT/.remote512_rclone_secret_dir" 2>/dev/null || true)
if [[ -n "$secret_dir" && "$secret_dir" == "$EXPERIMENT_ROOT"/rclone-auth.* ]]; then
    find "$secret_dir" -depth -delete
fi
rm -f "$EXPERIMENT_ROOT/.remote512_rclone_secret_dir"

printf '[finalizer] verified and merged: %s\n' "$DESTINATION"
