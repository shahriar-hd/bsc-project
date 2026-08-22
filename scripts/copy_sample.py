import os
from pathlib import Path
import pandas as pd


def create_test_video_symlinks(
    csv_path: str = "data/datasets/processed/csv/master.csv",
    output_dir: str = "data/sample",
):
    """Filter unique test videos from master.csv and create symbolic links

    named by their labels (e.g., real_1.mp4, spoof_2.mov) in the target directory.
    """
    csv_file = Path(csv_path)
    dst_dir = Path(output_dir)

    if not csv_file.exists():
        print(f"[ERROR] CSV file not found at: {csv_file}")
        return

    # Create target directory if it does not exist
    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading metadata from: {csv_file} ...")
    df = pd.read_csv(csv_file)

    # 1. Filter for split == 'test'
    test_df = df[df["split"].str.strip().str.lower() == "test"]

    if test_df.empty:
        print("[WARNING] No records found with split == 'test'.")
        return

    # 2. Drop duplicate frame entries to keep unique videos
    # We only need video_path and label (or spoof_type if needed)
    unique_videos = test_df.drop_duplicates(subset=["video_path"])[
        ["video_path", "label"]
    ]

    print(f"Found {len(unique_videos)} unique test video(s). Creating symlinks...")

    # Dictionary to keep track of counters for each label (e.g., label_counters['real'] = 1, 2, ...)
    label_counters = {}
    created_count = 0
    skipped_count = 0

    for _, row in unique_videos.iterrows():
        src_video_path = Path(str(row["video_path"]).strip())
        label = (
            str(row["label"]).strip().lower().replace(" ", "_")
        )  # Sanitize label name

        # Check if source video actually exists
        if not src_video_path.exists():
            print(
                f"[WARNING] Source video file does not exist on disk: {src_video_path}"
            )
            skipped_count += 1
            continue

        # Increment counter for this label
        label_counters[label] = label_counters.get(label, 0) + 1
        idx = label_counters[label]

        # Extract file extension (e.g., .mp4, .mov, .avi)
        extension = src_video_path.suffix

        # Destination filename format: e.g., real_1.mp4 or spoof_2.mov
        # If you prefer 'real1.mp4' instead of 'real_1.mp4', change to: f"{label}{idx}{extension}"
        link_name = f"{label}_{idx}{extension}"
        link_path = dst_dir / link_name

        # Remove existing symlink/file if it already exists to prevent FileExistsError
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()

        try:
            # Create symbolic link pointing to the absolute path of the original video
            link_path.symlink_to(src_video_path.resolve())
            created_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to create symlink for {src_video_path}: {e}")
            skipped_count += 1

    print("\n" + "=" * 45)
    print(f"Done! Summary:")
    print(f" - Successfully created symlinks: {created_count}")
    print(f" - Skipped / Errors: {skipped_count}")
    print(f" - Destination folder: {dst_dir.resolve()}")
    print("=" * 45)


if __name__ == "__main__":
    create_test_video_symlinks()