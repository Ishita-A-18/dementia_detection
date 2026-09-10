"""
Converts DementiaBank / ADReSS CHAT-format (.cha) transcripts into plain
.txt files containing only the participant's (PAR) speech -- ready to feed
into dementia_detection_pipeline.py's text feature extractor.

Requires: pip install pylangacq --break-system-packages

Usage:
    python cha_to_txt.py --input_dir ./raw_cha/AD --output_dir ./data/transcripts/AD
    python cha_to_txt.py --input_dir ./raw_cha/HC --output_dir ./data/transcripts/HC
"""
import argparse
from pathlib import Path


def convert_folder(input_dir: str, output_dir: str):
    import pylangacq

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cha_files = list(input_dir.glob("*.cha"))
    if not cha_files:
        print(f"No .cha files found in {input_dir}")
        return

    for f in cha_files:
        try:
            reader = pylangacq.read_chat(str(f))
            # PAR = participant (the person being assessed), as opposed to
            # INV = investigator/interviewer, which we deliberately exclude.
            words = reader.words(participants="PAR")
            text = " ".join(words)
            out_path = output_dir / (f.stem + ".txt")
            out_path.write_text(text, encoding="utf-8")
            print(f"  {f.name} -> {out_path.name} ({len(words)} words)")
        except Exception as e:
            print(f"  [warn] failed to convert {f.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder of .cha files")
    parser.add_argument("--output_dir", required=True, help="Where to save .txt files")
    args = parser.parse_args()
    convert_folder(args.input_dir, args.output_dir)
