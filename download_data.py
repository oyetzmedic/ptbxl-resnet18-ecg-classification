"""Download PTB-XL (100 Hz records + metadata) into ./ptbxl.
Preferred: python3 download_data.py   (uses wfdb's PhysioNet downloader)
Fallback (manual): download and unzip from
  https://physionet.org/content/ptb-xl/1.0.3/   (choose the ZIP; ~1.7 GB)
then ensure ptbxl_database.csv, scp_statements.csv and records100/ sit in ./ptbxl
"""
import os
try:
    import wfdb
    os.makedirs("ptbxl", exist_ok=True)
    print("Downloading PTB-XL via wfdb (metadata + 100 Hz records)...")
    wfdb.dl_database("ptb-xl", dl_dir="ptbxl")
    print("Done. Verify ptbxl/ptbxl_database.csv exists.")
except Exception as e:
    print("Automatic download failed:", e)
    print(__doc__)
