"""Download the Grand River watershed streamflow (GRWS) datasets from Borealis
(Kemp, J. (2026), https://doi.org/10.5683/SP3/CH3T6Z); Doon and Leggatt are in the repo.

    python scripts/download_grws.py                  # Parquet files -> data/grws/Parquet/
    python scripts/download_grws.py --folders CSV    # or several folders, or all
    python scripts/download_grws.py --list           # show what is available

Files already present and intact are skipped. Standard library only.
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request

DOI = "doi:10.5683/SP3/CH3T6Z"
SERVER = "https://borealisdata.ca"
FILE_LIST_URL = f"{SERVER}/api/datasets/:persistentId/?persistentId={DOI}"
DOWNLOAD_URL = SERVER + "/api/access/datafile/{id}"


def fetch_file_list():
	with urllib.request.urlopen(FILE_LIST_URL, timeout=60) as response:
		version = json.load(response)["data"]["latestVersion"]
	files = []
	for entry in version["files"]:
		data_file = entry["dataFile"]
		files.append({
			"folder": entry.get("directoryLabel", ""),
			"name": data_file["filename"],
			"id": data_file["id"],
			"size": data_file.get("filesize"),
			"md5": data_file.get("md5") or data_file.get("checksum", {}).get("value"),
			"tabular": bool(data_file.get("tabularData")),
		})
	return version.get("versionNumber"), files


def md5_of(path):
	digest = hashlib.md5()
	with open(path, "rb") as f:
		for block in iter(lambda: f.read(1 << 20), b""):
			digest.update(block)
	return digest.hexdigest()


def download(entry, path):
	url = DOWNLOAD_URL.format(id=entry["id"])
	tmp = path + ".part"
	with urllib.request.urlopen(url, timeout=300) as response, open(tmp, "wb") as out:
		while True:
			block = response.read(1 << 20)
			if not block:
				break
			out.write(block)
	os.replace(tmp, path)


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--folders", default="Parquet",
		help="Comma-separated dataset folders to download (e.g. Parquet, CSV, Metadata) or 'all'. Default: Parquet.")
	parser.add_argument("--dest", default=os.path.join("data", "grws"),
		help="Destination; files go to <dest>/<folder>/<file>. Default: data/grws.")
	parser.add_argument("--list", action="store_true", help="List the available files and exit.")
	args = parser.parse_args()

	version, files = fetch_file_list()
	print(f"GRWS dataset {DOI}, version {version}: {len(files)} files")
	folders = sorted({f["folder"] for f in files})
	if args.list:
		for folder in folders:
			in_folder = [f for f in files if f["folder"] == folder]
			print(f"  {folder or '(top level)'}: {len(in_folder)} files, e.g. {', '.join(f['name'] for f in in_folder[:3])}")
		return

	wanted = folders if args.folders.strip().lower() == "all" else [x.strip() for x in args.folders.split(",")]
	unknown = [w for w in wanted if w not in folders]
	if unknown:
		sys.exit(f"Unknown folder(s) {unknown}; available: {', '.join(f or '(top level)' for f in folders)}")

	selected = [f for f in files if f["folder"] in wanted]
	done = skipped = failed = 0
	for entry in selected:
		target_dir = os.path.join(args.dest, entry["folder"])
		os.makedirs(target_dir, exist_ok=True)
		path = os.path.join(target_dir, entry["name"])
		if os.path.exists(path) and (entry["tabular"] or md5_of(path) == entry["md5"]):
			skipped += 1
			continue
		print(f"  downloading {entry['folder']}/{entry['name']} ({(entry['size'] or 0) / 1e6:.1f} MB)")
		try:
			download(entry, path)
		except OSError as e:
			print(f"    failed: {e}")
			failed += 1
			continue
		# tabular files are served as .tab, which doesn't match the upload's checksum
		if not entry["tabular"] and entry["md5"] and md5_of(path) != entry["md5"]:
			print(f"    checksum mismatch for {entry['name']}; file removed")
			os.remove(path)
			failed += 1
			continue
		done += 1
	print(f"Downloaded {done}, already present {skipped}, failed {failed}. Files are in {args.dest}/<folder>/.")
	print("Please cite the dataset (see data/README.md) if you use it.")
	if failed:
		sys.exit(1)


if __name__ == "__main__":
	main()
