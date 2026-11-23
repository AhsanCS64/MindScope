"""Prepare the pinned OpenML v2 CSV without committing survey responses."""
import argparse
import csv
import hashlib
import io
import shlex
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data" / "raw" / "student_depression_openml_46874.csv"
SOURCE_URL = "https://openml.org/data/v1/download/22124437/student_depression_dataset.arff"
SOURCE_SHA256 = "4415c6196ddac7b0aaebd4321d23cc2a5cc3b94b55aff35550537600c91ce9ce"


def convert(source: bytes, destination: Path) -> None:
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("OpenML file does not match the pinned v2 snapshot.")
    header, data = source.decode("utf-8").split("@DATA", maxsplit=1)
    names = [shlex.split(line)[1] for line in header.splitlines() if line.startswith("@ATTRIBUTE ")]
    rows = list(csv.reader(io.StringIO(data.lstrip()), quotechar="'", escapechar="\\", skipinitialspace=True))
    if len(names) != 17 or len(rows) != 27901 or any(len(row) != len(names) for row in rows):
        raise ValueError("Unexpected OpenML dataset shape.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        writer.writerows(["" if value == "?" else value for value in row] for row in rows)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arff", type=Path, help="Use a verified local ARFF file.")
    args = parser.parse_args()
    if args.arff:
        source = args.arff.read_bytes()
    else:
        with urlopen(SOURCE_URL, timeout=45) as response:
            source = response.read(5_000_000)
    convert(source, DESTINATION)
    print(f"Prepared dataset at {DESTINATION.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
