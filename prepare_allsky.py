import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def safe_path(root, name):
    name = name.replace('\\', '/')
    parts = PurePosixPath(name)
    if parts.is_absolute() or '..' in parts.parts or any(':' in part for part in parts.parts):
        raise ValueError('Unsafe archive entry')
    target = root.joinpath(*parts.parts).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Archive entry leaves output directory')
    return target


def extract_month(archive, root, remaining):
    entries = sorted((entry for entry in archive.infolist() if not entry.is_dir()), key=lambda entry: entry.filename)
    selected = []
    for entry in entries:
        if Path(entry.filename).suffix.lower() in EXTENSIONS:
            selected.append(entry)
            if remaining and len(selected) >= remaining:
                break
    parents = {PurePosixPath(entry.filename.replace('\\', '/')).parent for entry in selected}
    records = [entry for entry in entries if entry.filename.lower().endswith('.csv')
               and PurePosixPath(entry.filename.replace('\\', '/')).parent in parents]
    for entry in selected + records:
        if (entry.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError('Archive symlinks are unsupported')
        target = safe_path(root, entry.filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(entry) as source, target.open('wb') as dest:
            shutil.copyfileobj(source, dest)
    return len(selected), len(records)


def main():
    parser = argparse.ArgumentParser(description='Extract monthly Allsky archives into local data')
    parser.add_argument('--archive', required=True)
    parser.add_argument('--output-dir', default='data/AllskyImagerData')
    parser.add_argument('--month', help='Optional YYYYMM month')
    parser.add_argument('--max-images', type=int, default=0, help='0 for all images')
    args = parser.parse_args()
    if args.max_images < 0:
        parser.error('max-images must be nonnegative')
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    count = records = 0
    with zipfile.ZipFile(args.archive) as outer:
        monthly = sorted((entry for entry in outer.infolist() if entry.filename.lower().endswith('.zip')), key=lambda entry: entry.filename)
        if args.month:
            monthly = [entry for entry in monthly if Path(entry.filename).stem == args.month]
            if not monthly:
                raise ValueError('Selected month is absent')
        if not monthly:
            count, records = extract_month(outer, root, args.max_images)
        for entry in monthly:
            # Stream one monthly ZIP to disk to avoid holding it in memory.
            with tempfile.TemporaryDirectory(dir=root) as temporary:
                nested = Path(temporary) / 'month.zip'
                with outer.open(entry) as source, nested.open('wb') as dest:
                    shutil.copyfileobj(source, dest)
                with zipfile.ZipFile(nested) as archive:
                    added, csv_count = extract_month(archive, root, args.max_images - count if args.max_images else 0)
                    count += added
                    records += csv_count
            if args.max_images and count >= args.max_images:
                break
    print(f'images={count} companion_csv={records}')


if __name__ == '__main__':
    main()
