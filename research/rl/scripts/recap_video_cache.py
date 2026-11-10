"""Build pixel-identical short-GOP videos for random-access RECAP training."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def canonical_frames(text):
    return [line.strip() for line in text.splitlines()
            if line.startswith('#tb ') or (line.strip() and not line.startswith('#'))]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def activate_cache(cache, run):
    marker = cache/'verified.json'
    if not marker.is_file() or not json.loads(marker.read_text()).get('verified'):
        raise RuntimeError('Refusing unverified video cache')
    pairs = [(run/'data'/name/'videos', cache/f'libero10_task0_{split}/videos')
             for name, split in [('rollout','train'), ('eval','eval')]]
    for link, target in pairs:
        if not target.is_dir():
            raise RuntimeError(f'Missing video cache split: {target}')
        if link.exists() and not link.is_symlink():
            raise RuntimeError(f'Refusing to replace real data directory: {link}')
    for link, target in pairs:
        temporary = link.with_name('videos.cache-link')
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(target.resolve(), target_is_directory=True)
        temporary.replace(link)


def convert_one(ffmpeg, source, dest, source_root, reserve_bytes):
    relative = source.relative_to(source_root)
    output = dest/relative
    receipt = dest/'.receipts'/relative.with_suffix('.json')
    stat = source.stat()
    if receipt.exists() and output.exists():
        old = json.loads(receipt.read_text())
        if (old['source_bytes'] == stat.st_size and old['source_mtime_ns'] == stat.st_mtime_ns
                and old['output_bytes'] == output.stat().st_size
                and old['output_sha256'] == file_hash(output)):
            return old
    if shutil.disk_usage(dest).free < reserve_bytes:
        raise RuntimeError('Video cache reached reserved checkpoint disk space')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.partial.mp4')
    receipt.unlink(missing_ok=True)
    base = [ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-threads','1','-filter_threads','1']
    subprocess.run([*base,'-y','-i',str(source),'-map','0:v:0','-an','-c:v','libx264rgb',
                    '-qp','0','-preset','veryfast','-g','8','-keyint_min','8','-sc_threshold','0',
                    '-threads','1','-pix_fmt','rgb24','-fps_mode','passthrough',str(temporary)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    def frames(path):
        return canonical_frames(subprocess.check_output(
            [*base,'-i',str(path),'-map','0:v:0','-pix_fmt','rgb24','-threads','1','-f','framemd5','-'],
            text=True, stderr=subprocess.PIPE))
    original, converted = frames(source), frames(temporary)
    if not original or original != converted:
        raise RuntimeError(f'Pixel/timestamp mismatch: {relative}')
    temporary.replace(output)
    result = {'path':str(relative),'source_bytes':stat.st_size,'source_mtime_ns':stat.st_mtime_ns,
              'output_bytes':output.stat().st_size,'output_sha256':file_hash(output),
              'frames':sum(not x.startswith('#') for x in original),
              'frames_sha256':hashlib.sha256('\n'.join(original).encode()).hexdigest(),
              'pixels_and_timestamps_identical':True}
    atomic_json(receipt,result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--dest',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--reserve-gib',type=float,default=180)
    p.add_argument('--activate-run',type=Path)
    a=p.parse_args();source=a.source.resolve();dest=a.dest.resolve()
    if source == dest or dest.is_relative_to(source):
        raise ValueError('Cache must be separate from raw data')
    if not (source/'verified.json').is_file():
        raise RuntimeError('Verified source dataset required')
    if a.workers <= 0 or a.reserve_gib < 0:
        raise ValueError('Workers must be positive and disk reserve nonnegative')
    dest.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(dest/'cache.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    videos=list(source.glob('libero10_task0_*/videos/**/*.mp4'))
    if not videos:
        raise RuntimeError('No source videos found')
    # Complete both cameras of early episodes first, making small diagnostic views possible.
    videos.sort(key=lambda x:(int(x.stem.split('_')[-1]),str(x)))
    import imageio_ffmpeg
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    started=time.time();records=[];total_source=sum(x.stat().st_size for x in videos)
    (dest/'verified.json').unlink(missing_ok=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(convert_one,ffmpeg,x,dest,source,int(a.reserve_gib*1024**3)) for x in videos]
        try:
            for future in as_completed(futures):
                records.append(future.result())
                if len(records)%32==0 or len(records)==len(videos):
                    progress={'files':len(records),'total_files':len(videos),
                              'source_bytes_done':sum(x['source_bytes'] for x in records),
                              'total_source_bytes':total_source,'cache_bytes':sum(x['output_bytes'] for x in records),
                              'seconds':time.time()-started}
                    atomic_json(dest/'progress.json',progress);print(json.dumps(progress),flush=True)
        except BaseException:
            for future in futures:future.cancel()
            raise
    summary={'verified':True,'files':len(records),'frames':sum(x['frames'] for x in records),
             'gop':8,'codec':'libx264rgb','qp':0,'source':str(source),
             'source_receipt_sha256':file_hash(source/'verified.json'),
             'pixels_and_timestamps_identical':True,'cache_bytes':sum(x['output_bytes'] for x in records),
             'seconds':time.time()-started}
    atomic_json(dest/'verified.json',summary)
    if a.activate_run:activate_cache(dest,a.activate_run)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
