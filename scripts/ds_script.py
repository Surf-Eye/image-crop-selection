import boto3
import re
from collections import defaultdict
import logging
import sys
from datetime import datetime
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def parse_existing_commands(file_path):
    """Parse existing commands file and group by video ID from source filenames"""
    logger = logging.getLogger('parser')
    existing_videos = defaultdict(dict)
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # Split command components
            parts = line.split()
            if len(parts) < 5:
                logger.warning(f"Skipping invalid line {line_num}: {line}")
                continue

            source = parts[3]
            dest = parts[4]

            # Parse source filename
            source_parts = source.split('/')
            src_filename = source_parts[-1]
            src_match = re.match(
                r'^(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})_(\d+)_(\d+)\.png$',
                src_filename
            )
            if not src_match:
                logger.warning(f"Skipping invalid source filename on line {line_num}: {src_filename}")
                continue

            video_id = src_match.group(2)
            try:
                crop_index = int(src_match.group(3))
            except ValueError:
                logger.warning(f"Invalid crop index on line {line_num}: {src_filename}")
                continue

            # Parse destination filename
            dest_match = re.fullmatch(r'(.+?)_(\d{7})_(\d+)\.png', dest)
            if not dest_match:
                logger.warning(f"Skipping invalid destination format on line {line_num}: {dest}")
                continue

            static_part = dest_match.group(1)
            ending_number = dest_match.group(3)
            s3_prefix = '/'.join(source_parts[:-1]) + '/'

            # Update video information
            if not existing_videos.get(video_id):
                existing_videos[video_id] = {
                    's3_prefix': s3_prefix,
                    'static_part': static_part,
                    'ending_number': ending_number,
                    'existing_indices': set(),
                    'source_example': src_filename
                }
                logger.info(f"New video {video_id} found (line {line_num})")
            else:
                # Verify consistency across entries
                current = existing_videos[video_id]
                if (current['static_part'] != static_part or 
                    current['ending_number'] != ending_number):
                    logger.warning(
                        f"Inconsistent destination format for video {video_id} "
                        f"on line {line_num}\n"
                        f"Existing: {current['static_part']}_..._{current['ending_number']}\n"
                        f"New: {static_part}_..._{ending_number}"
                    )

            existing_videos[video_id]['existing_indices'].add(crop_index)

    logger.info(f"Parsing complete. Found {len(existing_videos)} unique videos")
    return existing_videos

def process_single_video(video_id, data, max_missing=3):
    """Process one video: list S3 objects and find missing crops"""
    logger = logging.getLogger('finder')
    s3 = boto3.client('s3')
    missing_for_video = []
    found = 0

    s3_prefix = data['s3_prefix']
    if not s3_prefix.startswith('s3://'):
        logger.error(f"Invalid S3 prefix for video {video_id}: {s3_prefix}")
        return []

    s3_path = s3_prefix[5:]  # Remove 's3://'
    bucket_end = s3_path.find('/')
    bucket = s3_path[:bucket_end] if bucket_end != -1 else s3_path
    prefix = s3_path[bucket_end+1:] if bucket_end != -1 else ''

    try:
        logger.debug(f"Listing objects in s3://{bucket}/{prefix}")
        start_time = datetime.now()
        paginator = s3.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)

        for page in page_iterator:
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                key = obj['Key']
                filename = key.split('/')[-1]

                src_match = re.match(
                    r'^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_(\d+)_(\d+)\.png$',
                    filename
                )
                if not src_match:
                    continue

                current_vid = src_match.group(1)
                if current_vid != video_id:
                    continue

                try:
                    crop_num = int(src_match.group(2))
                except ValueError:
                    continue

                if crop_num not in data['existing_indices']:
                    padded_index = f"{crop_num:07d}"
                    dest_filename = f"{data['static_part']}_{padded_index}_{data['ending_number']}.png"
                    cmd = f"aws s3 cp s3://{bucket}/{key} {dest_filename}"
                    missing_for_video.append(cmd)
                    found += 1
                    if found >= max_missing:
                        logger.debug(f"Found {max_missing} missing crops for video {video_id}, stopping search")
                        break
            if found >= max_missing:
                break

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Video {video_id} processed in {elapsed:.2f}s. Found {len(missing_for_video)} missing crops")

    except Exception as e:
        logger.error(f"Error processing video {video_id}: {str(e)}")
        return []

    return missing_for_video

def find_missing_commands(existing_videos, max_missing=3, max_workers=8):
    """Find missing crops concurrently across videos"""
    logger = logging.getLogger('finder')
    missing_commands = []
    total_videos = len(existing_videos)
    processed_count = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_video, video_id, data, max_missing): video_id
            for video_id, data in existing_videos.items()
        }

        for future in as_completed(futures):
            video_id = futures[future]
            try:
                missing_for_video = future.result()
                missing_commands.extend(missing_for_video)
            except Exception as e:
                logger.error(f"Error processing video {video_id} in thread: {e}")

            # Thread-safe update and log progress
            with lock:
                processed_count += 1
                logger.info(f"Processed {processed_count}/{total_videos} videos")

    logger.info(f"Total missing commands found: {len(missing_commands)}")
    return missing_commands

def main():
    """Main execution flow"""
    parser = argparse.ArgumentParser(description='Find missing AWS S3 crop download commands.')
    parser.add_argument('--input', required=True, help='Input file path with existing commands')
    parser.add_argument('--output', required=True, help='Output file path for missing commands')
    args = parser.parse_args()

    logger = logging.getLogger('main')
    logger.info("Starting processing")
    
    existing_commands_file = args.input
    destination_file = args.output

    # Parse existing commands
    logger.info("Phase 1: Parsing existing commands")
    existing_videos = parse_existing_commands(existing_commands_file)

    # Find missing commands
    logger.info("\nPhase 2: Searching for missing crops in S3")
    missing_commands = find_missing_commands(existing_videos)

    # Write results
    logger.info("\nPhase 3: Writing results")
    with open(destination_file, 'w') as f:
        for cmd in missing_commands:
            f.write(f"{cmd}\n")
    
    logger.info(f"Successfully wrote {len(missing_commands)} commands to {destination_file}")

if __name__ == "__main__":
    main()
