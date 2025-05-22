import re
import os
import csv
import boto3
import shutil
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict
from boto3.s3.transfer import TransferConfig
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

re_PATTERN = r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_\d+)"

DATA_BASE_PATH = Path('_test_data')

class WavesDataParser:

    def __init__(
            self, 
            bucket_name='eu-north-1-data-management', 
            base_prefix = 'wavepools', # development/reid_outputs-2
            site_name = 'lakeside'
            ):
        
        self.s3_bucket = bucket_name
        self.base_prefix = base_prefix
        self.site_name = site_name

    def parse_session_csv(self, session_csv: Path, run_id=1):
        
        try:
            with open(str(session_csv), mode='r') as file:
                csv_reader = csv.reader(file)
                wave_names = []
                for idx, row in enumerate(csv_reader):
                    if idx!= 0:
                        path = row[8]
                        print(path)
                        parts = path.split('/')
                        # Extract the last part
                        last_part = parts[-1]
                        parts.append((last_part.rsplit('.', 1)[0]).replace("_high", ""))  
                        site_alias, camera_ref, yyyy_mm, dd, wave_naming = parts[1], parts[2], parts[4], parts[5], parts[8]
                        yyyy, mm = yyyy_mm.split('-'); mm = f"{int(mm):02d}"; dd = f"{int(dd):02d}"
                        prefix = f"{self.base_prefix}/{site_alias}/{camera_ref}/{yyyy}-{mm}/{dd}/crops/"
                        # print(prefix)
                        wave_names.append(wave_naming)
                    else:
                        try:
                            row.index('video')
                        except Exception:
                            print(f'CSV file not with the video header')
                # print(wave_names)
                session_id = session_csv.name.split('.')[0]
                save_dir = session_csv.parent.joinpath('sites', f'{site_alias}', f'{session_id}', 'img_crops')
                todays_date = datetime.now().strftime("%Y-%m-%d")
                
                save_dir = f'{str(DATA_BASE_PATH)}/{self.site_name}/session_crops/{todays_date}/runs/{run_id}'

                # print(prefix)
                self.download_matching_pngs(
                    bucket_name=self.s3_bucket,
                    prefix=prefix,
                    save_dir= save_dir,
                    wave_names=wave_names
                )
                # s3://eu-north-1-data-management/wavepools/citywave-vilnius-lt/citywlt_camera/2025-01/13/crops/2025-01-13-06-59-33_75871_0.png
                # organize the crops after 
                print()
                print('Organizing the crops')
                # self.parse_groups_data(save_dir, session_csv)

        except FileNotFoundError:
            print(f"The file '{session_csv}' does not exist.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def download_matching_pngs(self, bucket_name, prefix, save_dir, wave_names):
        if len(wave_names) == 0:
            return

        s3 = boto3.client('s3')
        os.makedirs(save_dir, exist_ok=True)
        continuation_token = None
        all_objects = []

        try:
            # Retrieve all objects matching the prefix.
            while True:
                params = {"Bucket": bucket_name, "Prefix": prefix}
                if continuation_token:
                    params["ContinuationToken"] = continuation_token

                response = s3.list_objects_v2(**params)
                if 'Contents' in response:
                    all_objects.extend(response['Contents'])
                if response.get('IsTruncated'):
                    continuation_token = response.get('NextContinuationToken')
                else:
                    break

            # Filter the objects: only PNG files that match any of the wave_names.
            keys_to_download = [
                obj['Key'] for obj in all_objects 
                if obj['Key'].endswith('.png') and any(wave in obj['Key'] for wave in wave_names)
            ]

            # Configure S3Transfer for high concurrency
            config = TransferConfig(
                max_concurrency=20,   
                use_threads=True
            )

            def download_file(key):
                file_name = os.path.basename(key)
                download_path = os.path.join(save_dir, file_name)
                if not os.path.exists(download_path):
                    try:
                        s3.download_file(bucket_name, key, download_path, Config=config)
                        print(f"Downloaded: {key} as {file_name}")
                    except Exception as e:
                        print(f"Error downloading {key}: {e}")
                else:
                    print(f"File exists: {file_name}")

            # Use a thread pool to download files concurrently.
            with ThreadPoolExecutor(max_workers=20) as executor:
                executor.map(download_file, keys_to_download)

        except Exception as e:
            print(f"An error occurred: {e}")

    def create_grouped_wave_session(self, df, grouped_files):
        df["associated_files"] = df["wave_prefix"].map(grouped_files)
        grp_df = df.groupby(by='group_id')
        return grp_df

    def create_wave_session_df(self, session_csv_path: Path):
        df = pd.read_csv(session_csv_path)
        df["wave_prefix"] = df["video"].str.extract(re_PATTERN)
        return df 

    def group_crop_imgs(self, img_crops_path: Path):
        grouped_files = defaultdict(list)
        pattern = re.compile(re_PATTERN)

        for img_path in img_crops_path.iterdir():
            match = pattern.search(str(img_path))
            if match:
                wave_prefix = match.group(1)
                grouped_files[wave_prefix].append(img_path)
        return grouped_files

    def parse_groups_data(self, crops_dir, session_data_csv):
        grouped_files = self.group_crop_imgs(Path(crops_dir))
        session_df = self.create_wave_session_df(session_data_csv)
        grouped_waves = self.create_grouped_wave_session(session_df, grouped_files)
    
        group_ids = [ idx[0] for idx in 
            grouped_waves['group_id'].unique().ravel().tolist()
            ]
        
        for grp_id in group_ids:
            grp_id_waves = grouped_waves.get_group(grp_id).apply(lambda x: x)
            try:
                self.parse_single_group(crops_dir, grp_id, grp_id_waves)
            except Exception as e:
                print(e)
        
        # clean up
        for obj in Path(crops_dir).iterdir():
            if not obj.is_dir():
                os.remove(obj)
                print(f'removed obj {obj}')
            
    def parse_single_group(self, crops_dir, grp_id, group_id_waves: pd.DataFrame):
        
        assert (group_id_waves['group_id'].values.ravel() == grp_id).all() == True, print(f'Ignoring group {grp_id}')
        grp_folder = Path(crops_dir).joinpath(str(grp_id))
        for idx in range(group_id_waves.shape[0]):
            wave_id = group_id_waves.iloc[idx]['id']
            wave_crops = group_id_waves.iloc[idx]['associated_files']
        
            crops_folder = grp_folder.joinpath(str(wave_id))
            crops_folder.mkdir(parents=True, exist_ok=True)

            for wave_crop in wave_crops:
                shutil.move(str(wave_crop), crops_folder)
            
    
    def parse_data(self, session_folder: Path, run_id=1):
        for obj in Path(session_folder).iterdir():
            if obj.name.endswith('.csv'): # obj.name in sessions_list:
                self.parse_session_csv(obj, run_id=run_id)
                # move the csv file to a new folder
                try:
                    todays_date = datetime.now().strftime("%Y-%m-%d")
                    parsed_folder = f'{str(DATA_BASE_PATH)}/{self.site_name}/sessions_csv/parsed_sessions/{todays_date}/runs/{run_id}'
                    os.makedirs(parsed_folder, exist_ok=True)
                    shutil.move(str(obj), parsed_folder)
                except Exception as e:
                    print(f'Error moving {obj} to {parsed_folder}: {e}')
                
            
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, default=1)
    parser.add_argument('--site_name', type=str, default='lakeside')
    parser.add_argument('--bucket_name', type=str, default='eu-north-1-data-management')
    parser.add_argument('--base_prefix', type=str, default='wavepools')
    args = parser.parse_args()

    data_parser = WavesDataParser(bucket_name=args.bucket_name, base_prefix=args.base_prefix, site_name=args.site_name)
    
    session_folder = f'{str(DATA_BASE_PATH)}/{args.site_name}/sessions_csv/recent'

    data_parser.parse_data(session_folder, args.run_id)

