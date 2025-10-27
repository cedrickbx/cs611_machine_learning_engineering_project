import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

# NYC Metro airports (our scope) with their USAF+WBAN identifier for NOAA data
target_stations = {
    "KJFK": "74486094789", #	John F. Kennedy Intl
    "KLGA": "72503014732", #	LaGuardia Intl
    "KEWR": "72502014734", #	Newark Liberty Intl
}

BUCKET_NAME = "noaa-global-hourly-pds"
LOCAL_BASE_DIR = "../data/weather_history"

#check if file exists before proceeding
def s3_key_exists(s3_client, bucket: str, key: str) -> bool:
    """Return True if object exists (and is readable), False if not."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        # 404/NoSuchKey => not found; 403 can also occur on public buckets for missing keys
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "Forbidden", "AccessDenied"):
            return False
        raise  # bubble up other unexpected errors

#download weather historical data from Amazon S3 bucket using boto3 API
def download_noaa_isd_data(start_year=2023, end_year=2025):
    """
    Download NOAA ISD data for specified stations and years from the public S3 bucket.

    - Creates a station folder only if at least one file for that station downloads successfully.
    - Records missing years per station in `missing_by_station`.
    """
    YEARS = range(start_year, end_year+1)
    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    print("Starting download of NOAA ISD data...")
    id_to_airport = {sid: ac for ac, sid in target_stations.items()}

    missing_by_station: dict[str, list[int]] = {}
    downloaded_counts: dict[str, int] = {}

    for airport_code, station_id in target_stations.items():
        success_count = 0
        missing_years: list[int] = []

        for year in YEARS:
            s3_key = f"{year}/{station_id}.csv"
            print(f"[{airport_code}] checking {s3_key} ... ", end="", flush=True)

            if not s3_key_exists(s3_client, BUCKET_NAME, s3_key):
                print("MISSING")
                missing_years.append(year)
                continue

            # Create the station folder only on first success
            if success_count == 0:
                local_dir = os.path.join(LOCAL_BASE_DIR, station_id)
                os.makedirs(local_dir, exist_ok=True)

            local_file_path = os.path.join(LOCAL_BASE_DIR, station_id, f"{year}.csv")
            try:
                s3_client.download_file(BUCKET_NAME, s3_key, local_file_path)
                print(f"downloaded → {local_file_path}")
                success_count += 1
            except Exception as e:
                print(f"FAILED: {e}")
                missing_years.append(year)

        if missing_years:
            missing_by_station[station_id] = missing_years
        if success_count:
            downloaded_counts[station_id] = success_count

    # Build a human-friendly mapping (airport code → missing years)
    stations_with_missing = {
        id_to_airport[sid]: years for sid, years in missing_by_station.items()
    }

    # ---- Summary ----
    print("\n=== Download Summary ===")
    total_expected = len(target_stations) * len(list(YEARS))
    total_downloaded = sum(downloaded_counts.values())
    print(f"Expected files: {total_expected}")
    print(f"Successfully downloaded: {total_downloaded}")
    print(f"Missing files: {total_expected - total_downloaded}\n")

    if stations_with_missing:
        print("Stations with missing files (by airport code):")
        for ac, years in sorted(stations_with_missing.items()):
            print(f"  {ac}: missing years {sorted(years)}")
    else:
        print("All requested files were available.")

    return missing_by_station, stations_with_missing

#use if running as script
# if __name__ == "__main__":
#     download_noaa_isd_data()
