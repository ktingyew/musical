

# MUSICAL

A program that automatically collates data of my music library and uploads them into BigQuery for subsequent data visualisation.

Information collected includes tags of `.mp3` and `.flac` music files, and my personal [last.fm](https://www.last.fm/) scrobbling data.



`daily_script.bat` - Batch Script that is to be executed recurrently by Windows Task Scheduler. 

- The script runs `musical.py` with the required source and destination arguments
- To be executed daily, e.g. 10pm every day

`config.json` - Configuration File that provides arguments to `musical.py`. Content as follows:

```json
{
  "log_fpath": <filepath of logfile>,
  "log_verbose": <boolean>,
  "temp_dir": <directory to store temporary files useful for debugging>,
  "music_dir": <directory to music library>,
  "report_out_dir": <directory to save report output>,
  "bq": {
    "cred_fpath": <filepath of json file of BigQuery credential>,
    "pj_id": <BigQuery project id>,
    "ds_id": <BigQuery dataset id>,
    "report_tbl_id": <BigQuert table id for report>,
    "scrob_tbl_id": <BigQuert table id for scrobbles>,
    "report_schema": <json string of report schema>,
    "scrob_schema": <json string of scrobbles schema>
  },
  "lastfm":{
    "username": <lastfm username>,
    "api_key": <lastfm api key>
  },
  "latest_scrob_fname": <filename of latest scrobble - updated every time program is ran; no need to edit manually>,
  "scrobble_out_dir": <directory to save processed scrobbles in jsonl format>,
  "mapper_fpath": <filepath to mapper>,
  "pd_schema_fpath": <filepath to pandas schema in YAML>,
  "unwanted_scrob_fpath": <filepath to unwanted scrobbles>
}
```





## Dependencies (python libraries)

- mutagen
- pandas
- tqdm
- google.cloud




## Data Studio Ideas

1. ~~A new dynamically generated field in report which contains the most recent date each song is scrobbled.~~

   - ~~Thought it would be interesting to have a ordered list of songs that I have not scrobbled (i.e. listened) for the longest time. Knowing how long I have not listened to some songs would be insightful.~~

2. ~~A graph of scrobble count since I started scrobbling.~~

3. Gender-rating breakdown

4. Average rating of the most populous artists

   
