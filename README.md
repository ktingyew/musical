

# README

`report.py` - Python 3 Script Code.

`daily_script.bat` - Batch Script that is executed recurrently by Windows Task Scheduler. 

- Runs `musical.py` with the required source and destination arguments.

`config.json` - Configuration File
```json
{
  "log_fpath": <filepath of logfile>,
  "log_verbose": <boolean>,
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
  "scrobble_out_dir": <directory to save processed scrobbles>,
  "mapper_fpath": <filepath to mapper>,
  "pd_schema_fpath": <filepath to pandas schema in YAML>,
  "unwanted_scrob_fpath": <filepath to unwanted scrobbles>
}
```





## Dependencies 

- mutagen
- pandas

