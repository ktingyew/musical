import os
import sys
import argparse
import pathlib
from pathlib import Path
import json
from datetime import datetime, timedelta
import logging
from collections import namedtuple
from mutagen.flac import FLAC
from mutagen.mp3 import MP3
from mutagen.mp3 import EasyMP3 as EMP3
import pandas as pd
from tqdm import tqdm
import yaml
from google.cloud import bigquery as bq
import requests

def init_dirpath(
    p: pathlib.Path,
    root: pathlib.Path
) -> pathlib.Path :
    """ Returns absolute directory path of pathlib.Path type.

    Accounts for both relative and absolute dirpaths.
    Raise error if path is not a directory.

    Args:
        p: A Path object of path to directory. Either relative, or absolute.
        root: A Path object of directory of this project (i.e. project root 
          directory).
    
    Returns: 
        A Path object containing an absolute path. If p is absolute to begin 
        with, then no changes. If p is relative, then the root is prepended to
        p and then returned. For example:

        Example 1:
        >> p=Path("C:/Project/dir") # absolute
        >> root=Path("C:/Project")
        >> init_dirpath(p, root) 
        Path("C:/Project/dir")

        Example 2:
        >> p=Path("./dir") # relative
        >> root=Path("C:/Project")
        >> init_dirpath(p, root) 
        Path("C:/Project/dir")

    Raises:
        FileNotFoundError: If p is not a valid path to a directory
    """
    # if not p.is_dir():
    #     raise FileNotFoundError(f"{p} is not a directory")
    # if p.is_absolute():
    #     return p
    # else:
    #     return root/p

    if p.is_absolute():
        if p.is_dir():
            return p
        else:
            raise FileNotFoundError(f"{p} is not a directory")
    else: # is relative
        if (root/p).is_dir():
            return root/p
        else:
            raise FileNotFoundError(f"{p} is not a directory")


def init_filepath(
    p: pathlib.Path,
    root: pathlib.Path
) -> pathlib.Path :
    """ Returns absolute file path of pathlib.Path type.

    Accounts for both relative and absolute filepaths.
    Raise error if file is not a directory.

    Args:
        p: A Path object of path to file. Either relative, or absolute.
        root: A Path object of directory of this project (i.e. project root 
          directory).
    
    Returns: 
        A Path object containing an absolute path. If p is absolute to begin 
        with, then no changes. If p is relative, then the root is prepended to
        p and then returned. For example:

        Example 1:
        >> p=Path("C:/Project/log.txt") # absolute
        >> root=Path("C:/Project")
        >> init_dirpath(p, root) 
        Path("C:/Project/dir/log.txt")

        Example 2:
        >> p=Path("./log.txt") # relative
        >> root=Path("C:/Project")
        >> init_dirpath(p, root) 
        Path("C:/Project/log.txt")

    Raises:
        FileNotFoundError: If p is not a valid path to a file
    """
    if p.is_dir():
        raise FileNotFoundError(f"{str(p)} is not a file")
    if p.is_absolute():
        return p
    else:
        return root/p

def flac_extractor(
    filepath: pathlib.Path
) -> dict :
    """ Extracts tags of interest of a .flac file into a dictionary.

    This function DOES NOT modify the tags of the file. It is read-only. 

    Args:
        filepath: A pathlib.Path object of absolute path to .flac file. 
    
    Returns: 
        A dictionary containing the following (19) keys:
        - Title
        - Artist
        - Album Artist
        - Album
        - Major Genre
        - Minor Genre
        - BPM
        - Key
        - Year
        - Rating
        - Major Language
        - Minor Language
        - Gender
        - DateAdded
        - Energy
        - KPlay
        - Time
        - Bitrate
        - Extension

        The values of each key in the dictionary (as of current implementation)
        contains are all un-nested.

    Raises:
        ValueError: If filepath does not lead to .flac file.
    """
    if filepath.suffix != '.flac':
        raise ValueError("filepath does not point to .flac file")
    
    file = FLAC(f"{filepath}")
    out = {}
    
    mapper = {
        'Title': 'title',
        'Artist': 'artist',
        'Album Artist': 'albumartist',
        'Album': 'album',
        'Genre': 'genre',
        'BPM': 'bpm',
        'Key': 'initial key',
        'Year': 'date',
        'Rating': 'rating',
        'Language': 'language',
        'Gender': 'copyright',
        'DateAdded': 'encodingtime',
        'Energy': 'energy',
        'KPlay': 'kplay'
    }
    
    for Tag in mapper.keys():
        try:
            t = file[mapper[Tag]]
            
            if Tag == 'Artist':
                out['Artist'] = "; ".join(t)
            
            elif Tag == 'Genre':
                if len(t) == 2:
                    out['Major Genre'], out['Minor Genre'] = t[0], t[1]
                else:
                    out['Major Genre'], out['Minor Genre'] = t[0], None
                    
            elif Tag == 'Language':
                if len(t) == 2:
                    out['Major Language'], out['Minor Language'] = t[0], t[1]
                else:
                    out['Major Language'], out['Minor Language'] = t[0], None

            elif Tag == 'Rating':
                out['Rating'] = float(t[0]) / 20.0
                
            elif Tag == 'DateAdded':
                out['DateAdded'] = \
                    datetime.strptime(t[0], '%d/%m/%Y').strftime("%Y-%m-%d")
                      
            else: # all other Tags
                out[Tag] = t[0]
                
        # Some tags are empty. Like 'energy' and 'kplay'. 
        # So an except block to catch these and give them None value.
        except:
            out[Tag] = None
            
    out['Time'] = file.info.length
    out['Bitrate'] = file.info.bitrate
    out['Extension'] = 'flac'
             
    return out

def mp3_extractor(
    filepath: pathlib.Path
) -> dict:
    """ Extracts tags of interest of a .mp3 file into a dictionary.

    This function DOES NOT modify the tags of the file. It is read-only. 

    Args:
        filepath: A pathlib.Path object of absolute path to .mp3 file. 
    
    Returns: 
        A dictionary containing the following (19) keys:
        - Title
        - Artist
        - Album Artist
        - Album
        - Major Genre
        - Minor Genre
        - BPM
        - Key
        - Year
        - Rating
        - Major Language
        - Minor Language
        - Gender
        - DateAdded
        - Energy
        - KPlay
        - Time
        - Bitrate
        - Extension

        The values of each key in the dictionary (as of current implementation)
        contains are all un-nested.

    Raises:
        ValueError: If filepath does not lead to .mp3 file.
    """
    if filepath.suffix != '.mp3':
        raise ValueError("filepath does not point to .mp3 file")
    
    out = {}
    
    file = EMP3(f"{filepath}")
    
    mapper = {
        'Title': 'title',
        'Artist': 'artist',
        'Album Artist': 'albumartist',
        'Album': 'album',
        'Genre': 'genre',
        'BPM': 'bpm',
        'Year': 'date',
        'Language': 'language',
        'Gender': 'copyright',
    }
    
    for Tag in mapper.keys():
        try:
            t = file[mapper[Tag]]
            
            if Tag == 'Artist':
                out['Artist'] = "; ".join(t)
            
            elif Tag == 'Genre':
                if len(t) == 2:
                    out['Major Genre'], out['Minor Genre'] = t[0], t[1]
                else:
                    out['Major Genre'], out['Minor Genre'] = t[0], None
                    
            elif Tag == 'Language':
                if len(t) == 2:
                    out['Major Language'], out['Minor Language'] = t[0], t[1]
                else:
                    out['Major Language'], out['Minor Language'] = t[0], None
                
            elif Tag == 'Rating':
                out[Tag] = t[0] / 20.0
            else: # all other Tags
                out[Tag] = t[0]
        except KeyError:
            out[Tag] = None

    # MP3 != EMP3. MP3 is more "dirty" compared to EMP3, but it has everything. 
    file = MP3(f"{filepath}")  

    out['Energy'] = out['DateAdded'] = out['KPlay'] = None
    for t in file.tags.getall('TXXX'):
        if   t.desc == 'EnergyLevel':
            out['Energy'] = t.text[0]
        elif t.desc == 'ENCODINGTIME':
            out['DateAdded'] = datetime.strptime(t.text[0], '%d/%m/%Y').strftime("%Y-%m-%d")
        elif t.desc == 'KPLAY':
            out['KPlay'] = t.text[0]
        else:
            pass

    out['Key'] = file.tags.getall('TKEY')[0].text[0]
    
    def _mp3rating(mutagen_mp3):
        """ Converts mp3 internal rating-values to proper no. of stars. 

        This is a simple helper function meant to specfically adress the weird
        rating storage in .mp3 files. It has no need to exist outside of this
        scope.
        
        Args:
            mutagen_mp3: mutagen.mp3.MP3

        Returns:
            A float value, representing the number of stars rated in this song,
             following the 5-star rating system.
        """
        try:
            rating_map = {
                13: 0.5, 1: 1.0, 54: 1.5, 64: 2.0, 118: 2.5, 
                128: 3.0, 186: 3.5, 196: 4.0, 242: 4.5, 255: 5.0
            }
            return rating_map[mutagen_mp3.tags.getall('POPM')[0].rating]
        except KeyError:
            return 0.0
    
    out['Rating'] = _mp3rating(file)     
    out['Time'] = file.info.length
    out['Bitrate'] = file.info.bitrate
    out['Extension'] = 'mp3'
             
    return out



if __name__ == '__main__':
    """
    """

    # As this source code resides in a 'src' dir
    PROJECT_DIR = Path(os.path.dirname(__file__)).parent

    # ========== ARG PARSING ==========

    argparser = argparse.ArgumentParser(
        description="Musical - BigQueries your music and scrobbles"
    )
    argparser.add_argument(
        "-conf", "--config",
        help="Specify path where configuration file is located",
        metavar="*.json",
        type=Path,
        required=True
    )

    ARGS = argparser.parse_args()

    # Config
    with open(ARGS.config) as cf:
        CONFIG = json.load(cf)


    # ===== LOGGING CONFIGURATION =====

    # log filepath
    LOG_FPATH = Path(CONFIG['log_fpath'])
    if not LOG_FPATH.is_absolute():
        LOG_FPATH = PROJECT_DIR/LOG_FPATH

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    fmtter = logging.Formatter(
        "[%(asctime)s]; %(levelname)s; %(name)s; %(message)s", 
        "%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(LOG_FPATH, encoding='utf8')
    if CONFIG['log_verbose']:
        file_handler.setLevel(logging.DEBUG)
    else:
        file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmtter)

    logger.addHandler(file_handler)

    # =================================

    logger.info(f"STARTED: =========================================")

    print(f"""Running Musical 2021 "{sys.argv[0]}".""")


    # ========== INIT PATHS ===========

    try:
        # music directory
        MUSIC_DIR = init_dirpath(
            p=Path(CONFIG["music_dir"]),
            root=PROJECT_DIR)

        # report output directory
        REPORT_DIR = init_dirpath(
            p=Path(CONFIG['report_out_dir']),
            root=PROJECT_DIR)

        # temp directory (to store intermediate files for debugging)
        TEMP_DIR = init_dirpath(
            p=Path(CONFIG['temp_dir']),
            root=PROJECT_DIR)

        # bigquery credentials filepath
        BQ_CRED_FPATH = init_filepath(
            p=Path(CONFIG['bq']['cred_fpath']),
            root=PROJECT_DIR)

        # scrobble output directory
        SCROB_DIR = init_dirpath(
            p=Path(CONFIG['scrobble_out_dir']),
            root=PROJECT_DIR)

        # mapper filepath
        MAPPER_FPATH = Path(CONFIG['mapper_fpath'])
        if not MAPPER_FPATH.is_absolute():
            MAPPER_FPATH = PROJECT_DIR/MAPPER_FPATH

        # pandas schema
        PD_SCHEMA_FPATH = Path(CONFIG['pd_schema_fpath'])
        if not PD_SCHEMA_FPATH.is_absolute():
            PD_SCHEMA_FPATH = PROJECT_DIR/PD_SCHEMA_FPATH

        # unwanted scrobbles filepath
        UNWANTED_FPATH = Path(CONFIG['unwanted_scrob_fpath'])
        if not UNWANTED_FPATH.is_absolute():
            UNWANTED_FPATH = PROJECT_DIR/UNWANTED_FPATH
        
    except FileNotFoundError:
        logger.exception('')
        exit()

    # =================================

    try:

        records = []
        music_file_ls = tqdm(os.listdir(MUSIC_DIR))
        for f in music_file_ls:
            if f.endswith('.flac'):
                records.append(flac_extractor(MUSIC_DIR/f))
            elif f.endswith('.mp3'):
                records.append(mp3_extractor(MUSIC_DIR/f))
            else:
                logger.warning(
                    f"Invalid file format detected in {MUSIC_DIR}: {f}")
                print(f)
        logger.info(
            f"Report: Extracted {len(records)} "
            + "songs successfully from {MUSIC_DIR}")

        df = pd.DataFrame.from_records(records)
        dt = datetime.now().strftime("%Y-%m-%d %H-%M-%S") # SQL Datetime format
        
        # convert df columns into appropriate dtypes by use of schema
        with open(PD_SCHEMA_FPATH, "r") as stream:
            yaml_gen = yaml.safe_load_all(stream) # load generator
            
            pd_schema_init = next(yaml_gen) # schema 1
            pd_schema_completion = next(yaml_gen) # schema 2
            
            # chained schema
            df = df.astype(pd_schema_init).astype(pd_schema_completion) 
          
        # rename columns to replace whitespaces with underscore 
        # (whitespace in col name is illegal in BigQuery)
        df.columns = [ col.replace(' ', '_') for col in df.columns.tolist() ]
        
        # Check for missing values in DataFrame
        for c in [
            'Title', 'Artist', 'Album_Artist', 'Album', 'Major_Genre', 'BPM', 
            'Key', 'Year', 'Rating', 'Major_Language' , 'Gender', 'DateAdded', 
            'Time', 'Bitrate', 'Extension'
            ]:
            if len(df[df[c].isna()]) > 0:
                # isolate all records with missing values in column c
                subdf = df[df[c].isna()][['Title', 'Artist']].values 
                for ks in subdf:
                    logger.warning(f"Column `{c}`: No value: {ks}")
        
        # Save df as report in newline delimited json format
        df_out_path = f"{REPORT_DIR}/report {dt}.json"
        df.to_json(df_out_path, orient='records', lines=True)
        logger.info(f"Report: Saved successfully in {df_out_path}")
        
        
        # UPLOADING REPORT TO BIGQUERY ========================================
        
        # Set environmental variables needed for bq authentication
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(BQ_CRED_FPATH)

        # Start BigQuery client
        client = bq.Client()
        logger.info(f"BigQuery: Started client")
        
        # Define project, dataset, and table references
        pj_id = CONFIG['bq']['pj_id']
        ds_id = CONFIG['bq']['ds_id']
        tbl_id = CONFIG['bq']['report_tbl_id']
        ds_ref = bq.dataset.DatasetReference(pj_id, ds_id) # create dataset ref
        tbl_ref = bq.table.TableReference(ds_ref, tbl_id) # create table ref
        
        # Define schema of report to bq
        fields = CONFIG['bq']['report_schema']
        schema = [bq.SchemaField(name=f.split(':')[0], 
                                 field_type=f.split(':')[1], 
                                 mode='NULLABLE') 
                                 for f in fields.split(',')]
        
        # Truncate the table
        # initialise table with schema using its (tbl) reference
        tbl = bq.Table(tbl_ref, schema=schema) 
        client.delete_table(tbl, not_found_ok=True)
        # set optional parameter exists_ok=True to ignore error of table 
        # already existing
        client.create_table(tbl) 
        
        # load report to bq using tbl_ref
        # schema in bq is DATE; requires this to be in pd's datetime format
        df['DateAdded'] = pd.to_datetime(df['DateAdded']) 
        client.load_table_from_dataframe(df, tbl_ref)
        logger.info(
            "BigQuery: Report uploaded successfully: "
            + f"{pj_id}-{ds_id}-{tbl_id}"
        )
        
        
        # RETRIEVING LATEST SCROBBLES FROM LAST.FM ============================
        
        # Get request to last.fm API, receive json object
        logger.info(f"Last.fm: Connecting to Last.fm API")
        PAGES = [1,2]
        page_ls = []
        for i in PAGES: 
            payload = {
                'limit': 200,
                'method': 'user.getrecenttracks', 
                'page': i,
                'user': CONFIG['lastfm']['username'],
                'api_key': CONFIG['lastfm']['api_key'],
                'format': 'json'
            }
            r = requests.get(
                'https://ws.audioscrobbler.com/2.0/', 
                params=payload
            )
            page_ls.append(r)
            
            if r.status_code != 200:
                raise ConnectionError(
                    f"Status Code of {r.status_code} from {r.url}. Aborting.")
        logger.info(
            f"Last.fm: Retrieval of JSON object from last.fm " 
            + "API successful"
        )
        
        # Construct df of new scrobbles from json
        Scrobble = namedtuple("Scrobble", "Title Artist Album Datetime")
        records = []
        for page in page_ls:
            pg = page.json()['recenttracks']['track']
            for i in pg:
                try:
                    records.append(
                        Scrobble(
                            i['name'], 
                            i['artist']['#text'], 
                            i['album']['#text'], 
                            i['date']['#text']
                        )
                    )
                except KeyError:
                    records.append(
                        Scrobble(
                            i['name'], 
                            i['artist']['#text'],
                            i['album']['#text'], 
                            None
                        )
                    )
        # drop entries with missing dt
        new = pd.DataFrame(data=records) \
            .dropna(axis=0, subset=['Datetime']) \
            .reset_index(drop=True) 
        # store new scrob records to temp directory (debugging)
        new.to_csv(
            TEMP_DIR/'new.csv',
            sep='\t',
            index=False)
        logger.info(f"new scrobble records stored in {TEMP_DIR}")

        # load the most recent comprehensive scrobble dataset; 
        # the one where we will be adding the new scrobbles   
        fpath = f"{SCROB_DIR}/{CONFIG['latest_scrob_fname']}"  
        old = pd.read_json(
            fpath, 
            orient='records', 
            convert_dates=False, 
            lines=True) # also pandas by default convert the string format of 
                        # datetime to their datetime format, which is annoying.
        logger.info(f"Scrobble: Old file detected (n={len(old)}): {fpath}")
        
        # Filter
        most_rct_datetime = old.iloc[0]['Datetime'] # UNADJUSTED DT
        idx = new[new['Datetime'] == most_rct_datetime].index.tolist()[0]
        new = new.iloc[:idx].copy()
        
        # load mapper as df
        mapper = pd.read_csv(f"{MAPPER_FPATH}", sep='\t') # tsv file
        
        # Apply mapper over the `new` df, finding and applying the corrected 
        # Title and Artist from last.fm (potentially) mislabeled ones
        for i in range(len(new)):
            title, artist = new.iloc[i]['Title'], new.iloc[i]['Artist'] 
            # attempt to look for "correct answer" in `mapper` by generating 
            # filtered df
            ans_df = mapper[
                (mapper['Artist_s'] == artist) 
                & (mapper['Title_s'] == title)
            ] 
            if len(ans_df) == 1: # there is an answer
                new.at[i, 'Title_c'] = ans_df.values.tolist()[0][2]
                new.at[i, 'Artist_c'] = ans_df.values.tolist()[0][3]
            else: # we populate the field with easy to find tags
                new.at[i, 'Title_c'] = "XXxXX"
                new.at[i, 'Artist_c'] =  "XXxXX" 

        # Adjust datetime of scrobbles 8-hours ahead (to SGT)
        dt_formatter = lambda x : (
            datetime.strptime(x, "%d %b %Y, %H:%M") \
            + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S"
        )
        new['Datetime_n'] = new['Datetime'].apply(dt_formatter)
        
        # Prepend the new df on the old df
        # concat and reset/reorder the index. 
        # We have our most recent scrobbles df! Yay.
        out = pd.concat([new, old], ignore_index=True) 
        
        
        # IDENTIFY MISTAKES IN SCROBBLES AND REPORT ===========================
        
        # Find unmapped and log to warning
        # Look at most recent 2500 scrobbles (or so) to see of any unmapped. 
        # Why not check everything? Cos waste computation.
        CHECK_PAST = 2500 
        mistake_ls = [] # list to store mistake 
        for i in range(CHECK_PAST):
            # retrieve artist, title from `out`
            title, artist = out.iloc[i]['Title'], out.iloc[i]['Artist']
            # attempt to look for "correct answer" in `mapper` by generating 
            # filtered df
            ans_df = mapper[
                (mapper['Artist_s'] == artist) 
                & (mapper['Title_s'] == title)
            ]

            if len(ans_df) == 0: 
                mistake_ls.append((title, artist, 0)) # '0' for unmapped    
                logger.warning(
                    f"Unmapped scrobble found (No. {i+1}): " \
                    + f"{title, artist}")
            elif len(ans_df) >= 2: 
                mistake_ls.append((title, artist, 2)) # '2' for multiple maps   
                logger.warning(
                    f"Scrobble mapped to more than 1 output (No. {i+1}): " \
                    + f"{title, artist}")

        # Create .tsv file and store it in temp folder
        mistake_df = pd.DataFrame(
            data=mistake_ls,
            columns=['Title', 'Artist', 'Code'])
        mistake_df.to_csv(
            TEMP_DIR / 'mistaken_maps.csv',
            sep='\t',
            index=False
        )
        logger.info(f"mistake maps stored in {TEMP_DIR}")
                        
        # Find scrobbles that have missing/null videos
        def display_col_na(input_df, col_name):
            df = input_df.copy()
            mask = df[col_name].isna()
            return df[mask]

        for col in ['Title_c', 'Artist_c', 'Datetime', 'Datetime_n']:
            tmp = display_col_na(out, col).values.tolist()
            if len(tmp) != 0:
                for entry in tmp:
                    logger.warning(
                        f"Scrobble with null value in column `{col}`. " 
                        + "Full entry: {entry}")
        
        # Find discrepancies between report and scrobbles
        rep_ls = df[['Title', 'Artist']].values.tolist()
        rep_ls = [(x[0], x[1]) for x in rep_ls] # report
        out_ls = out[['Title_c', 'Artist_c']] \
            .groupby(['Title_c', 'Artist_c']).count() \
            .index.tolist() 

        l1 = list(set(rep_ls) - set(out_ls))
        l2 = set(out_ls) - set(rep_ls)
        unwanted = pd.read_csv(f"{UNWANTED_FPATH}", sep='\t')
        unwanted = set(unwanted.itertuples(index=False, name=None))
        l2 = l2 - unwanted
        
        for tup in l1:
            logger.warning(
                f"Discrepancy: In report but not in scrobble: {tup}")
        for tup in l2:
            logger.warning(
                f"Discrepancy: In scrobble but not in report: {tup}")

        # EXPORT AND UPLOAD SCROBBLES =========================================
        
        # Export to local com
        fname = "scrobbles " + \
                datetime.strptime(
                    out.iloc[0]['Datetime_n'], 
                    "%Y-%m-%d %H:%M:%S"
                ).strftime("%Y-%m-%d %H-%M-%S") + \
                ".jsonl" 
        fpath = f"{SCROB_DIR}/{fname}"
        # save df as json newline delimited (.jsonl) with utf-8 encoding
        with open(fpath, 'w', encoding='utf-8') as fh:
            out.to_json(fh, force_ascii=False, orient='records', lines=True) 
            logger.info(
                f"Scrobble: File successfully saved (n={len(out)}): {fpath}")
        
        # Update latest_scrob_fpath.yaml to reflect the filename of the latest
        # scrobble file
        CONFIG['latest_scrob_fname'] = fname
        with open(ARGS.config, 'w') as fp:
            json.dump(CONFIG, fp, indent=4)
            logger.debug(f"Configuration updated for latest scrobbles " \
                + "filename")
        
        # Prep references and schema then upload to bq
        pj_id = CONFIG['bq']['pj_id']
        ds_id = CONFIG['bq']['ds_id']
        tbl_id = CONFIG['bq']['scrob_tbl_id']
        ds_ref = bq.dataset.DatasetReference(pj_id, ds_id) # create dataset ref
        tbl_ref = bq.table.TableReference(ds_ref, tbl_id) # create table ref
        
        fields = CONFIG['bq']['scrob_schema']
        schema = [
            bq.SchemaField(
                name=f.split(':')[0], 
                field_type=f.split(':')[1], 
                mode='NULLABLE'
            ) 
            for f in fields.split(',')
        ]
        
        # initialise table with schema using its (tbl) ref
        tbl = bq.Table(tbl_ref, schema=schema) 
        client.delete_table(tbl, not_found_ok=True) # Truncate the table
        # set optional parameter exists_ok=True to ignore error of table 
        # already existing
        client.create_table(tbl) 
        
        # schema in bq is DATETIME; requires this to be in pd's datetime format
        out['Datetime_n'] = pd.to_datetime(out['Datetime_n']) 
        client.load_table_from_dataframe(out, tbl_ref)
        logger.info(
            "BigQuery: Scrobbles uploaded successfully: " \
            + f"{pj_id}-{ds_id}-{tbl_id}"
        )

    except Exception:
        logger.exception('')
        logger.critical("EXCEPTION: =========================================")
        
    else:
        logger.info(f"COMPLETED: =======================================")



