from pathlib import Path, PurePath
import shutil
from tqdm import tqdm
import json
import logging

from logger_config import setup_logger

# ------------------------------------------------- #

setup_logger()
logger = logging.getLogger(__name__)
                           
# ------------------------------------------------- #


def write_json(
        file_path: str, 
        data: str
               ):

    with open(file_path, 'w') as f:
        json.dump(data, f)

# ------------------------------------------------- #

def read_json(
        file_path: str
        ) -> dict | bool:

    file_name = str(PurePath(file_path).stem)
    dir_path = str(PurePath(file_path).parent)

    data = False
    
    try:
        with open(file_path, 'r') as file:
            data = json.load(file) 

        logger.info("JSON file was successfully read in.")

        return data      
    
    except FileNotFoundError:

        logger.error(f"File {file_name} in {dir_path} doesn't exist.")
        
        data = False

    except json.JSONDecodeError:

        logger.error(f"There was an error decoding {file_name} in {dir_path}.")

        data = False

    except Exception as e:

        logger.exception(f"An unexpected error occurred: {e}")

        data = False    
        
    return data

# ----------------------------------------------------------------- #

def get_f_ps_ns(
        dir_path: Path | str,
        file_ext: str = '*',
        dir_switch: bool = False,
        subfolder_switch: bool = False,
        tqdm_switch: bool = True,
        verbose=-1
        ) -> dict:
    """
    Returns all files and file paths or files and
    files paths with a specific extension
    from the given directory.

    Args

        dir_path (str): Path to a directory
        file_ext: Default returns all files and directories.
             File extension format: E.g. 'json', 'wav' etc.
        dir_switch (bool): If True, file_names contains all
            subdirectory names in dir_path and file_paths the paths of
            the subdirectories. The switch is for control purposes only.

    Returns

        dict: Keys: Dir/ File names.
              Values: Dir/ File paths.
    """

    tqdm_switch = not tqdm_switch
    base_dir = Path(dir_path)

    # guarantees e.g. file_ext == '*.wav'
    if file_ext[0] != '*':
        file_ext = '*' + file_ext
        if file_ext[1] != '.':    
            file_ext_splitted = file_ext.split('*')
            file_ext = '*' + '.' + ''.join(file_ext_splitted[1:])
    elif file_ext is None:
        raise ValueError("No file extension has been chosen.")

    if verbose > -1:
        logger.info(f"file_ext = {file_ext}")
    
    files_in_dir_dict = {}

    if subfolder_switch:
        path_iterator = base_dir.rglob(file_ext)
        desc_text = f'Get all {file_ext} files in all subdirectories'
        
    else:
        path_iterator = base_dir.glob(file_ext)
        desc_text = 'Arrange dictionary'


    for file_path in tqdm(list(path_iterator), disable=tqdm_switch, desc=desc_text):

        if dir_switch:
            if file_path.is_dir():
                files_in_dir_dict[file_path.name] = str(file_path)
            else:
                logger.warning(f'{file_path} is not a directory.')
        else: 
            if file_path.is_file():
                if subfolder_switch:
                    files_in_dir_dict[(str(file_path.parent), file_path.name)] = str(file_path)
                else:
                    files_in_dir_dict[file_path.name] = str(file_path)
            else:
                logger.warning(f'{file_path} is not a file.')

    if verbose > -1:
        logger.info(f'Found {len(files_in_dir_dict)} files.')

    return files_in_dir_dict

# ---------------------------------------------------------------------- #

def get_only_dir(path: Path | str):

    logger.info(f'Get directory paths from: {str(path)}')

    return [p.name for p in Path(path).iterdir() if p.is_dir()]


# -------------------------------------------------------- # 

def copy_audio_slice(
        input_dir: Path | str,
        output_dir: Path | str,
        dir_file_tuple: tuple,
        speaker_name: str,
        dir_distinction: str | None = None
    ):
    """
    Copies files from input directory to a parent directory in the
    directory output_dir using the string reference_file_name
    to name the parent and the child directory.

    Args:

        input_dir (str):
            Input directory for the audio files.
        
        output_dir (str):
            Upper output directory for the audio files.
        
        dir_file_tuple (tuple):
            Tuple of the form
            (name of the event dir, speaker ID, Name of the speaker file),
            e.g. (dir_name, SPEAKER_01, interview-10072.wav)
            
        speaker_name (str): 
            Name of the speaker after whom a directory is named.
        
        dir_distinction (str):
            In case you want to create different directories for one speaker,
            e.g. because you are using different reference files.

    Returns:

        No return.
    """
    
    src_path = Path(input_dir) / dir_file_tuple[0] / dir_file_tuple[1] / dir_file_tuple[2]

    # destination path
    if dir_distinction is not None:

        speaker_name = speaker_name + f'_{dir_distinction}'

    dst_path = Path(output_dir) / speaker_name / dir_file_tuple[0] / dir_file_tuple[2]

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy(src_path, dst_path)
