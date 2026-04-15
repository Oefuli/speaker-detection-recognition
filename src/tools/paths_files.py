from pathlib import Path, PurePath
import shutil
from tqdm import tqdm
import json
import logging
from typing import Any

# ------------------------------------------------- #

logger = logging.getLogger(__name__)
                           
# ------------------------------------------------- #


def write_json(
        file_path: str | Path, 
        data: Any
               ):

    with open(file_path, 'w') as f:
        json.dump(data, f)

# ------------------------------------------------- #

def read_json(
        file_path: str | Path
        ) -> dict:

    file_name = PurePath(file_path).stem
    dir_path = PurePath(file_path).parent
    
    try:
        with open(file_path, 'r') as file:
            data = json.load(file) 
        logger.info(
            f"JSON file {file_name} in {dir_path} was successfully read in."
            )
        return data      
    
    except FileNotFoundError:
        logger.exception(f"File {file_name} in {dir_path} doesn't exist.")
        raise        

    except json.JSONDecodeError:
        logger.exception(f"There was an error decoding {file_name} in {dir_path}.")
        raise

    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        raise

# ----------------------------------------------------------------- #

def get_f_ps_ns(
        dir_path: Path | str,
        file_ext: str = '*',
        dir_switch: bool = False,
        subfolder_switch: bool = False,
        show_progress: bool = True,
        verbose: int = -1
        ) -> dict:
    """
    Returns all files and file paths or directories
    with a specific extension from the given directory.

    Args:
        dir_path (Path | str): Path to a directory.
        file_ext (str): File extension format (e.g. 'json', 'wav'). Default '*' returns all.
        dir_switch (bool): If True, returns subdirectory names and paths instead of files.
        subfolder_switch (bool): If True, searches recursively in all subdirectories.
        show_progress (bool): If True, displays a tqdm progress bar.
        verbose (int): Verbosity level for logging.

    Returns:
        dict: Keys are file/dir names (or tuples if subfolder_switch=True), 
              Values are absolute file/dir paths as strings.
    """

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

    # Die verwirrende Invertierung ist weg. Wir nutzen 'not show_progress' direkt im Aufruf:
    for file_path in tqdm(list(path_iterator), disable=not show_progress, desc=desc_text):

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
        logger.info(f'Found {len(files_in_dir_dict)} elements.')

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
