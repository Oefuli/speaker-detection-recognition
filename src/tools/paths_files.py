import os
import glob
from pathlib import Path, PurePath
import shutil
from tqdm import tqdm
import json
import inspect

# ------------------------------------------------- #
# ------------------------------------------------- #


def write_json(file_path: str = None, 
               data: str = None):
    
    with open(file_path, 'w') as f:
        json.dump(data, f)

# ------------------------------------------------- #

def read_json(file_path) -> dict:

    file_name = str(PurePath(file_path).stem)

    dir_path = str(PurePath(file_path).parent)

    data = False
    
    try:
        with open(file_path, 'r') as file:
            data = json.load(file) 

        print(f'\nFunction: {inspect.currentframe().f_code.co_name}' \
                      f"\n\tJSON file was successfully read in.")

        return data      
    
    except FileNotFoundError:

        print(f'\nFunction: {inspect.currentframe().f_code.co_name}' \
        f'\n\tError: File {file_name} in {dir_path}' \
        " doesn't exist.")
        
        data = False

    except json.JSONDecodeError:

        print(f'\nFunction: {inspect.currentframe().f_code.co_name}' \
        f'\n\tError: There was an error decoding {file_name}' \
        f'\n\tin {dir_path}.')

        data = False

    except Exception as e:

        print(f'\nFunction: {inspect.currentframe().f_code.co_name}' \
        f'\n\tError: An unexpected error occurred: {e}')

        data = False    
    
    finally:
        return data

# ----------------------------------------------------------------- #

def get_f_ps_ns(dir_path: str = None,
                file_ext: str = '*',
                dir_switch: bool = False,
                subfolder_switch: bool = False,
                tqdm_switch: bool = True,
                verbose=-1) -> dict:
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

    # guarantees e.g. file_ext == '*.wav'
    if file_ext[0] != '*':

        file_ext = '*' + file_ext

        if file_ext[1] != '.':
    
            file_ext_splitted = file_ext.split('*')
            file_ext = '*' + '.' + ''.join(file_ext_splitted[1:])

    elif file_ext is None:
        raise ValueError("No file extension has been chosen.")

    if verbose > -1:
        print('file_ext = ', file_ext)
    
    files_in_dir_dict = {}

    file_paths = []
    file_names = []

    if subfolder_switch:

        glob_path = [file_path for file_path in\
                     tqdm(glob.glob(os.path.join(dir_path, '**/' + file_ext), recursive=True),
                          disable=tqdm_switch,
                          desc=f'Get all {file_ext} files in all subdirectories')]

    else:

        glob_path = glob.glob(os.path.join(dir_path, file_ext))


    for file in tqdm(glob_path,
                     disable=tqdm_switch,
                     desc='Arrange dictionary'):

        # The switch is for control purposes only.
        if dir_switch:

            if os.path.isdir(file):

                files_in_dir_dict[os.path.basename(file)] = file

            else:

                print(f'{file}\n is not a directory.\n')

        else: 
            if os.path.isfile(file):

                if subfolder_switch:

                    files_in_dir_dict[(str(PurePath(file).parent), str(PurePath(file).name))] = file


                else:
                    files_in_dir_dict[os.path.basename(file)] = file

            else:
                print(f'{file}\n is not a file.\n')

    if verbose > -1:

        print(f'\nFound {len(file_names)} files.')

    return files_in_dir_dict

# ---------------------------------------------------------------------- #

def get_only_dir(path):

    print('\n Get directory paths from:')
    print('\t', path)

    return next(os.walk(path))[1]


# -------------------------------------------------------- # 

def copy_audio_slice(input_dir:str = None,
                     output_dir: str = None,
                     dir_file_tuple: tuple = None,
                     speaker_name: str = None,
                     dir_distinction: str = None
                    ):
    """
    Copies files from input directory to a parent directory in the
    directory output_dir using the string reference_file_name
    to name the parent and the child directory.

    Args:

        input_dir (str): Input directory for the audio files.
        
        output_dir (str): Upper output directory for the audio files.
        
        dir_file_tuple (tuple): (name of the event dir, speaker ID, Name of the speaker file),
            e.g. (dir_name, SPEAKER_01, interview-10072.wav)
            
        speaker_name (str): Name of the speaker after whom a directory is named.
        
        dir_distinction (str): In case you want to create different directories for one speaker,
            e.g. because you are using different reference files.

    Returns:

        No return.
    """
    
    src_path = os.path.join(input_dir,
                             dir_file_tuple[0],
                             dir_file_tuple[1],
                             dir_file_tuple[2])

    # destination path
    if dir_distinction is not None:

        speaker_name = speaker_name + f'_{dir_distinction}'

    dst_path = os.path.join(output_dir,
                            speaker_name,
                            dir_file_tuple[0],                            
                            dir_file_tuple[2])

    dst_parents_path = PurePath(dst_path).parent

    if not Path(dst_parents_path).is_dir():

        Path(dst_parents_path).mkdir(parents=True)

    shutil.copy(src_path, dst_path)
