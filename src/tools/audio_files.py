import os
from pathlib import Path, PurePath
from subprocess import CalledProcessError, run
import time
from datetime import datetime
import inspect

import librosa
import pandas as pd
import numpy as np

from tqdm import tqdm

from joblib import Parallel, delayed

import soundfile as sf

import torch

from pyannote.audio import Model, Inference, Pipeline
from pyannote.core import Annotation

from scipy.spatial.distance import cdist

from src.tools.paths_files import get_only_dir, get_f_ps_ns

# ------------------------------------------------------------- #

SCRIPT_PATH = Path(__file__)

SRC_PATH = SCRIPT_PATH.parent.parent

PATH_PYANNOTE_EMBEDDING = SRC_PATH / 'pyannote_clf' / 'embedding_model' / 'pytorch_model.bin'

# ------------------------------------------------------------- #

def cut_out_audio(input_file_path: Path | str = None,
                  output_file_path: Path | str = None,
                  start: float = None,
                  end: float = None,
                  samplerate: int = 44100): 
    """
    Cuts out a relevant part (start, end) from the
    audio file and saves it in output_file_path
    under the number of the speaker.

    INPUT
        input_file_path (Path | str): Path of the audio file.
        output_file_path (Path | str): Output path of the
            extracted audio segment.
        start (float): Start of the audio segment in seconds.
        end (float): End of the audio segment in seconds.
        samplerate (int): Samplerate of the original file. 

    OUTPUT
        Saves the audio segment in output_file_path.
    """
    length = end - start
    
    cmd = ["ffmpeg", "-ss",
           str(start),
           "-i", input_file_path,
           "-t", str(length),
           "-vn",
           "-acodec",
           "pcm_s16le",
           "-ar",
           str(samplerate),
           "-ac",
           "1",
           output_file_path]

    # https://ffmpeg.org/ffmpeg.html
    # -ss 
    # -t : duration
    # -vn : As an input option, blocks all video streams of
    #   a file from being filtered or being automatically
    #   selected or mapped for any output.
    # -acodec : Set the audio codec. This is an alias for -codec:a.
    # pcm_s16le : PCM signed 16-bit little-endian
    # https://en.wikipedia.org/wiki/Endianness
    # -ar : sample rate
    # -ac : channels?
    
    # cmd = ["ffmpeg", "-ss",
    #        str(start),
    #        "-i", input_file_path,
    #        "-t", str(length),
    #        "-vn",
    #        "-acodec",
    #        "pcm_s16le",
    #        "-ar",
    #        "22050",
    #        "-ac",
    #        "1",
    #        output_file_path]

    
    try:
        run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"FFMPEG error {str(e)}")

# -------------------------------------------------- #


def split_audio(input_dir: Path | str = None,
                input_file_name: str = None,
                output_dir: Path | str = None,
                make_output_dir: bool = False,
                diarization: Annotation = None):
    """
    Splits the audio file based on the recognized speakers and
    saves individual sequences that are assigned to the speakers.

    Args:
        input_dir           (Path | str): Input directory of the audio file.
        input_file_name     (str): Name of the audio file.
        output_dir          (Path | str): Name of the parent directory in
            which subdirectories with the names of the
            speakers is to be created.
        make_output_dir     (bool): If a directory does not exist
            and should be created, use True.
        diarization         (pyannote Annotation object): Result of
            Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", ...)

    Returns:
        - Creates a folder with the name of the audio file
          in the output_dir directory and saves the split
          audio files there.
        - dict: Contains the file name and the start and end
          time for the splits.
        
    """

    split_dict = {}

    input_file_name_no_ext = Path(input_file_name).stem

    output_dir = os.path.join(output_dir,
                              input_file_name_no_ext)

    if not os.path.exists(output_dir) and make_output_dir:

        print(f'\nMake directory {output_dir}.')
        os.mkdir(output_dir)

    input_file_path = os.path.join(input_dir,
                                   input_file_name)

    if isinstance(diarization, dict):

        diarization = diarization[input_file_name_no_ext]

    count = 10001

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # print(f"start={turn.start}s stop={turn.end}s speaker_{speaker}")

        speaker_dir = f"{output_dir}/{speaker}/"

        if not Path(speaker_dir).is_dir():
            Path(speaker_dir).mkdir(parents=True)

        file_name = f"interview-{count}.wav"

        output_file_path = os.path.join(speaker_dir, file_name)
        
        cut_out_audio(input_file_path, output_file_path, turn.start, turn.end)        

        split_dict[(input_file_name, speaker, file_name)] =\
            {'start_seconds': round(turn.start, 3),
             'end_seconds': round(turn.end, 3)}
        
        count += 1

    return split_dict

# -------------------------------------------------- #


def dump_diariza(output_dir: Path | str = None,
                 output_file_name: str = None,
                 diarization: Annotation = None):
    """
    Dumps the diarization in the output_dir directory
    and names it output_file_name.
    """
    
    output_path = PurePath(output_dir).joinpath(output_file_name)

    with open(output_path, "w") as rttm_file:
        diarization.write_rttm(rttm_file)

# -------------------------------------------------- #

def diariza(input_dir: Path | str = None,
            input_file_name: str = None,
            pipeline: Pipeline = None,
            cuda_switch: bool = False,
            dump_switch: bool = False,
            dump_output_dir: Path | str = None,
            dump_output_file_name: Path | str = None):
    """
    Uses pyannotes pipeline =
    Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", ...)
    for the diarization of an audio file

    INPUT
        input_dir               (Path | str): Directory in which the audio file is located.
        input_file_name         (Path | str): Name of the audio file in input_dir.
        pipeline                (pyannote Pipeline): Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", ...)
        cuda_switch             (bool): Switch regarding the use of CUDA
            (cuda_switch=True).
        dump_switch             (bool): Switch for dumping the diarization result
            (dump_switch=True).
        dump_output_dir         (Path | str): Directory to dump the result.
        dump_output_file_name   (Path | str): Name of the dump file. If it is
            None it takes the stem of input_file_name.

    OUTPUT
        Pyannotes diarization object.
    """

    input_file_path = r'%s' % os.path.join(input_dir, input_file_name)

    if cuda_switch:

        # send pipeline to GPU (when available)
        pipeline.to(torch.device("cuda"))

    print("Duration of audio file: "
          f"{librosa.get_duration(path=input_file_path)} s.")

    t0 = time.time()
    print(f"\nStarted diarization at {datetime.now()}.")
    print(f"\tCUDA switch: {cuda_switch}.")

    diarization = pipeline(input_file_path)

    t1 = time.time()
    print(f"\nEnded diarization at {datetime.now()}.")
    print(f"\n\t--> Elapsed time: {t1 - t0} s.")

    if dump_switch:

        if dump_output_file_name is None:

            dump_output_file_name = \
             PurePath(input_file_name).stem + '.rttm'

        dump_diariza(dump_output_dir,
                     dump_output_file_name,
                     diarization)

    return diarization

# -------------------------------------------------- #

def cdist_to_df(dist_dict: dict = None,
                cdist_metric: Path | str = None):

    df_lst = []

    for key in dist_dict.keys():

        slices_pred = pd.Series(dist_dict[key],
                                name='scipy.cdist')

        col_lst = key.split('/')

        event_lst = [col_lst[1]]*slices_pred.shape[0]
        speaker_lst = [col_lst[2]]*slices_pred.shape[0]

        df = pd.DataFrame([event_lst,
                           speaker_lst,
                           slices_pred.index,
                           slices_pred],
                           index=['event',
                                 'speaker',
                                 'slice',
                                 'scipy.cdist.' + cdist_metric]).transpose()

        df_lst.append(df)

    # df.set_index('path', inplace=True, append=False)

    df_return = pd.concat(df_lst)

    df_return.reset_index(inplace=True, drop=True)

    return df_return

# ---------------------------------------------------------------------- #

def norm_array(vec):
    """
    This code normalizes a vector.
    """

    vec_len = np.linalg.norm(vec)

    if vec_len != 0:

        return vec / np.linalg.norm(vec)

    else:

        return vec

# ---------------------------------------------------------------------- #

def compare_audios_dist(speaker1_path: Path | str = None,
                        speaker2_path: Path | str = None,
                        metric: str = None,                        
                        cuda_switch: bool = False,
                        norm: bool = False):
    """
    INPUT
        speaker1_path (Path | str): Path to the audo file
            for the first speaker.
        speaker2_path (Path | str): Path to the audo file
            for the second speaker.
    """    
    
    model = Model.from_pretrained(PATH_PYANNOTE_EMBEDDING)

    inference = Inference(model, window="whole")

    if cuda_switch:

        inference.to(torch.device("cuda"))

    if norm:

        embedding1 = norm_array(inference(speaker1_path).reshape(1, -1))
        embedding2 = norm_array(inference(speaker2_path).reshape(1, -1))

    else:
        # `embeddingX` is (1 x D) numpy array extracted
        # from the file as a whole.
        embedding1 = inference(speaker1_path).reshape(1, -1)
        embedding2 = inference(speaker2_path).reshape(1, -1)    

    return cdist(embedding1,
                 embedding2,
                 metric=metric)

# -------------------------------------------------------------------------- #

def get_speaker_dist(reference_dir_path: Path | str=None,
                     reference_file_name: str=None,
                     split_dir_path: Path | str=None,                      
                     metric: str=None,
                     ext: str='wav',
                     duration_val: float=float(5),
                     cuda_switch: bool=False,
                     verbose: int=-1):
    """
    Returns the distances between the given
    audio files in split_dir_path and the reference in
    reference_dir.

    Args:
        reference_dir_path (Path | str): Directory for the reference file.
        reference_file_name (str): File name of the reference file.
        split_dir_path (Path | str): Directory in which the splits of an audio
            file are located. The splits must be sorted by speaker
            in subdirectories.
        metric (str): Metric to compute the distance.
        ext (str): File extension of the files in the subdirectories
            of the split directory.
        duration_val (float): Minimum duration of the audio file.
        cuda_switch (bool): Use CUDA or not.

    Returns:
        dict: 
            Keys: split_dir - Name of the split directory (see above).
                  subdir_name - Names of the subdirectories in split_dir.
                  file_name - Names of the files in subdir_name.
                  ref_file_name - Name of the reference file.
                  metric - Name of the metric used.
            Values: Computed distances.
    """

    dist_dict = {}

    path_reference = os.path.join(reference_dir_path,
                                  reference_file_name)

    if verbose > -1:
        print(f'\nFunction: {inspect.currentframe().f_code.co_name}\n')
    
        print(f"\n\tUse CUDA: {cuda_switch}")
    
        print(f"\tReference path: {path_reference}\n")
    
        print(f'\tDuration value: {duration_val} s.')

    # all directories in the split directory
    subdirs_split_dir_path = get_f_ps_ns(split_dir_path, dir_switch=True)

    pbar = tqdm(list(subdirs_split_dir_path.keys()))

    for subdir_name in pbar:

        # print('dir_name: ', dir_name)

        subdir_path = os.path.join(split_dir_path, subdir_name)

        # print('dir_path: ', dir_path)

        file_paths_names = get_f_ps_ns(
                           subdir_path, file_ext=ext)

        file_paths = list(file_paths_names.values())
        file_names = list(file_paths_names.keys())

        # print('file_paths_names: ', file_paths_names)

        for file_path, file_name in zip(file_paths, file_names):

            # print('file_path: ', file_path)

            duration = librosa.get_duration(filename=file_path)

            # print('Duration: ', duration)

            if duration >= duration_val:

                dist = compare_audios_dist(speaker1_path=path_reference,
                                           speaker2_path=file_path,
                                           metric=metric,                                        
                                           cuda_switch=cuda_switch)
                if dist.shape != (1,1):

                    print(f'\n!!!!! Warning: Shape of distance is {dist.shape}!!!!!\n')

                dist_dict[(PurePath(split_dir_path).name,
                           subdir_name,
                           file_name,
                           reference_file_name
                           )] = {f'metric_{metric}': dist[0,0]}

    return dist_dict

# ------------------------Audio Infos------------------------------------ #
# ----------------------------------------------------------------------- #

def get_audio_durations(output_dir: Path | str = None):
    """
    Returns a dictionary containing the durations
    of the audio files in seconds.
    """

    print('Duration in seconds.')

    audio_dir_dict = {}

    for dir_name in get_only_dir(output_dir):

        dir_path = os.path.join(output_dir, dir_name)

        audio_dir_dict[dir_path] = {}

        file_paths_names = get_f_ps_ns(
            os.path.join(output_dir, dir_name), file_ext='wav')

        file_paths = list(file_paths_names.values())
        file_names = list(file_paths_names.keys())

        for file_path, file_name in zip(file_paths, file_names):

            duration = librosa.get_duration(filename=file_path)

            audio_dir_dict[dir_path][file_name] = duration

    return audio_dir_dict

# -------------------------------------------------- #

def get_audio_infos(path_in: Path | str = None):
    """
    Get informations of an audio file.
    
    https://audeering.github.io/audiofile/usage.html
    """
    file_name = os.path.basename(path_in)

    # soundfile object
    audio_infos = sf.info(path_in)

    
    skip_lst = ['verbose', 'extra_info', 'name']

    # list of strings
    infos_lst = [info for info in dir(audio_infos) if not\
                 (info.startswith('__') or\
                  info in skip_lst)]
    
    audio_infos_dict = {}

    for attr in infos_lst:

        audio_infos_dict[(file_name, attr)] =\
        eval(f"audio_infos.{attr}")
        
    audio_infos_dict[(file_name, 'size_bytes')] = os.path.getsize(path_in)

    return audio_infos_dict

# ---------------------------------------------------------------------- #    

def coll_infos(path_in: Path | str = None,
               ext: str = 'wav',               
               n_jobs: int = 1):
    """
    Collects informations of the audio files. Works with
    jolib and parallel computing.

    Args:
        path_in (Path | str): Path to the directory of the audio files.
        ext (str): File format extension for the input files.

    Returns:
        dict:
            keys: 
    
    Raises:
        No extraordinary raises.
    """

    start_time = time.perf_counter()

    print(f'Program started at: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')

    #audios_infos_dict = {}

    file_paths_names = get_f_ps_ns(path_in,
                                   file_ext=ext)

    print(f'\n\tCollect infos from {len(file_paths_names)} files.')

    result = Parallel(n_jobs=n_jobs)(delayed(get_audio_infos)(file_path)\
                                     for file_path in list(file_paths_names.values()))

    finish_time = time.perf_counter()

    print(f'\nProgram finished in {finish_time - start_time} seconds.')
    print(f'Program finished at: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')

    return result
# --------------------------------------------------------------------- #

def audio_infos_to_df(infos_dict: dict = None):
    """
    Converts a dictionary into a data frame.
    """

    start_time = time.perf_counter()

    print(f'Program started at: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
    
    df = pd.Series({key: value for list_item in infos_dict for key, value in list_item.items()}).unstack()

    finish_time = time.perf_counter()
    
    print(f'Program finished at: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
    print(f'\nProgram finished in {finish_time - start_time} seconds.')
    

    return df
