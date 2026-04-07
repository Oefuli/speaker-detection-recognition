from pathlib import Path, PurePath
from subprocess import CalledProcessError, run
import time
from datetime import datetime

from typing import cast, Any

import librosa
import polars as pl
import numpy as np

from tqdm import tqdm
import logging

from joblib import Parallel, delayed

import soundfile as sf

import torch

from pyannote.audio import Model, Inference, Pipeline
from pyannote.core import Annotation

from scipy.spatial.distance import cdist

from .paths_files import get_only_dir, get_f_ps_ns
from logger_config import setup_logger

# ------------------------------------------------- #

setup_logger()
logger = logging.getLogger(__name__)
                           
# ------------------------------------------------- #


# ------------------------------------------------------------- #

SCRIPT_PATH = Path(__file__)

SRC_PATH = SCRIPT_PATH.parent.parent

PATH_PYANNOTE_EMBEDDING = SRC_PATH / 'pyannote_clf' / 'embedding_model' / 'pytorch_model.bin'

# ------------------------------------------------------------- #

def cut_out_audio(
        input_file_path: Path | str,
        output_file_path: Path | str,
        start: float,
        end: float,
        samplerate: int = 44100
        ): 
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


def split_audio(
        input_dir: Path | str,
        input_file_name: str,
        output_dir: Path | str,
        diarization: Annotation | dict,
        make_output_dir: bool = False,
        
        ):
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

    output_dir_path = Path(output_dir) / input_file_name_no_ext

    if not output_dir_path.exists() and make_output_dir:
        logging.info(f'Make directory {output_dir_path}.')
        output_dir_path.mkdir(parents=True, exist_ok=True)

    input_file_path = Path(input_dir) / input_file_name

    if isinstance(diarization, dict):
        diar_obj = diarization[input_file_name_no_ext]
    else:
        diar_obj = diarization

    count = 10001

    for turn, _, speaker in cast(Any, diar_obj.itertracks(yield_label=True)):
        
        speaker_dir = output_dir_path / speaker

        if not speaker_dir.is_dir():
            speaker_dir.mkdir(parents=True, exist_ok=True)

        file_name = f"interview-{count}.wav"

        output_file_path = speaker_dir / file_name
        
        cut_out_audio(
            input_file_path=str(input_file_path),
            output_file_path=str(output_file_path),
            start=turn.start,
            end=turn.end
            )

        split_dict[(input_file_name, speaker, file_name)] =\
            {'start_seconds': round(turn.start, 3),
             'end_seconds': round(turn.end, 3)}
        
        count += 1

    return split_dict

# -------------------------------------------------- #


def dump_diariza(
        output_dir: Path | str,
        output_file_name: Path | str,
        diarization: Annotation
                 ):
    """
    Dumps the diarization in the output_dir directory
    and names it output_file_name.
    """
    
    output_path = PurePath(output_dir).joinpath(output_file_name)

    with open(output_path, "w") as rttm_file:
        diarization.write_rttm(rttm_file)

# -------------------------------------------------- #

def diariza(
        input_dir: Path | str,
        input_file_name: str,
        pipeline: Pipeline,
        cuda_switch: bool = False,
        dump_switch: bool = False,
        dump_output_dir: Path | str | None = None,
        dump_output_file_name: Path | str | None = None
        ):
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

    input_file_path = str(Path(input_dir) / input_file_name)

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

        if dump_output_dir is None:
            dump_output_dir = input_dir

        if dump_output_file_name is None:
            dump_output_file_name = \
             PurePath(input_file_name).stem + '.rttm'

        dump_diariza(
            dump_output_dir,
            dump_output_file_name,
            diarization
            )

    return diarization

# -------------------------------------------------- #

def cdist_to_df(
        dist_dict: dict,
        cdist_metric: str
        ):

    df_lst = []

    for key, val in dist_dict.items():
        
        col_lst = key.split('/')
        event_name = col_lst[1]
        speaker_name = col_lst[2]

        # Ermitteln von Slices und Werten, unabhängig davon ob val ein Dict oder Array/Liste ist
        if isinstance(val, dict):
            slices = list(val.keys())
            values = list(val.values())
        else:
            slices = list(range(len(val)))
            values = list(val)

        # Polars DataFrame direkt über ein Dictionary-Mapping erstellen
        df = pl.DataFrame({
            'event': [event_name] * len(values),
            'speaker': [speaker_name] * len(values),
            'slice': slices,
            f'scipy.cdist.{cdist_metric}': values
        })

        df_lst.append(df)

    # Leere Liste abfangen, falls das Dict leer war
    if not df_lst:
        return pl.DataFrame()

    # DataFrames zusammenfügen (äquivalent zu pd.concat, aber reset_index entfällt)
    df_return = pl.concat(df_lst)

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

def compare_audios_dist(
        speaker1_path: Path | str,
        speaker2_path: Path | str,
        metric: str,                        
        cuda_switch: bool = False,
        norm: bool = False
        ):
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

    emb1_raw = np.asarray(inference(speaker1_path))
    emb2_raw = np.asarray(inference(speaker2_path))

    if norm:

        embedding1 = norm_array(emb1_raw.reshape(1, -1))
        embedding2 = norm_array(emb2_raw.reshape(1, -1))

    else:
        # `embeddingX` is (1 x D) numpy array extracted
        # from the file as a whole.
        embedding1 = emb1_raw.reshape(1, -1)
        embedding2 = emb2_raw.reshape(1, -1)    

    return cdist(
        embedding1,
        embedding2,
        metric=metric # type: ignore
        ) 

# -------------------------------------------------------------------------- #

def get_speaker_dist(
        reference_dir_path: Path | str,
        reference_file_name: str,
        split_dir_path: Path | str,                      
        metric: str,
        ext: str='wav',
        duration_val: float=float(5),
        cuda_switch: bool=False
        ):
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

    path_reference = str(Path(reference_dir_path) / reference_file_name)

    
    logger.info("Start computing distances.") 
    logger.info(f"Use CUDA: {cuda_switch}.")    
    logger.info(f"Reference path: {path_reference}.")    
    logger.info(f'Duration value: {duration_val} s.')

    # all directories in the split directory
    subdirs_split_dir_path = get_f_ps_ns(split_dir_path, dir_switch=True)

    pbar = tqdm(list(subdirs_split_dir_path.keys()))

    for subdir_name in pbar:

        subdir_path = Path(split_dir_path) / subdir_name

        file_paths_names = get_f_ps_ns(
                           subdir_path, file_ext=ext)

        file_paths = list(file_paths_names.values())
        file_names = list(file_paths_names.keys())

        for file_path, file_name in zip(file_paths, file_names):

            duration = librosa.get_duration(filename=file_path)

            if duration >= duration_val:

                dist = compare_audios_dist(speaker1_path=path_reference,
                                           speaker2_path=file_path,
                                           metric=metric,                                        
                                           cuda_switch=cuda_switch)
                if dist.shape != (1,1):

                    logger.warning(f'Shape of distance is {dist.shape}!')

                dist_dict[(PurePath(split_dir_path).name,
                           subdir_name,
                           file_name,
                           reference_file_name
                           )] = {f'metric_{metric}': dist[0,0]}

    return dist_dict

# ------------------------Audio Infos------------------------------------ #
# ----------------------------------------------------------------------- #

def get_audio_durations(
        output_dir: Path | str
        ):
    """
    Returns a dictionary containing the durations
    of the audio files in seconds.
    """

    print('Duration in seconds.')

    audio_dir_dict = {}

    for dir_name in get_only_dir(output_dir):

        dir_path = str(Path(output_dir) / dir_name)

        audio_dir_dict[dir_path] = {}

        file_paths_names = get_f_ps_ns(Path(output_dir) / dir_name, file_ext='wav')

        file_paths = list(file_paths_names.values())
        file_names = list(file_paths_names.keys())

        for file_path, file_name in zip(file_paths, file_names):

            duration = librosa.get_duration(filename=file_path)

            audio_dir_dict[dir_path][file_name] = duration

    return audio_dir_dict

# -------------------------------------------------- #

def get_audio_infos(
        path_in: Path | str
        ):
    """
    Get informations of an audio file.
    
    https://audeering.github.io/audiofile/usage.html
    """
    path_obj = Path(path_in)
    file_name = path_obj.name

    # soundfile object
    audio_infos = sf.info(str(path_in))

    
    skip_lst = ['verbose', 'extra_info', 'name']

    # list of strings
    infos_lst = [info for info in dir(audio_infos) if not\
                 (info.startswith('__') or\
                  info in skip_lst)]
    
    audio_infos_dict = {}

    for attr in infos_lst:

        audio_infos_dict[(file_name, attr)] =\
        eval(f"audio_infos.{attr}")
        
    audio_infos_dict[(file_name, 'size_bytes')] = path_obj.stat().st_size

    return audio_infos_dict

# ---------------------------------------------------------------------- #    

def coll_infos(
        path_in: Path | str,
        ext: str = 'wav',               
        n_jobs: int = 1
        ):
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

    logger.info("Gathering information about the audio files has started.")

    file_paths_names = get_f_ps_ns(path_in,
                                   file_ext=ext)

    logger.info(f'Collect infos from {len(file_paths_names)} files.')

    result = Parallel(n_jobs=n_jobs)(delayed(get_audio_infos)(file_path)\
                                     for file_path in list(file_paths_names.values()))

    logger.info("Gathering information about the audio files has finished.")

    return result
# --------------------------------------------------------------------- #

def audio_infos_to_df(
        infos_dict: list # Korrigiert auf 'list', da joblib Parallel eine Liste von Dicts zurückgibt
        ):
    """
    Converts a list of dictionaries into a polars data frame.
    """

    logger.info("Conversion of the dictionary to a DataFrame started.")
    
    # Daten für Polars flachklopfen: { 'Dateiname1': {'attr1': val1, ...}, ... }
    data_rows = {}
    
    for list_item in infos_dict:
        for (file_name, attr), value in list_item.items():
            if file_name not in data_rows:
                # Wir legen den Dateinamen direkt als Spalte an
                data_rows[file_name] = {'file_name': file_name}
            data_rows[file_name][attr] = value

    # Liste der umgewandelten Dictionaries direkt in Polars laden
    df = pl.DataFrame(list(data_rows.values()))

    logger.info("Conversion of the dictionary to a DataFrame completed.")
    
    return df