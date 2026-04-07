from pathlib import Path

from pyannote.audio import Pipeline
from pyannote.database.util import load_rttm

from src.tools.utils import remap_dict_keys
from src.tools.audio_files import split_audio, get_speaker_dist, diariza
from src.tools.paths_files import write_json, copy_audio_slice

# ----------------------------------------------- #

SCRIPT_PATH = Path(__file__)

SRC_PATH = SCRIPT_PATH.parent

PROJECT_DIR_PATH = SRC_PATH.parent

OUTPUT_DIR_PATH = PROJECT_DIR_PATH / 'output'

PATH_PYANNOTE_MODEL = SRC_PATH / 'pyannote_clf' / 'pyannote_offline_config.yaml'

# ----------------------------------------------- #

def pyann_load_model(path_model: str = PATH_PYANNOTE_MODEL):
    """
    Loads a Pyannote model for diarization from
    src/pyannote_clf/pyannote_offline_config.yaml.
    The yaml file contains the path to the model 
    pyannote_pytorch_model.bin.
    
    Args
    	path_model (str): Path to the yaml file which 
    	    contains additional informations for the
    	    model and the path to the file of the model.
    	    
    Returns
        A Pyannote model.
    """

    return Pipeline.from_pretrained(path_model)

# ------------------------------------------------------------- #


class diarize_voice_rec():
    """
    Diarizes the audio recording input_file_name_to_diar, splits it,
    stores the splits, compares the splits to a reference audio file
    and is able to copy the nearest audio files in an extra directory.
    """

    def __init__(self,
                 input_dir_to_diar: Path | str = None,
                 input_file_name_to_diar: str = None,
                 dump_rttm_dir: Path | str = OUTPUT_DIR_PATH / 'pyannote_diarization_rttm',
                 split_dir: Path | str = OUTPUT_DIR_PATH / 'audio_splits',
                 output_sel_slices: Path | str = OUTPUT_DIR_PATH / 'audio_sel_person'
                 ) -> None:
        
        """
        input_dir_to_diar (Path | str): Path to the directory in which the files
          that are to be diarized are located.
        input_file_name_to_diar (str): Name of the file that is to be
          diarized.
        dump_rttm_dir (Path | str): Path to the directory in which pyannote can
          store the diarization.
        split_dir (Path | str): Path for saving the splits.
        output_sel_slices (Path | str): Path for saving the selected splits

        
        """

        self.input_dir_to_diar = input_dir_to_diar
        self.input_file_name_to_diar = input_file_name_to_diar
        self.dump_rttm_dir = dump_rttm_dir
        self.split_dir = split_dir        
        self.output_sel_slices = output_sel_slices

    # ------------------------------------------------------------------- #

    def diarize(self,
                cuda_switch: bool = False,
                dump_switch: bool = False,
                dump_output_file_name: str = None):
        """
        Uses pyannotes pipeline =
        Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", ...)
        for the diarization of an audio file
        """

        # pipeline = Pipeline.from_pretrained(
        #     "pyannote/speaker-diarization-3.1",
        #     use_auth_token=self.access_token
        #                                     )

        pipeline = pyann_load_model()

        self.diarization = diariza(
            input_dir=self.input_dir_to_diar,
            input_file_name=self.input_file_name_to_diar,
            pipeline=pipeline,
            cuda_switch=cuda_switch,
            dump_switch=dump_switch,
            dump_output_dir=self.dump_rttm_dir,
            dump_output_file_name=dump_output_file_name)

    # ------------------------------------------------------------------- #
    
    def load_diarized(self,
                      rttm_dir_path: Path | str = None,
                      rttm_file_name:str = None):
        """
        Loads the RTTM File of a diarized audio file.
        """

        if rttm_dir_path is None:

            rttm_dir_path = self.dump_rttm_dir

        if rttm_file_name is None:

            rttm_file_name = Path(self.input_file_name_to_diar).stem + '.rttm'

        file_path = Path(rttm_dir_path) / rttm_file_name
        self.diarization = load_rttm(str(file_path))

    # ------------------------------------------------------------------- #

    def split(self):
        """
        Splits the audio file based on the recognized speakers and
        saves individual sequences that are assigned to the speakers.
        """

        self.split_dict = split_audio(input_dir=self.input_dir_to_diar,
                                      input_file_name=self.input_file_name_to_diar,
                                      output_dir=self.split_dir,
                                      diarization=self.diarization)

    # ------------------------------------------------------------------- #
    
    def split_dict_to_json(self,
                           dir_path: Path | str = None):
        """
        Saves the split_dict to a JSON file.
        The JSON file contains the file name and the start and end
        time for the splits.

        Args
            dir_path (Path | str): Directory in which the JSON file is to be saved.
        """

        if dir_path is None:

            # Saves the JSON in the main directory of the splits.
            dir_path = OUTPUT_DIR_PATH

        file_path = (dir_path / Path(self.input_file_name_to_diar).stem).with_suffix('.json')

        write_json(file_path,
                   remap_dict_keys(self.split_dict))

    # ------------------------------------------------------------------- #

    def ref_cdist(self,
                  ref_dir_path: Path | str = None,
                  ref_file_name: str = None,
                  split_subdir: Path | str = None,
                  metric: str = 'euclidean',
                  ext='wav',
                  duration_val=5,
                  cuda_switch: bool = False):
        """

        Args:
            ref_dir_path (Path | str): Directory for the reference file.
            ref_file_name (str): File name of the reference file.
            split_subdir (Path | str): Name of the subdirectory in split_dir                
                (default is stem of input_file_name_to_diar).
            metric (str): Metric to compute the distance. Default: 'euclidean'.
            ext (str): Extension of the audio files in the subdirectories.
            duration_val (float): Minimum duration of the audio file.
            cuda_switch (bool): Use CUDA.
        """

        self.ref_file_name = ref_file_name

        if split_subdir is None:

            split_subdir = Path(self.input_file_name_to_diar).stem

        split_dir_path = Path(self.split_dir) / split_subdir

        self.dist_dict = get_speaker_dist(reference_dir_path=ref_dir_path,
                                          reference_file_name=ref_file_name,
                                          split_dir_path=str(split_dir_path),                                            
                                           metric=metric,
                                           ext=ext,
                                           duration_val=duration_val,
                                           cuda_switch=cuda_switch)

    # ------------------------------------------------------------------- #

    def copy_sel_slices(self,
                        input_dir: Path | str = None,
                        output_dir: Path | str = None,
                        dir_file_lst: list = None,
                        speaker_name: str = None):
        """
        Copies files from input_dir to output_dir using the directory structure in input_dir:
        input_dir/audio_dir/speaker_dir/filename.wav, e.g.
        output_audio_splits/audio_file_dir/SPEAKER_00/interview-10002.wav.
    
        Args:
            input_dir (Path | str): This is the Input_dir which is adapted to the structure of the Pyannote output, i.e.:
                input_dir/event_dir/speaker_dir/filename.wav, e.g.
                output_audio_splits/audio_dir/SPEAKER_00/interview-10002.wav.

            output_dir (Path | str): Directory in which the output is to take place.

            dir_file_lst (list): This is a list of tuples. The elements of this list are adapted to the
                output of Pyannote and therefore the input_dir must also have this structure, i.e.:
                tuple = (name of the audio file/dir, speaker ID, Name of the speaker file),
                e.g. (audio_dir, SPEAKER_01, interview-10072.wav)
        """

        if input_dir is None:

            input_dir=self.split_dir

        if output_dir is None:

            output_dir=self.output_sel_slices

        for dir_file_tuple in dir_file_lst:

            copy_audio_slice(input_dir=input_dir,
                             output_dir=output_dir,
                             dir_file_tuple=dir_file_tuple,
                             speaker_name=speaker_name)