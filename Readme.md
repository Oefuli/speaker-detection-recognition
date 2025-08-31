# The Algorithm

This algorithm offers the following functions: You can pass an audio file to it,  
and it will use pyannote to attempt to recognize the individual speakers in the  
audio file and separate them from each other. The cut-out snippets belonging to  
the speakers are stored in separate folders. You can then use an audio file that  
serves as a reference for a person to go through the snippets and search for a person.  

# Core Technologies
This algorithm is based on pyannote and especially on pyannote-audio:  
- [pyannote](https://github.com/pyannote)  
- [pyannote.audio](https://github.com/pyannote/pyannote-audio): pip install pyannote.audio  
- pyannote-audio was published under an [MIT License](https://github.com/pyannote/pyannote-audio/blob/main/LICENSE)

The system is very powerful for the size of the [model (5.91MB)](https://huggingface.co/pyannote/segmentation-3.0/tree/main)

[Offline use](https://github.com/pyannote/pyannote-audio/tree/develop/tutorials): Have a look at applying_a_model.ipynb and applying_a_pipeline.ipynb.  

# Get Started
## Install on Ubuntu
- open terminal in the folder where the TOML file is located  
- pip install .  

## Conditions for the Audio Files
- **The format must be wav.**
- For the other parameters, I usually use:
    - Samplerate: 44100  
    - Bit: PCM_16 (16 Bit)  
    - Channels: 2  



