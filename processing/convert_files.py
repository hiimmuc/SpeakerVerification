import glob
from pydub import AudioSegment
from multiprocessing import Pool


def convert_audio(src):
    """Convert audio format and samplerate to target"""
    ext = 'wav'
    sample_rate = 8000
    channels = 1
    codec = 'pcm_s16le'
    dst = None
    try:
        org_format = src.split('.')[-1].strip()
        if ext != org_format:
            audio = AudioSegment.from_file(src)
            # export file as new format
            src = src.replace(org_format, ext)
            audio.export(src, format=ext)
    except Exception as e:
        raise e

    try:
        sound = AudioSegment.from_file(src, format='wav')
        sound = sound.set_frame_rate(sample_rate)
        sound = sound.set_channels(channels)

        dst = src if not dst else dst

        sound.export(dst, format='wav')
    except Exception as e:
        raise e


audio_list = glob.glob("dataset/**/*.wav", recursive=True)
p = Pool(processes=12)
p.map(convert_audio, audio_list)
p.close()
p.join()
print('-------')
