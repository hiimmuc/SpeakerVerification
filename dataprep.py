import argparse
import csv
import glob
import hashlib
import os

import random
import subprocess
import tarfile

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import soundfile as sf
from scipy.io import wavfile
from pydub import AudioSegment
from tqdm import tqdm
from multiprocessing import Pool

from processing.audio_loader import AugmentWAV, loadWAV
from processing.dataset import get_audio_properties, read_blacklist
from processing.wav_conversion import convert_audio_shell
from utils import read_config


def get_audio_path(folder):
    """
    Get the audio path for a given folder

    Args:
        folder ([type]): [description]

    Returns:
        list: [description]
    """
    return glob.glob(os.path.join(folder, '*.wav'))


def md5(fname):
    """
    MD5SUM
    """
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download(args, lines):
    """
    Download with wget
    """
    for line in lines:
        url = line.split()[0]
        md5gt = line.split()[1]
        outfile = url.split('/')[-1]

        # Download files
        out = subprocess.call('wget %s -O %s/%s' %
                              (url, args.save_path, outfile),
                              shell=True)
        if out != 0:
            raise ValueError(
                'Download failed %s. If download fails repeatedly, use alternate URL on the VoxCeleb website.'
                % url)

        # Check MD5
        md5ck = md5('%s/%s' % (args.save_path, outfile))
        if md5ck == md5gt:
            print('Checksum successful %s.' % outfile)
        else:
            raise Warning('Checksum failed %s.' % outfile)


def full_extract(args, fname):
    """
    Extract zip files
    """
    print('Extracting %s' % fname)
    if fname.endswith(".tar.gz"):
        with tarfile.open(fname, "r:gz") as tar:
            tar.extractall(args.save_path)
    elif fname.endswith(".zip"):
        with ZipFile(fname, 'r') as zf:
            zf.extractall(args.save_path)


def part_extract(args, fname, target):
    """
    Partially extract zip files
    """
    print('Extracting %s' % fname)
    with ZipFile(fname, 'r') as zf:
        for infile in zf.namelist():
            if any([infile.startswith(x) for x in target]):
                zf.extract(infile, args.save_path)


def split_musan(args):
    """
    Split MUSAN for faster random access
    """

    files = glob.glob('%s/musan/*/*/*.wav' % args.noise_folder)

    audlen = 16000 * 5
    audstr = 16000 * 3

    for idx, file in enumerate(tqdm(files)):
        fs, aud = wavfile.read(file)
        writedir = os.path.splitext(file.replace('/musan/',
                                                 '/musan_split/'))[0]
        os.makedirs(writedir)
        for st in range(0, len(aud) - audlen, audstr):
            wavfile.write(writedir + '/%05d.wav' % (st / fs), fs,
                          aud[st:st + audlen])


def prepare_augmentation(args):
    """
    Check wether the augmentation dataset is already downloaded

    Args:
        args ([type]): [description]
    """
    split_musan(args)
#     if all(os.path.exists(path) for _, path in args.augment_options['augment_paths'].items()):
#         print('Downloading augmentation dataset...')
#         if os.path.exists(f'{args.noise_folder}/augment.txt'):
#             with open(f'{args.noise_folder}/augment.txt', 'r') as f:
#                 augfiles = f.readlines()
#             download(args, augfiles)

# #             full_extract(args, os.path.join(
# #                 args.augment_path, 'rirs_noises.zip'))

# #             full_extract(args, os.path.join(args.augment_path, 'musan.tar.gz'))

#             split_musan(args)
#     else:
#         print('Augmentation dataset already exists in',
#               f'{args.noise_folder}/augment.txt', 'r')


def concatenate(args, lines):
    # ========== ===========
    # Concatenate file parts
    # ========== ===========

    for line in tqdm(lines):
        infile = line.split()[0]
        outfile = line.split()[1]
        md5gt = line.split()[2]

        # Concatenate files
        out = subprocess.call(
            'cat %s/%s > %s/%s' % (args.save_path, infile, args.save_path, outfile), shell=True)

        # Check MD5
        md5ck = md5('%s/%s' % (args.save_path, outfile))
        if md5ck == md5gt:
            print('Checksum successful %s.' % outfile)
        else:
            raise Warning('Checksum failed %s.' % outfile)

        out = subprocess.call('rm %s/%s' %
                              (args.save_path, infile), shell=True)


def convert_aac_wav(fname):
    outfile = fname.replace('.m4a', '.wav').replace('aac/', 'wav/')
    os.makedirs(Path(outfile).parent, exist_ok=True)
    out = subprocess.call(
        'ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s >/dev/null 2>/dev/null' % (fname, outfile), shell=True)
    if out != 0:
        raise ValueError('Conversion failed %s.' % fname)


def convert_voxceleb(args):
    # dataset/train_data/VoxCeleb/ voxceleb2/id00012/_raOc3-IRsw/00110.m4a
    files = glob.glob('%s/voxceleb2/*/*/*/*.m4a' % args.save_path)
    files.sort()
    print('Converting files from AAC to WAV')

    p = Pool(processes=96)
    results = p.map(convert_aac_wav, files)
    p.close()
    p.join()

    print('-----------')


def augmentation(args, audio_paths, mode='train', max_frames=200, step_save=5, **kwargs):
    """
    Perfrom augmentation on the raw dataset
    """

    prepare_augmentation(args)  # check if augumentation data is ready

    aug_rate = args.aug_rate
    # musan_path = str(os.path.join(args.augment_path, '/musan_split'))
    # rir_path = str(os.path.join(args.augment_path, '/RIRS_NOISES/simulated_rirs'))
    musan_path = "dataset/augment_data/musan_split"
    rir_path = "dataset/augment_data/RIRS_NOISES/simulated_rirs"
    print('Start augmenting data with', musan_path, 'and', rir_path)

    if mode == 'train':
        print('Augment Full')
        num_aug = len(audio_paths)
        augment_audio_paths = audio_paths
    elif mode == 'test':
        num_aug = int(aug_rate * len(audio_paths))
        random_indices = random.sample(range(len(audio_paths)), num_aug)
        augment_audio_paths = [audio_paths[i] for i in random_indices]
    else:
        raise ValueError('mode should be train or test')

    print('Number of augmented data: {}/{}'.format(num_aug, len(audio_paths)))

    augment_engine = AugmentWAV(musan_path=musan_path,
                                rir_path=rir_path,
                                max_frames=max_frames,
                                sample_rate=8000, target_db=None)

    for idx, fpath in enumerate(tqdm(augment_audio_paths, unit='files', desc=f"Augmented process")):

        audio = loadWAV(fpath, max_frames=max_frames,
                        evalmode=False,
                        augment=False,
                        sample_rate=8000,
                        augment_chain=[])

        modes = ['music', 'speech', 'noise', 'noise_vad']
        p_base = [0.25, 0.25, 0.25, 0.25]
        # augment types
        aug_rev = np.squeeze(augment_engine.reverberate(audio))
        aug_noise_music = np.squeeze(
            augment_engine.additive_noise('music', audio))
        aug_noise_speech = np.squeeze(
            augment_engine.additive_noise('speech', audio))
        aug_noise_noise = np.squeeze(
            augment_engine.additive_noise('noise', audio))
        aug_noise_noise_vad = np.squeeze(
            augment_engine.additive_noise('noise_vad', audio))
        aug_both = np.squeeze(augment_engine.reverberate(
            augment_engine.additive_noise(np.random.choice(modes, p=p_base), audio)))
        list_audio = [[aug_rev, 'rev'],
                      [aug_noise_music, 'music'],
                      [aug_noise_speech, 'speech'],
                      [aug_noise_noise, 'noise'],
                      [aug_noise_noise_vad, 'noise_vad'],
                      [aug_both, 'both']]

        for audio_data, aug_t in list_audio:
            save_path = os.path.join(
                f"{fpath.replace('.wav', '')}_augmented_{aug_t}.wav")

            if os.path.exists(save_path):
                os.remove(save_path)
            sf.write(str(save_path), audio_data, 8000)

    print('Done!')


class DataGenerator():
    def __init__(self, args, **kwargs):
        self.args = args
        self.data_folder = args.data_folder
        self.spkID_list = glob.glob(str(Path(self.data_folder + '/wav/*')))

    def convert(self):
        # convert data to one form 16000Hz, only works on Linux
        spk_files = self.spkID_list
        spk_files.sort()
        audio_spec = self.args.audio_spec

        files = []
        for spk in spk_files:
            files += list(Path(spk).glob('*.wav'))

        print(f"Converting process, Total: {len(files)}/{len(spk_files)}")

        for fpath in tqdm(files):
            convert_audio_shell(src=fpath,
                                sample_rate=audio_spec['sample_rate'],
                                channels=audio_spec['sample_rate'])
        print('Done!')

    def generate_metadata(self, num_spks=-1, lower_num=40, upper_num=-1):
        """
        Generate train test lists for generate_metadata data
        """
        valid_spks = []
        invalid_spks = []

        root = Path(self.data_folder) / 'wav'

        classpaths = [d for d in root.iterdir() if d.is_dir()]
        classpaths.sort()

        if num_spks == -1:
            num_spks = len(classpaths)

        print('Generate dataset metadata files, total:', num_spks)
        print("Minimum utterances per speaker required:", lower_num)
        print("Maximum utterances per speaker required:", upper_num)

        train_filepaths_list = []
        val_filepaths_list = []

        loader_bar = tqdm(list(classpaths)[:], desc="Processing:...")
        for classpath in loader_bar:

            filepaths = list(classpath.glob('*.wav'))
            # check duration, sr
            filepaths = check_valid_audio(
                filepaths, self.args.audio_spec['sentence_len'] * 0.5, self.args.audio_spec['sample_rate'])

            # checknumber of files
            if len(filepaths) < lower_num:
                continue
            elif upper_num > 0:
                if len(filepaths) >= upper_num:
                    filepaths = filepaths[:upper_num]
            if len(filepaths) == 0:
                continue

            valid_spks.append(str(Path(classpath)))

            random.shuffle(filepaths)

            # 3 utterances per speaker for val
            val_num = min(3, len(filepaths))

            if self.args.dataloader_options['split_ratio'] > 0:
                val_num = int(
                    self.args.dataloader_options['split_ratio'] * len(filepaths))

            val_filepaths = random.sample(filepaths, val_num)

            train_filepaths = list(set(filepaths).difference(
                set(val_filepaths))) if self.args.dataloader_options['split_ratio'] > 0 else filepaths

            # write train file
            for train_filepath in train_filepaths:
                spkID = str(train_filepath.parent.stem.split('-')[0])
                ext = str(train_filepath).split('.')[-1]
                duration, rate = get_audio_properties(str(train_filepath))
                train_filepaths_list.append(
                    [spkID, str(train_filepath), duration, ext])

            val_filepaths_list.append(val_filepaths)

            # break when reach numer of maximum spk
            if len(valid_spks) >= num_spks:
                break

            # set post fix for tqdm
            loader_bar.set_postfix(
                Valid_speakers=f" {len(valid_spks)} speakers")

        ######################## gathering val files #########################
        val_pairs = []
        for val_filepaths in tqdm(val_filepaths_list, desc="Generating validation file..."):
            for i in range(len(val_filepaths) - 1):
                for j in range(i + 1, len(val_filepaths)):
                    # positive pairs
                    label = '1'
                    val_pairs.append(
                        [label, str(val_filepaths[i]), str(val_filepaths[j])])

                    label = '0'
                    while True:
                        x = random.randint(0, len(val_filepaths_list) - 1)
                        if not val_filepaths_list[x]:
                            continue
                        if val_filepaths_list[x][0].parent.stem != val_filepaths[i].parent.stem:
                            break

                    y = random.randint(0, len(val_filepaths_list[x]) - 1)
                    # negative pairs
                    val_pairs.append(
                        [label, str(val_filepaths[i]), str(val_filepaths_list[x][y])])

        # write train and val files
        print("Generating metadata files...")
        os.makedirs(Path(root.parent / 'metadata'), exist_ok=True)
        with open(Path(root.parent, f'metadata/train.csv'), 'w', newline='') as wf:
            spamwriter = csv.writer(wf, delimiter=',')
            spamwriter.writerow(['ID', 'path', 'duration', 'audio_format'])
            for spkid, path, duration, audio_format in train_filepaths_list:
                spamwriter.writerow([spkid, path, duration, audio_format])

        with open(Path(root.parent, f'metadata/valid.csv'), 'w', newline='') as wf:
            spamwriter = csv.writer(wf, delimiter=',')
            spamwriter.writerow(['label', 'audio1', 'audio2'])
            for label, audio1, audio2 in val_pairs:
                spamwriter.writerow([label, audio1, audio2])

        # copy to save dir
        os.makedirs(Path(self.args.train_annotation).parent, exist_ok=True)
        subprocess.call(
            f"cp {Path(root.parent, f'metadata/train.csv')} {str(Path(self.args.train_annotation))}", shell=True)
        subprocess.call(
            f"cp {Path(root.parent, f'metadata/valid.csv')} {str(Path(self.args.valid_annotation))}", shell=True)
        # some information
        print("Valid speakers:", len(valid_spks))
        print("Valid audio files:", len(train_filepaths_list))
        print("Validation pairs:", len(val_pairs))

        return valid_spks, invalid_spks


def check_valid_audio(files, duration_lim=1.5, sr=8000):
    filtered_list = []
    files = [str(path) for path in files]

    for fname in files:
        duration, rate = get_audio_properties(fname)
        if rate == sr and duration >= duration_lim:
            filtered_list.append(fname)
        else:
            pass
    filtered_list.sort(reverse=True, key=lambda x: get_audio_properties(x)[0])
    filtered_list = [Path(path) for path in filtered_list]
    return filtered_list


def restore_dataset(raw_dataset, **kwargs):
    raw_data_dir = raw_dataset

    data_paths = []
    for fdir in tqdm(os.listdir(raw_data_dir), desc="Checking directory..."):
        data_paths.extend(
            glob.glob(os.path.join(raw_data_dir, f'{fdir}/*.wav')))

    raw_paths = list(
        filter(lambda x: 'augment' not in str(x) and 'vad' not in str(x), data_paths))
#     extended_paths = list(filter(lambda x: x not in raw_paths, data_paths))
    augment_paths = list(filter(lambda x: 'augment' in str(x), data_paths))
    vad_paths = list(filter(lambda x: 'vad' in str(x), data_paths))

    print(len(raw_paths))
    print(len(augment_paths), '/', len(vad_paths))

    for audio_path in tqdm(augment_paths):
        if os.path.isfile(audio_path):
            os.remove(audio_path)
            pass
    for audio_path in tqdm(vad_paths):
        if os.path.isfile(audio_path):
            #             os.remove(audio_path)
            pass


try:
    from processing.vad_tool import VAD

    def vad_on_dataset(raw_dataset):
        raw_data_dir = raw_dataset
        vad_engine = VAD()

        data_paths = []
        for fdir in os.listdir(raw_data_dir):
            data_paths.extend(
                glob.glob(os.path.join(raw_data_dir, f'{fdir}/*.wav')))

        # filters audiopaths
        raw_paths = list(
            filter(lambda x: 'augment' not in str(x) and 'vad' not in str(x), data_paths))

        for audio_path in tqdm(raw_paths):
            vad_engine.detect(audio_path, duration_min=1.5)
        print("Done!")
except:
    print('can not import vad_tools')

    def vad_on_dataset(raw_dataset):
        pass

parser = argparse.ArgumentParser(description="Data preparation")
if __name__ == '__main__':
    # ========== ===========
    # Parse input arguments
    # ========== ===========
    parser.add_argument('--save_path', 	type=str,
                        default="dataset/train_data/VoxCeleb", help='Target directory')
    parser.add_argument('--user', 		type=str, default="user", help='Username')
    parser.add_argument('--password', 	type=str,
                        default="pass", help='Password')

    parser.add_argument('--download', dest='download',
                        action='store_true', help='Enable download')
    parser.add_argument('--extract', dest='extract',
                        action='store_true', help='Enable extract')
    # YAML
    parser.add_argument('--config', type=str, default=None)
    ##
    parser.add_argument('--details_dir',
                        type=str,
                        default='dataset/train_details_full/',
                        help='Download and extract augmentation files')

    parser.add_argument('--split_ratio',
                        type=float,
                        default=-1,
                        help='Split ratio')
    parser.add_argument('--num_spks',
                        type=int,
                        default=-1,
                        help='number of speaker')
    parser.add_argument('--lower_num',
                        type=int,
                        default=10,
                        help='lower_num of speaker')
    parser.add_argument('--upper_num',
                        type=int,
                        default=-1,
                        help='upper_num of speaker')
    # mode
    parser.add_argument('--convert',
                        default=False,
                        action='store_true',
                        help='Enable coversion')
    parser.add_argument('--generate',
                        default=False,
                        action='store_true',
                        help='Enable generate')
    parser.add_argument('--restore',
                        default=False,
                        action='store_true',
                        help='Restore dataset to origin(del augment and vad)')
    # augmentation
    parser.add_argument('--augment',
                        default=False,
                        action='store_true',
                        help='Download and extract augmentation files')
    parser.add_argument('--augment_mode',
                        type=str,
                        default='train',
                        help='')
    parser.add_argument('--augment_path',
                        type=str,
                        default='dataset/augment_data',
                        help='Directory include augmented data')
    parser.add_argument('--aug_rate',
                        type=float,
                        default=0.5,
                        help='')

    args = parser.parse_args()

    if args.config is not None:
        args = read_config(args.config, args)
        args = argparse.Namespace(**args)

    print('Start processing...')
    if not os.path.exists(args.save_path):
        raise ValueError('Target directory does not exist.')

    if args.generate:
        data_generator = DataGenerator(args)
        data_generator.generate_metadata(
            num_spks=args.num_spks, lower_num=args.lower_num, upper_num=args.upper_num)
    if args.restore:
        restore_dataset(args.data_folder)

    if args.augment:
        f = open('lists/augment.txt', 'r')
        augfiles = f.readlines()
        f.close()
        download(args, augfiles)
        part_extract(args, os.path.join(args.save_path, 'rirs_noises.zip'), [
                     'RIRS_NOISES/simulated_rirs/mediumroom', 'RIRS_NOISES/simulated_rirs/smallroom'])
        full_extract(args, os.path.join(args.save_path, 'musan.tar.gz'))
        split_musan(args)
        # augmentation(
        #     args=args, audio_paths=data_generator.spkID_list, step_save=100, mode=args.augment_mode)

    if args.download:
        f = open('lists/fileparts.txt', 'r')
        fileparts = f.readlines()
        f.close()
        download(args, fileparts)

    if args.extract:
        f = open('lists/files.txt', 'r')
        files = f.readlines()
        f.close()
        concatenate(args, files)
        for file in tqdm(files):
            full_extract(args, os.path.join(args.save_path, file.split()[1]))
        out = subprocess.call('mv %s/dev/aac/* %s/aac/ && rm -r %s/dev' %
                              (args.save_path, args.save_path, args.save_path), shell=True)
        out = subprocess.call('mv %s/wav %s/voxceleb1' %
                              (args.save_path, args.save_path), shell=True)
        out = subprocess.call('mv %s/dev/aac %s/voxceleb2' %
                              (args.save_path, args.save_path), shell=True)

    if args.convert:
        print('Converting...')
        convert_voxceleb(args)
