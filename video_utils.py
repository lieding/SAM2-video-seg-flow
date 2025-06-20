
import hashlib
import os
import re
import shutil
import subprocess
import time
from typing import Mapping
import uuid
from venv import logger
from comfy_utils import ProgressBar, common_upscale
import torch
import numpy as np
import itertools
import psutil
import cv2


BIGMIN = -(2**53-1)
BIGMAX = (2**53-1)

DIMMAX = 8192

ENCODE_ARGS = ("utf-8", 'backslashreplace')

temp_directory = './download_videos'

VHSLoadFormats = {
    'None': {},
    'AnimateDiff': {'target_rate': 8, 'dim': (8,0,512,512)},
    'Mochi': {'target_rate': 24, 'dim': (16,0,848,480), 'frames':(6,1)},
    'LTXV': {'target_rate': 24, 'dim': (32,0,768,512), 'frames':(8,1)},
    'Hunyuan': {'target_rate': 24, 'dim': (16,0,848,480), 'frames':(4,1)},
    'Cosmos': {'target_rate': 24, 'dim': (16,0,1280,704), 'frames':(8,1)},
    'Wan': {'target_rate': 16, 'dim': (8,0,832,480), 'frames':(4,1)},
}

def ffmpeg_suitability(path):
    try:
        version = subprocess.run([path, "-version"], check=True,
                                 capture_output=True).stdout.decode(*ENCODE_ARGS)
    except:
        return 0
    score = 0
    #rough layout of the importance of various features
    simple_criterion = [("libvpx", 20),("264",10), ("265",3),
                        ("svtav1",5),("libopus", 1)]
    for criterion in simple_criterion:
        if version.find(criterion[0]) >= 0:
            score += criterion[1]
    #obtain rough compile year from copyright information
    copyright_index = version.find('2000-2')
    if copyright_index >= 0:
        copyright_year = version[copyright_index+6:copyright_index+9]
        if copyright_year.isnumeric():
            score += int(copyright_year)
    return score

if "VHS_FORCE_FFMPEG_PATH" in os.environ:
    ffmpeg_path = os.environ.get("VHS_FORCE_FFMPEG_PATH")
else:
    ffmpeg_paths = []
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        imageio_ffmpeg_path = get_ffmpeg_exe()
        ffmpeg_paths.append(imageio_ffmpeg_path)
    except:
        if "VHS_USE_IMAGEIO_FFMPEG" in os.environ:
            raise
        logger.warn("Failed to import imageio_ffmpeg")
    if "VHS_USE_IMAGEIO_FFMPEG" in os.environ:
        ffmpeg_path = imageio_ffmpeg_path
    else:
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is not None:
            ffmpeg_paths.append(system_ffmpeg)
        if os.path.isfile("ffmpeg"):
            ffmpeg_paths.append(os.path.abspath("ffmpeg"))
        if os.path.isfile("ffmpeg.exe"):
            ffmpeg_paths.append(os.path.abspath("ffmpeg.exe"))
        if len(ffmpeg_paths) == 0:
            logger.error("No valid ffmpeg found.")
            ffmpeg_path = None
        elif len(ffmpeg_paths) == 1:
            #Evaluation of suitability isn't required, can take sole option
            #to reduce startup time
            ffmpeg_path = ffmpeg_paths[0]
        else:
            ffmpeg_path = max(ffmpeg_paths, key=ffmpeg_suitability)

def get_ffmpeg_path():
    return ffmpeg_path

class PromptServer:
    number = 0
    
    def __init__(self):
        PromptServer.instance = self
        execution.PromptQueue(self)

requeue_guard = [None, 0, 0, {}]

def target_size(width, height, custom_width, custom_height, downscale_ratio=8) -> tuple[int, int]:
    if downscale_ratio is None:
        downscale_ratio = 8
    if custom_width == 0 and custom_height ==  0:
        pass
    elif custom_height == 0:
        height *= custom_width/width
        width = custom_width
    elif custom_width == 0:
        width *= custom_height/height
        height = custom_height
    else:
        width = custom_width
        height = custom_height
    width = int(width/downscale_ratio + 0.5) * downscale_ratio
    height = int(height/downscale_ratio + 0.5) * downscale_ratio
    return (width, height)

def cv_frame_generator(video, force_rate, frame_load_cap, skip_first_frames,
                       select_every_nth, meta_batch=None, unique_id=None):
    video_cap = cv2.VideoCapture(video)
    if not video_cap.isOpened() or not video_cap.grab():
        raise ValueError(f"{video} could not be loaded with cv.")

    # extract video metadata
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    width = 0

    if width <=0 or height <=0:
        _, frame = video_cap.retrieve()
        height, width, _ = frame.shape

    # set video_cap to look at start_index frame
    total_frame_count = 0
    total_frames_evaluated = -1
    frames_added = 0
    base_frame_time = 1 / fps
    prev_frame = None

    if force_rate == 0:
        target_frame_time = base_frame_time
    else:
        target_frame_time = 1/force_rate

    if total_frames > 0:
        if force_rate != 0:
            yieldable_frames = int(total_frames / fps * force_rate)
        else:
            yieldable_frames = total_frames
        if select_every_nth:
            yieldable_frames //= select_every_nth
        if frame_load_cap != 0:
            yieldable_frames =  min(frame_load_cap, yieldable_frames)
    else:
        yieldable_frames = 0
    yield (width, height, fps, duration, total_frames, target_frame_time, yieldable_frames)
    pbar = ProgressBar(yieldable_frames)
    time_offset=target_frame_time
    while video_cap.isOpened():
        if time_offset < target_frame_time:
            is_returned = video_cap.grab()
            # if didn't return frame, video has ended
            if not is_returned:
                break
            time_offset += base_frame_time
        if time_offset < target_frame_time:
            continue
        time_offset -= target_frame_time
        # if not at start_index, skip doing anything with frame
        total_frame_count += 1
        if total_frame_count <= skip_first_frames:
            continue
        else:
            total_frames_evaluated += 1

        # if should not be selected, skip doing anything with frame
        if total_frames_evaluated%select_every_nth != 0:
            continue

        # opencv loads images in BGR format (yuck), so need to convert to RGB for ComfyUI use
        # follow up: can videos ever have an alpha channel?
        # To my testing: No. opencv has no support for alpha
        unused, frame = video_cap.retrieve()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # convert frame to comfyui's expected format
        # TODO: frame contains no exif information. Check if opencv2 has already applied
        frame = np.array(frame, dtype=np.float32)
        torch.from_numpy(frame).div_(255)
        if prev_frame is not None:
            inp  = yield prev_frame
            if inp is not None:
                #ensure the finally block is called
                return
        prev_frame = frame
        frames_added += 1
        if pbar is not None:
            pbar.update_absolute(frames_added, yieldable_frames)
        # if cap exists and we've reached it, stop processing frames
        if frame_load_cap > 0 and frames_added >= frame_load_cap:
            break
    if meta_batch is not None:
        meta_batch.inputs.pop(unique_id)
        meta_batch.has_closed_inputs = True
    if prev_frame is not None:
        yield prev_frame

def ffmpeg_frame_generator(video, force_rate, frame_load_cap, start_time,
                           custom_width, custom_height, downscale_ratio=8,
                           meta_batch=None, unique_id=None):
    args_dummy = [ffmpeg_path, "-i", video, '-c', 'copy', '-frames:v', '1', "-f", "null", "-"]
    size_base = None
    fps_base = None
    try:
        dummy_res = subprocess.run(args_dummy, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        raise Exception("An error occurred in the ffmpeg subprocess:\n" \
                + e.stderr.decode(*ENCODE_ARGS))
    lines = dummy_res.stderr.decode(*ENCODE_ARGS)

    for line in lines.split('\n'):
        match = re.search("^ *Stream .* Video.*, ([1-9]|\\d{2,})x(\\d+)", line)
        if match is not None:
            size_base = [int(match.group(1)), int(match.group(2))]
            fps_match = re.search(", ([\\d\\.]+) fps", line)
            if fps_match:
                fps_base = float(fps_match.group(1))
            else:
                fps_base = 1
            alpha = re.search("(yuva|rgba)", line) is not None
            break
    else:
        raise Exception("Failed to parse video/image information. FFMPEG output:\n" + lines)

    durs_match = re.search("Duration: (\\d+:\\d+:\\d+\\.\\d+),", lines)
    if durs_match:
        durs = durs_match.group(1).split(':')
        duration = int(durs[0])*360 + int(durs[1])*60 + float(durs[2])
    else:
        duration = 0

    if start_time > 0:
        if start_time > 4:
            post_seek = ['-ss', '4']
            pre_seek = ['-ss', str(start_time - 4)]
        else:
            post_seek = ['-ss', str(start_time)]
            pre_seek = []
    else:
        pre_seek = []
        post_seek = []
    args_all_frames = [ffmpeg_path, "-v", "error", "-an"] + pre_seek + \
            ["-i", video, "-pix_fmt", "rgba64le"] + post_seek

    vfilters = []
    if force_rate != 0:
        vfilters.append("fps=fps="+str(force_rate))
    if custom_width != 0 or custom_height != 0:
        size = target_size(size_base[0], size_base[1], custom_width,
                           custom_height, downscale_ratio=downscale_ratio)
        ar = float(size[0])/float(size[1])
        if abs(size_base[0]*ar-size_base[1]) >= 1:
            #Aspect ratio is changed. Crop to new aspect ratio before scale
            vfilters.append(f"crop=if(gt({ar}\\,a)\\,iw\\,ih*{ar}):if(gt({ar}\\,a)\\,iw/{ar}\\,ih)")
        size_arg = ':'.join(map(str,size))
        vfilters.append(f"scale={size_arg}")
    else:
        size = size_base
    if len(vfilters) > 0:
        args_all_frames += ["-vf", ",".join(vfilters)]
    yieldable_frames = (force_rate or fps_base)*duration
    if frame_load_cap > 0:
        args_all_frames += ["-frames:v", str(frame_load_cap)]
        yieldable_frames = min(yieldable_frames, frame_load_cap)
    yield (size_base[0], size_base[1], fps_base, duration, fps_base * duration,
           1/(force_rate or fps_base), yieldable_frames, size[0], size[1], alpha)

    args_all_frames += ["-f", "rawvideo", "-"]
    pbar = ProgressBar(yieldable_frames)
    try:
        with subprocess.Popen(args_all_frames, stdout=subprocess.PIPE) as proc:
            #Manually buffer enough bytes for an image
            bpi = size[0] * size[1] * 8
            current_bytes = bytearray(bpi)
            current_offset=0
            prev_frame = None
            while True:
                bytes_read = proc.stdout.read(bpi - current_offset) # type: ignore
                if bytes_read is None:#sleep to wait for more data
                    time.sleep(.1)
                    continue
                if len(bytes_read) == 0:#EOF
                    break
                current_bytes[current_offset:len(bytes_read)] = bytes_read
                current_offset+=len(bytes_read)
                if current_offset == bpi:
                    if prev_frame is not None:
                        yield prev_frame
                        pbar.update(1)
                    prev_frame = np.frombuffer(current_bytes, dtype=np.dtype(np.uint16).newbyteorder("<")).reshape(size[1], size[0], 4) / (2**16-1)
                    if not alpha:
                        prev_frame = prev_frame[:, :, :-1]
                    current_offset = 0
    except BrokenPipeError as e:
        raise Exception("An error occured in the ffmpeg subprocess:\n" \
                + proc.stderr.read().decode(*ENCODE_ARGS)) # type: ignore
    if meta_batch is not None:
        meta_batch.inputs.pop(unique_id)
        meta_batch.has_closed_inputs = True
    if prev_frame is not None:
        yield prev_frame

def resized_cv_frame_gen(custom_width, custom_height, downscale_ratio, **kwargs):
    gen = cv_frame_generator(**kwargs)
    info =  next(gen)
    width, height = info[0], info[1]
    frames_per_batch = (1920 * 1080 * 16) // (width * height) or 1
    if kwargs.get('meta_batch', None) is not None:
        frames_per_batch = min(frames_per_batch, kwargs['meta_batch'].frames_per_batch)
    if custom_width != 0 or custom_height != 0 or downscale_ratio is not None:
        new_size = target_size(width, height, custom_width, custom_height, downscale_ratio)
        yield (*info, new_size[0], new_size[1], False)
        if new_size[0] != width or new_size[1] != height:
            def rescale(frame):
                s = torch.from_numpy(np.fromiter(frame, np.dtype((np.float32, (height, width, 3)))))
                s = s.movedim(-1,1)
                s = common_upscale(s, new_size[0], new_size[1], "lanczos", "center")
                return s.movedim(1,-1).numpy()
            yield from itertools.chain.from_iterable(map(rescale, batched(gen, frames_per_batch)))
            return
    else:
        yield (*info, info[0], info[1], False)
    yield from gen

#Python 3.12 adds an itertools.batched, but it's easily replicated for legacy support
def batched(it, n):
    while batch := tuple(itertools.islice(it, n)):
        yield batch
def batched_vae_encode(images, vae, frames_per_batch):
    for batch in batched(images, frames_per_batch):
        image_batch = torch.from_numpy(np.array(batch))
        yield from vae.encode(image_batch).numpy()

def is_safe_path(path, strict=False):
    if "VHS_STRICT_PATHS" not in os.environ and not strict:
        return True
    basedir = os.path.abspath('.')
    try:
        common_path = os.path.commonpath([basedir, path])
    except:
        #Different drive on windows
        return False
    return common_path == basedir

def strip_path(path):
    #This leaves whitespace inside quotes and only a single "
    #thus ' ""test"' -> '"test'
    #consider path.strip(string.whitespace+"\"")
    #or weightier re.fullmatch("[\\s\"]*(.+?)[\\s\"]*", path).group(1)
    path = path.strip()
    if path.startswith("\""):
        path = path[1:]
    if path.endswith("\""):
        path = path[:-1]
    return path

# modified from https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
def calculate_file_hash(filename: str, hash_every_n: int = 1):
    #Larger video files were taking >.5 seconds to hash even when cached,
    #so instead the modified time from the filesystem is used as a hash
    h = hashlib.sha256()
    h.update(filename.encode())
    h.update(str(os.path.getmtime(filename)).encode())
    return h.hexdigest()


def hash_path(path):
    if path is None:
        return "input"
    if is_url(path):
        return "url"
    return calculate_file_hash(strip_path(path))

def get_audio(file, start_time=0, duration=0):
    args = [ffmpeg_path, "-i", file]
    if start_time > 0:
        args += ["-ss", str(start_time)]
    if duration > 0:
        args += ["-t", str(duration)]
    try:
        #TODO: scan for sample rate and maintain
        res =  subprocess.run(args + ["-f", "f32le", "-"],
                              capture_output=True, check=True)
        audio = torch.frombuffer(bytearray(res.stdout), dtype=torch.float32)
        match = re.search(', (\\d+) Hz, (\\w+), ',res.stderr.decode(*ENCODE_ARGS))
    except subprocess.CalledProcessError as e:
        raise Exception(f"VHS failed to extract audio from {file}:\n" \
                + e.stderr.decode(*ENCODE_ARGS))
    if match:
        ar = int(match.group(1))
        #NOTE: Just throwing an error for other channel types right now
        #Will deal with issues if they come
        ac = {"mono": 1, "stereo": 2}[match.group(2)]
    else:
        ar = 44100
        ac = 2
    audio = audio.reshape((-1,ac)).transpose(0,1).unsqueeze(0)
    return {'waveform': audio, 'sample_rate': ar}

class LazyAudioMap(Mapping):
    def __init__(self, file, start_time, duration):
        self.file = file
        self.start_time=start_time
        self.duration=duration
        self._dict=None
    def __getitem__(self, key):
        if self._dict is None:
            self._dict = get_audio(self.file, self.start_time, self.duration)
        return self._dict[key]
    def __iter__(self):
        if self._dict is None:
            self._dict = get_audio(self.file, self.start_time, self.duration)
        return iter(self._dict)
    def __len__(self):
        if self._dict is None:
            self._dict = get_audio(self.file, self.start_time, self.duration)
        return len(self._dict)

def is_url(url):
    return url.split("://")[0] in ["http", "https"]

def lazy_get_audio(file, start_time=0, duration=0, **kwargs):
    return LazyAudioMap(file, start_time, duration)

"""
External plugins may add additional formats to nodes.VHSLoadFormats
In addition to shorthand options, direct widget names will map a given dict to options.
Adding a third arguement to a frames tuple can enable strict checks on number
of loaded frames, i.e (8,1,True)
"""
nodes = {}
if not hasattr(nodes, 'VHSLoadFormats'):
    nodes["VHSLoadFormats"] = {}

def get_format(format):
    if format in VHSLoadFormats:
        return VHSLoadFormats[format]
    return nodes["VHSLoadFormats"].get(format, {})


def load_video(meta_batch=None, unique_id=None, memory_limit_mb=None, vae=None,
               generator=resized_cv_frame_gen, format='None',  **kwargs):
    if 'force_size' in kwargs:
        kwargs.pop('force_size')
        logger.warn("force_size has been removed. Did you reload the webpage after updating?")
    format = get_format(format)
    kwargs['video'] = strip_path(kwargs['video'])
    if vae is not None:
        downscale_ratio = getattr(vae, "downscale_ratio", 8)
    else:
        downscale_ratio = format.get('dim', (1,))[0]
    if meta_batch is None or unique_id not in meta_batch.inputs:
        gen = generator(meta_batch=meta_batch, unique_id=unique_id, downscale_ratio=downscale_ratio, **kwargs)
        (width, height, fps, duration, total_frames, target_frame_time, yieldable_frames, new_width, new_height, alpha) = next(gen)

        if meta_batch is not None:
            meta_batch.inputs[unique_id] = (gen, width, height, fps, duration, total_frames, target_frame_time, yieldable_frames, new_width, new_height, alpha)
            if yieldable_frames:
                meta_batch.total_frames = min(meta_batch.total_frames, yieldable_frames)

    else:
        (gen, width, height, fps, duration, total_frames, target_frame_time, yieldable_frames, new_width, new_height, alpha) = meta_batch.inputs[unique_id]

    memory_limit = None
    if memory_limit_mb is not None:
        memory_limit *= 2 ** 20 # type: ignore
    else:
        #TODO: verify if garbage collection should be performed here.
        #leaves ~128 MB unreserved for safety
        try:
            memory_limit = (psutil.virtual_memory().available + psutil.swap_memory().free) - 2 ** 27
        except:
            logger.warn("Failed to calculate available memory. Memory load limit has been disabled")
            memory_limit = BIGMAX
    if vae is not None:
        #space required to load as f32, exist as latent with wiggle room, decode to f32
        max_loadable_frames = int(memory_limit//(width*height*3*(4+4+1/10)))
    else:
        #TODO: use better estimate for when vae is not None
        #Consider completely ignoring for load_latent case?
        max_loadable_frames = int(memory_limit//(width*height*3*(.1)))
    if meta_batch is not None:
        if 'frames' in format:
            if meta_batch.frames_per_batch % format['frames'][0] != format['frames'][1]:
                error = (meta_batch.frames_per_batch - format['frames'][1]) % format['frames'][0]
                suggested = meta_batch.frames_per_batch - error
                if error > format['frames'][0] / 2:
                    suggested += format['frames'][0]
                raise RuntimeError(f"The chosen frames per batch is incompatible with the selected format. Try {suggested}")
        if meta_batch.frames_per_batch > max_loadable_frames:
            raise RuntimeError(f"Meta Batch set to {meta_batch.frames_per_batch} frames but only {max_loadable_frames} can fit in memory")
        gen = itertools.islice(gen, meta_batch.frames_per_batch)
    else:
        original_gen = gen
        gen = itertools.islice(gen, max_loadable_frames)
    frames_per_batch = (1920 * 1080 * 16) // (width * height) or 1
    if vae is not None:
        gen = batched_vae_encode(gen, vae, frames_per_batch)
        vw,vh = new_width//downscale_ratio, new_height//downscale_ratio
        channels = getattr(vae, 'latent_channels', 4)
        images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (channels,vh,vw)))))
    else:
        #Some minor wizardry to eliminate a copy and reduce max memory by a factor of ~2
        images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (new_height, new_width, 4 if alpha else 3)))))
    if meta_batch is None and memory_limit is not None:
        try:
            next(original_gen)
            raise RuntimeError(f"Memory limit hit after loading {len(images)} frames. Stopping execution.")
        except StopIteration:
            pass
    if len(images) == 0:
        raise RuntimeError("No frames generated")
    if 'frames' in format and len(images) % format['frames'][0] != format['frames'][1]:
        err_msg = f"The number of frames loaded {len(images)}, does not match the requirements of the currently selected format."
        if len(format['frames']) > 2 and format['frames'][2]:
            raise RuntimeError(err_msg)
        div, mod = format['frames'][:2]
        frames = (len(images) - mod) // div * div + mod
        images = images[:frames]
        #Commenting out log message since it's displayed in UI. consider further
        logger.warn(err_msg + f" Output has been truncated to {len(images)} frames.")
    if 'start_time' in kwargs:
        start_time = kwargs['start_time']
    else:
        start_time = kwargs['skip_first_frames'] * target_frame_time
    target_frame_time *= kwargs.get('select_every_nth', 1)
    #Setup lambda for lazy audio capture
    audio = lazy_get_audio(kwargs['video'], start_time, kwargs['frame_load_cap']*target_frame_time)
    #Adjust target_frame_time for select_every_nth
    video_info = {
        "source_fps": fps,
        "source_frame_count": total_frames,
        "source_duration": duration,
        "source_width": width,
        "source_height": height,
        "loaded_fps": 1/target_frame_time,
        "loaded_frame_count": len(images),
        "loaded_duration": len(images) * target_frame_time,
        "loaded_width": new_width,
        "loaded_height": new_height,
    }
    if vae is None:
        return (images, len(images), audio, video_info)
    else:
        return ({"samples": images}, len(images), audio, video_info)
    
ytdl_path = os.environ.get("VHS_YTDL", None) or shutil.which('yt-dlp') \
        or shutil.which('youtube-dl')
download_history = {}
    

def try_download_video(url):
    if ytdl_path is None:
        return None
    if url in download_history:
        return download_history[url]
    os.makedirs(temp_directory, exist_ok=True)
    #Format information could be added to only download audio for Load Audio,
    #but this gets hairy if same url is also used for video.
    #Best to just always keep defaults
    #dl_format = ['-f', 'ba'] if is_audio else []
    try:
        res = subprocess.run([ytdl_path, "--print", "after_move:filepath",
                              "-P", temp_directory, url],
                             capture_output=True, check=True)
        #strip newline
        file = res.stdout.decode(*ENCODE_ARGS)[:-1]
    except subprocess.CalledProcessError as e:
        raise Exception("An error occurred in the yt-dl process:\n" \
                + e.stderr.decode(*ENCODE_ARGS))
        file = None
    download_history[url] = file
    return file

def validate_path(path, allow_none=False, allow_url=True):
    if path is None:
        return allow_none
    if is_url(path):
        #Probably not feasible to check if url resolves here
        if not allow_url:
            return "URLs are unsupported for this path"
        return is_safe_path(path)
    if not os.path.isfile(strip_path(path)):
        return "Invalid file path: {}".format(path)
    return is_safe_path(path)


class LoadVideoPath:

    def load_video(self, **kwargs):
        if kwargs['video'] is None or validate_path(kwargs['video']) != True:
            raise Exception("video is not a valid path: " + kwargs['video'])
        if is_url(kwargs['video']):
            kwargs['video'] = try_download_video(kwargs['video']) or kwargs['video']
        return load_video(**kwargs)

    @classmethod
    def IS_CHANGED(s, video, **kwargs):
        return hash_path(video)

    @classmethod
    def VALIDATE_INPUTS(s, video):
        return validate_path(video, allow_none=True)