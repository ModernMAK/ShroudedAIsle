import ffmpy


def build_params(framerate, title, codec, output):
    print(output)
    params = '-f gdigrab -framerate {0} -i title="{1}" -vcodec {2} -crf 0 -preset ultrafast "{3}"'
    print(params)
    params = params.format(framerate, title, codec, output)
    print(params)
    return params


def get_ffmpeg(framerate, title, codec, output):
    ffmpeg = ffmpy.FFmpeg(global_options=build_params(framerate, title, codec, output))
    return ffmpeg


def run_ffmpeg(ffmpeg):
    ffmpeg.run()


def close_ffmpeg(ffmpeg):
    ffmpeg.process.terminate()
