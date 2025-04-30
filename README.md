# AERIS

0. Setup: Update the config.ini file
- Make sure to update the:
    * target screen name
    * presentation video paths
    * The mic_index (0) is used by default -- window's default mic
    * The rest of the configuration file can be left as it is.

1. To run: double click the aeris.bat file
2. Edge browser should launch, if it does not, open any browser and head to '127.0.0.1:5000' to see the dashboard.

------

## Development logs

TODO:
- [ ] volume control
- [ ] fix faster whisper, currently fallback to google
- [ ] sometimes aeris is triggered twice


TEST:
1. pip install whisper-live --> has issue with installing ffmpeg, av


ERROR list:
Failed to execute command: 'Name' object has no attribute 'value', Received: command_str='pause_video(all_videos, all_screens)'
Traceback (most recent call last):
  File "C:\Users\gca20\Documents\voice_ai\app.py", line 449, in execute_command
    args.append(a.value)
                ^^^^^^^
AttributeError: 'Name' object has no attribute 'value'
