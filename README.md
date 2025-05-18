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

Voice Assistant Flow:

1. use wake word: hello aeris 
-> please provide identification (1 time check) 
  -> if success -> ask user to give command -> takes in 1 command and tries to perform it 
  -> if fails, it will try to ask for identification again next time wake word is used.

2. normal conversation/presentation 

3. use wake word: hello aeris
-> (if previously identified): ask user to give command -> takes in 1 command
-> (if previously failed identification): repeat identification process

expected command input samples:
"Can you play the GCA video on screen one?",
"Turn on the experience wall, please.",
"I'd like to stop the video that's playing on screen four.",
"Hey, pause whatever is running on screen two.",
"Can you start the pi tech video on screen three?",
"Shut off the experience wall now.",
"Play the aero video on the second screen.",
"I'd like to see the GAT video on screen 4.",
"Can you play the GAT video on screen 3?",
"Could you queue up the pi tech one on screen number two?",
"Pause the video on screen three, please.",
"Start playing the GAT video on screen 1.",
"Let’s watch cw aero on screen four.",
"Turn off the wall display thing.",
"Can you raise the volume to like 80%?",
"Stop whatever’s on screen two.",
"Play the GCA clip on the third screen.",

-------

screens == panel
valid screens: 1, 2, 3, 4

-------

Here's the command to that the bot expects:

class CommandControl(dspy.Signature):
    """Determine the command details based on the input request and intention.

    Input:
    - request: actual request from the user
    Output:
    # IMPORTANT:
    # Here are the valid responses:
    - play_video(video_name: str, screen_number :int)
    - pause_video(screen_number: int)
    - stop_video(screen_number: int)
    - experience_wall_control(action: str)  # valid actions: on/off
    - set_volume(volume_level: float)  # 0.0 to 1.0 value for volume_level
    - system_sleep()

    [PRIORITY] Try your best to match file names and screen numbers.
    [FALLBACK] If there's no matching file names, try using similar sounding phonetics to match the valid video files names below.

    # Video Controls:
    ## Valid video files names:
    gca_video
    cw_aero_video
    pi_tech_video
    gat_video

    # Valid screen number are (1,2,3,4). However, please map it to (0, 1, 2, 3)

    ## Output format:
    The output should only contain the function name and positional args.

