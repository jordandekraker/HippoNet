import gymnasium as gym
import ale_py
import math
import random, datetime, os, copy, time
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

from PIL import Image, ImageSequence
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using " + str(device))

testonimport=False

# utilities
current_frame = torch.zeros((241,153)).to(device)
possible_actions = 9 # possible actions in all games (inc 0)
action_names = ['toggle down', 'toggle up', 'noop', 'fire', 'up', 'right', 'left', 'down']
games = ['ALE/Tetris-v5', 'ALE/Adventure-v5', 'ALE/AirRaid-v5', 'ALE/Alien-v5', 'ALE/Amidar-v5', 'ALE/Assault-v5', 'ALE/Asterix-v5', 
         'ALE/Asteroids-v5', 'ALE/Atlantis-v5', 'ALE/Atlantis2-v5', 'ALE/Backgammon-v5', 'ALE/BankHeist-v5', 'ALE/BasicMath-v5', 
         'ALE/BattleZone-v5', 'ALE/BeamRider-v5', 'ALE/Berzerk-v5', 'ALE/Blackjack-v5', 'ALE/Bowling-v5', 'ALE/Boxing-v5', 'ALE/Breakout-v5', 
         'ALE/Carnival-v5', 'ALE/Casino-v5', 'ALE/Centipede-v5', 'ALE/ChopperCommand-v5', 'ALE/CrazyClimber-v5', 'ALE/Crossbow-v5', 
         'ALE/Darkchambers-v5', 'ALE/Defender-v5', 'ALE/DemonAttack-v5', 'ALE/DonkeyKong-v5', 'ALE/DoubleDunk-v5', 'ALE/Earthworld-v5', 
         'ALE/ElevatorAction-v5', 'ALE/Enduro-v5', 'ALE/Entombed-v5', 'ALE/Et-v5', 'ALE/FishingDerby-v5', 'ALE/FlagCapture-v5', 
         'ALE/Freeway-v5', 'ALE/Frogger-v5', 'ALE/Frostbite-v5', 'ALE/Galaxian-v5', 'ALE/Gopher-v5', 'ALE/Gravitar-v5', 'ALE/Hangman-v5', 
        ]         # 'ALE/HauntedHouse-v5', 'ALE/Hero-v5', 'ALE/HumanCannonball-v5', 'ALE/IceHockey-v5', 'ALE/Jamesbond-v5', 'ALE/JourneyEscape-v5', 
         # 'ALE/Kaboom-v5', 'ALE/Kangaroo-v5', 'ALE/KeystoneKapers-v5', 'ALE/KingKong-v5', 'ALE/Klax-v5', 'ALE/Koolaid-v5', 'ALE/Krull-v5', 
         # 'ALE/KungFuMaster-v5', 'ALE/LaserGates-v5', 'ALE/LostLuggage-v5', 'ALE/MarioBros-v5', 'ALE/MiniatureGolf-v5', 'ALE/MontezumaRevenge-v5', 
         # 'ALE/MrDo-v5', 'ALE/MsPacman-v5', 'ALE/NameThisGame-v5', 'ALE/Othello-v5', 'ALE/Pacman-v5', 'ALE/Phoenix-v5', 'ALE/Pitfall-v5', 
         # 'ALE/Pitfall2-v5', 'ALE/Pong-v5', 'ALE/Pooyan-v5', 'ALE/PrivateEye-v5', 'ALE/Qbert-v5', 'ALE/Riverraid-v5', 'ALE/RoadRunner-v5', 
         # 'ALE/Robotank-v5', 'ALE/Seaquest-v5', 'ALE/SirLancelot-v5', 'ALE/Skiing-v5', 'ALE/Solaris-v5', 'ALE/SpaceInvaders-v5', 
         # 'ALE/SpaceWar-v5', 'ALE/StarGunner-v5', 'ALE/Superman-v5', 'ALE/Surround-v5', 'ALE/Tennis-v5', 
         # 'ALE/TimePilot-v5', 'ALE/Trondead-v5', 'ALE/Turmoil-v5', 'ALE/Tutankham-v5', 'ALE/UpNDown-v5', 'ALE/Venture-v5', 'ALE/VideoCheckers-v5', 
         # 'ALE/VideoPinball-v5', 'ALE/WizardOfWor-v5', 'ALE/WordZapper-v5', 'ALE/YarsRevenge-v5', 'ALE/Zaxxon-v5']
g=0
env = gym.make(games[g], obs_type='grayscale', 
                        render_mode='rgb_array', 
                        full_action_space=True)
env.reset()


def AtariPress(action):
    global env,g,current_frame
    if action==0:
        g = g-1
    elif action==1:
        g = g+1
    if g>=len(games):
        g=0
    if g<0:
        g=len(games)-1
    if action==0 or action==1:
        env = gym.make(games[g], obs_type='grayscale', 
                        render_mode='rgb_array', 
                        full_action_space=True)
        frame = env.reset()[0]
    else:
        frame = env.step(action-2)[0]
    while np.max(frame)==0: # some games have a blank frame
        frame = env.step(0)[0]

    frame = np.expand_dims(np.expand_dims(frame, 0), 0)
    frame = torch.as_tensor(frame, dtype=torch.float32)
    means = frame.mean()
    stds = frame.std()
    #stds[stds==0] = 1e-4
    frame = (frame - means) / stds
    frame = torchvision.transforms.Resize((241,153))(frame)
    current_frame = frame.to(device)

    return current_frame

# utilities
def live_plot(imgs, file_name=None, duration=200, loop=0):
    """
    Display a list of images in a single row. If a file_name is provided, 
    append the plot to an existing GIF or create a new GIF.

    Args:
        imgs (list): List of image tensors or arrays to display.
        file_name (str, optional): Path to the .gif file. If None, no saving is performed.
        duration (int): Duration of each frame in the GIF in milliseconds.
        loop (int): Number of times the GIF will loop. 0 means infinite.
    """
    clear_output(wait=True)
    fig, axes = plt.subplots(1, len(imgs), figsize=(len(imgs) * 4, 4))
    
    if len(imgs) == 1:  # Handle single image case
        axes = [axes]
    
    # Plot each image
    for ax, img in zip(axes, imgs):
        ax.axis('off')
        if isinstance(img, torch.Tensor) and img.dtype == torch.float32:
            ax.imshow(torch.squeeze(img).cpu().detach(), cmap='gray')
        else:
            ax.imshow(img)
    
    plt.tight_layout()
    plt.show()

    # Save to GIF if file_name is provided
    if file_name:
        # Save the current plot as an image
        fig.canvas.draw()
        image = Image.frombytes(
            'RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        plt.close(fig)  # Close the figure to avoid overlapping plots

        if os.path.exists(file_name):
            # Append to existing GIF
            with Image.open(file_name) as gif:
                frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
                frames.append(image)
                frames[0].save(
                    file_name, save_all=True, append_images=frames[1:], duration=duration, loop=loop
                )
        else:
            # Create a new GIF
            image.save(file_name, save_all=True, append_images=[image], duration=duration, loop=loop)
            
def frameproc(frame):
    global current_frame
    # trim, resize to suit our convolutions, and standardize
    # frame = frame[25:225,:]
    frame = np.expand_dims(np.expand_dims(frame, 0), 0)
    frame = torch.as_tensor(frame, dtype=torch.float32)
    means = frame.mean()
    stds = frame.std()
    frame = (frame - means) / (stds+1e-06)
    frame = torchvision.transforms.Resize((241,153))(frame)
    current_frame = frame.to(device)
    return

frame = AtariPress(0)
#live_plot(current_frame)

# test
def AtariTest():
    for gg in range(len(games)+5):
        for a in range(possible_actions):
            frame = AtariPress(a+2)
            frameproc(frame)
            live_plot([current_frame])
        AtariPress(1)
if testonimport:
    AtariTest()