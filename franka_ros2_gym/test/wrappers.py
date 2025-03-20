import gymnasium as gym
from gymnasium.spaces import Box, Dict
import numpy as np
from collections import deque
import cv2
import imageio
import os
from gymnasium.spaces import flatten_space, flatten

class PixelPairStack(gym.Wrapper):
    def __init__(self, env, image1_key='image1', image2_key='image2', stack_key='pixels'):
        super().__init__(env)
        self.image1_key = image1_key
        self.image2_key = image2_key
        self.stack_key = stack_key

        # Get the original image shape from the observation space.
        image_shape = env.observation_space[image1_key].shape
        # Remove a batch dimension if it exists.
        if len(image_shape) == 4:
            image_shape = image_shape[1:]
        # New observation will have shape (2, *original image shape*)
        stacked_shape = (2,) + image_shape
        self.observation_space[self.stack_key] = gym.spaces.Box(
            low=0, high=255, shape=stacked_shape, dtype=np.uint8
        )

    def _extract_image(self, obs, key):
        image = obs[key]
        # Remove batch dimension if present.
        if len(image.shape) == 4:
            image = image[0]
        return image.copy()  # Leave the image in its original format.

    def _transform_observation(self, obs):
        # Extract both images and stack them along a new axis (axis=0)
        img1 = self._extract_image(obs, self.image1_key)
        img2 = self._extract_image(obs, self.image2_key)
        obs[self.stack_key] = np.stack([img1, img2], axis=0)
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._transform_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._transform_observation(obs), reward, terminated, truncated, info

class PixelFrameStack(gym.Wrapper):
    def __init__(self, env, num_frames, stack_key='pixels'):
        super().__init__(env)
        self._num_frames = num_frames
        self.stack_key = stack_key
        self._frames = deque([], maxlen=num_frames)
        pixels_shape = env.observation_space[stack_key].shape
        

        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self.observation_space[stack_key] = Box(low=0, high=255, shape=(num_frames*pixels_shape[-1], *pixels_shape[:-1]), dtype=np.uint8)

    def _transform_observation(self, obs):
        assert len(self._frames) == self._num_frames
        obs[self.stack_key] = np.concatenate(list(self._frames), axis=0)
        return obs

    def _extract_pixels(self, obs):
        pixels = obs[self.stack_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        obs, info = self.env.reset()
        pixels = self._extract_pixels(obs)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pixels = self._extract_pixels(obs)
        self._frames.append(pixels)
        return self._transform_observation(obs), reward, terminated, truncated, info
    
class StateFrameStack(gym.Wrapper):
    def __init__(self, env, num_frames, stack_key='state', flatten=True):
        super().__init__(env)
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)   
        self.stack_key = stack_key
        self.flatten = flatten

        shape = self.env.observation_space[stack_key].shape
        if isinstance(shape, int):
            shape = (shape,)  # Convert to a tuple for consistency
        else:
            shape = shape  # If it's already a tuple or list, keep it as is
        if flatten: 
            self.observation_space[stack_key] = Box(low=-np.inf, high=np.inf, shape=(num_frames * shape[-1],), dtype=np.float32)
        else:
            self.observation_space[stack_key] = Box(low=-np.inf, high=np.inf, shape=(num_frames, *shape), dtype=np.float32)

    def _transform_observation(self):
        assert len(self._frames) == self._num_frames
        obs = np.array(self._frames)
        if self.flatten:
            obs = obs.flatten()
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self._num_frames):
            self._frames.append(obs[self.stack_key])
        obs[self.stack_key] = self._transform_observation()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs[self.stack_key])
        obs[self.stack_key] = self._transform_observation()
        return obs, reward, terminated, truncated, info

class CustomPixelObservation(gym.ObservationWrapper):
    """Resize (and optionally crop) the pixel observation to a given resolution, handling any number of channels."""
    def __init__(self, env, pixel_key='pixels', crop_resolution=None, resize_resolution=None):
        super().__init__(env)
        
        # Allow integer input to be converted into (H, W) tuples.
        if isinstance(resize_resolution, int):
            resize_resolution = (resize_resolution, resize_resolution)
        if isinstance(crop_resolution, int):
            crop_resolution = (crop_resolution, crop_resolution)
            
        self.pixel_key = pixel_key
        self.crop_resolution = crop_resolution
        self.resize_resolution = resize_resolution
        
        # Retrieve the original number of channels from the environment's observation space.
        orig_shape = self.observation_space[pixel_key].shape  # e.g., (H_orig, W_orig, C)
        n_channels = orig_shape[-1]
        
        # Determine new shape: if resize_resolution is provided, use that; otherwise, keep the original spatial size.
        if self.resize_resolution is not None:
            new_shape = (*self.resize_resolution, n_channels)
        else:
            new_shape = orig_shape
        
        # Update the observation space with the new shape.
        self.observation_space[pixel_key] = Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def observation(self, observation):
        pixels = observation[self.pixel_key]
        
        # Crop the image if a crop resolution is provided.
        if self.crop_resolution is not None:
            # Only crop if the current spatial dimensions don't match the desired crop resolution.
            if pixels.shape[:2] != self.crop_resolution:
                orig_h, orig_w = pixels.shape[:2]
                crop_h, crop_w = self.crop_resolution
                # Compute top-left coordinates for center crop.
                y = int((orig_h - crop_h) / 2)
                x = int((orig_w - crop_w) / 2)
                pixels = pixels[y:y+crop_h, x:x+crop_w]
        
        # Resize the image if a resize resolution is provided.
        if self.resize_resolution is not None:
            # Only resize if the spatial dimensions don't already match.
            if pixels.shape[:2] != self.resize_resolution:
                pixels = cv2.resize(
                    pixels,
                    dsize=self.resize_resolution,
                    interpolation=cv2.INTER_CUBIC,
                )
        
        observation[self.pixel_key] = pixels
        return observation

class VideoRecorder(gym.Wrapper):
  """Wrapper for rendering and saving rollouts to disk.
  Reference: https://github.com/ikostrikov/jaxrl/
  """

  def __init__(
      self,
      env,
      save_dir,
      crop_resolution,
      resize_resolution,
      fps = 10,
      current_episode=0,
      record_every=2,
  ):
    super().__init__(env)

    self.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)
    num_vids = len(os.listdir(save_dir))
    current_episode = num_vids*record_every

    if isinstance(resize_resolution, int):
      self.resize_resolution = (resize_resolution, resize_resolution)
    if isinstance(crop_resolution, int):
      self.crop_resolution = (crop_resolution, crop_resolution)

    self.resize_h, self.resize_w = self.resize_resolution
    self.crop_h, self.crop_w = self.crop_resolution
    self.fps = fps
    self.enabled = True
    self.current_episode = current_episode
    self.record_every = record_every
    self.frames = []

  def step(self, action):
    observation, reward, terminated, truncated, info = self.env.step(action)
    if self.current_episode % self.record_every == 0:
      frame = observation['pixels']
      if self.crop_resolution is not None:
        # Crop
        if frame.shape[:2] != (self.crop_h, self.crop_w):
          center = frame.shape
          x = center[1]/2 - self.crop_w/2
          y = center[0]/2 - self.crop_h/2
          frame = frame[int(y):int(y+self.crop_h), int(x):int(x+self.crop_w)]
      if self.resize_resolution is not None:
        if frame.shape[:2] != (self.resize_h, self.resize_w):
          frame = cv2.resize(
              frame,
              dsize=(self.resize_h, self.resize_w),
              interpolation=cv2.INTER_CUBIC,
          )
      # Write rewards on the frame
      cv2.putText(
          frame,
          f"{reward:.3f}",
          (10, 40),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (0, 255, 0),
          1,
          cv2.LINE_AA,
      )
      # Save
      self.frames.append(frame)
    if terminated or truncated:
      if self.current_episode % self.record_every == 0:
        if info['success'] != 0:
          filename = os.path.join(self.save_dir, f"{self.current_episode}_success.mp4")
        else:
          filename = os.path.join(self.save_dir, f"{self.current_episode}.mp4")
        imageio.mimsave(filename, self.frames, fps=self.fps)
        self.frames = []
      self.current_episode += 1
    return observation, reward, terminated, truncated, info
  
class VideoRecorderReal(gym.Wrapper):
    """Wrapper for rendering and saving rollouts to disk from a specific camera."""

    def __init__(
        self,
        env,
        save_dir,
        crop_resolution,
        resize_resolution,
        camera_name="wrist2",
        fps=10,
        current_episode=0,
        record_every=2,
    ):
        super().__init__(env)

        self.save_dir = save_dir
        self.camera_name = camera_name
        os.makedirs(save_dir, exist_ok=True)
        num_vids = len([f for f in os.listdir(save_dir) if f.endswith(f"{camera_name}.mp4")])
        print(f"num_vids: {num_vids}")
        current_episode = num_vids * record_every

        if isinstance(resize_resolution, int):
            self.resize_resolution = (resize_resolution, resize_resolution)
        if isinstance(crop_resolution, int):
            self.crop_resolution = (crop_resolution, crop_resolution)

        self.resize_h, self.resize_w = self.resize_resolution
        self.crop_h, self.crop_w = self.crop_resolution
        self.fps = fps
        self.enabled = True
        self.current_episode = current_episode
        self.record_every = record_every
        self.frames = []

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        if self.current_episode % self.record_every == 0:
            frame = observation[self.camera_name].copy()
            
            if self.crop_resolution is not None:
                if frame.shape[:2] != (self.crop_h, self.crop_w):
                    center = frame.shape
                    x = center[1] // 2 - self.crop_w // 2
                    y = center[0] // 2 - self.crop_h // 2
                    frame = frame[int(y):int(y + self.crop_h), int(x):int(x + self.crop_w)]

            if self.resize_resolution is not None:
                if frame.shape[:2] != (self.resize_h, self.resize_w):
                    frame = cv2.resize(
                        frame,
                        dsize=(self.resize_w, self.resize_h),
                        interpolation=cv2.INTER_CUBIC,
                    )
            frame  = np.ascontiguousarray(frame)
            # cv2.putText(
            #     frame,
            #     f"{reward:.3f}",
            #     (10, 40),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     (0, 255, 0),
            #     1,
            #     cv2.LINE_AA,
            # )

            self.frames.append(frame)

        if terminated or truncated:
            if self.current_episode % self.record_every == 0:
                filename = os.path.join(self.save_dir, f"{self.current_episode}_{self.camera_name}.mp4")
                imageio.mimsave(filename, self.frames, fps=self.fps)
                self.frames = []

            self.current_episode += 1

        return observation, reward, terminated, truncated, info
  
class ActionRepeat(gym.Wrapper):
  """Repeat the agent's action N times in the environment.
  Reference: https://github.com/ikostrikov/jaxrl/
  """

  def __init__(self, env, repeat):
    """Constructor.
    Args:
      env: A gym env.
      repeat: The number of times to repeat the action per single underlying env
        step.
    """
    super().__init__(env)

    assert repeat > 1, "repeat should be greater than 1."
    self._repeat = repeat

  def step(self, action):
    total_reward = 0.0
    for _ in range(self._repeat):
      observation, reward, terminated, truncated, info = self.env.step(action)
      total_reward += reward
      if terminated or truncated:
        break
    return observation, total_reward, terminated, truncated, info
  
class FrankaObservation(gym.ObservationWrapper):
  """Resize the observation to a given resolution"""
  def __init__(self, env, camera_name='front'):
    super().__init__(env)
    self.camera_name = camera_name
    pixel_space = self.observation_space['images'][camera_name]
    self.state_keys = ['tcp_pose', 'gripper_pos']
    state_dim = 0
    for key in self.state_keys:
      state_dim += self.observation_space['state'][key].shape[0]
    state_space = Box(-np.inf, np.inf, shape=(state_dim,), dtype=np.float32)
    self.observation_space = Dict({'pixels': pixel_space, 'state': state_space})
    
  def observation(self, observation):
    pixels = observation['images'][self.camera_name]
    state = np.concatenate([observation['state'][key] for key in self.state_keys])
    observation = {}
    observation['pixels'] = pixels
    observation['state'] = state
    return observation    
  

class DualCamObservation(gym.ObservationWrapper):
    """Stack two camera images along the channel dimension and leave the state unchanged."""
    def __init__(self, env, camera_name1='wrist1', camera_name2='wrist2'):
        super().__init__(env)
        self.camera_name1 = camera_name1
        self.camera_name2 = camera_name2

        # Get the pixel spaces from both cameras.
        pixel_space1 = self.observation_space[camera_name1]
        pixel_space2 = self.observation_space[camera_name2]

        # Extract shapes from each camera.
        h, w, c1 = pixel_space1.shape
        h2, w2, c2 = pixel_space2.shape
        assert h == h2 and w == w2, "Both cameras must have the same height and width."
        new_shape = (h, w, c1 + c2)

        # Concatenate the low and high bounds along the channel dimension.
        low = np.concatenate([pixel_space1.low, pixel_space2.low], axis=-1)
        high = np.concatenate([pixel_space1.high, pixel_space2.high], axis=-1)

        # Explicitly pass the new shape to the Box.
        stacked_space = Box(low=low, high=high, shape=new_shape, dtype=pixel_space1.dtype)

        # Leave the state observation space unchanged.
        state_space = self.observation_space['state']

        # Define the new observation space.
        self.observation_space = Dict({'pixels': stacked_space, 'state': state_space})

    def observation(self, observation):
        # Get images from both cameras.
        pixels1 = observation[self.camera_name1]
        pixels2 = observation[self.camera_name2]

        # Stack images along the channel dimension.
        pixels = np.concatenate([pixels1, pixels2], axis=-1)
        
        # Pass state through without modification.
        state = observation['state']
        
        return {'pixels': pixels, 'state': state}
  

  
# class FrankaDualCamObservation(gym.ObservationWrapper):
#   """Resize the observation to a given resolution"""
#   def __init__(self, env, camera1_name='wrist1', camera2_name='wrist2'):
#     super().__init__(env)
#     self.camera1_name = camera1_name
#     self.camera2_name = camera2_name
#     img1_space = self.observation_space['images'][camera1_name]
#     img2_space = self.observation_space['images'][camera2_name]
#     self.state_keys = ['panda/tcp_pos', 'panda/tcp_orientation', 'panda/gripper_pos', 'panda/gripper_vec']
#     state_dim = 0
#     for key in self.state_keys:
#       state_dim += self.observation_space['state'][key].shape[0]
#     state_space = Box(-np.inf, np.inf, shape=(state_dim,), dtype=np.float32)
#     self.observation_space = Dict({'img1': img1_space, 'img2': img2_space, 'state': state_space})
    
#   def observation(self, observation):
#     img1 = observation['images'][self.camera1_name]
#     img2 = observation['images'][self.camera2_name]
#     state = np.concatenate([observation['state'][key] for key in self.state_keys])
#     observation = {}
#     observation['img1'] = img1
#     observation['img2'] = img2
#     observation['state'] = state
#     return observation  
  
class ActionState(gym.Wrapper):
    # Add previous action to the state
    def __init__(self, env, state_key='state', action_key='action'):
        super().__init__(env)
        self.action_key = action_key
        self.state_key = state_key
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space[state_key].shape[0]
        self.observation_space[state_key] = Box(low=-np.inf, high=np.inf, shape=(self.state_dim + self.action_dim,), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.action = np.zeros(self.action_dim)
        obs[self.state_key] = np.concatenate([obs[self.state_key], self.action])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs[self.state_key] = np.concatenate([obs[self.state_key], action])
        return obs, reward, terminated, truncated, info
    
class RotateImage(gym.ObservationWrapper):
    """Rotate the pixel observation by 180 degrees."""

    def __init__(self, env, pixel_key='pixels'):
        super().__init__(env)
        self.pixel_key = pixel_key

        # Optionally, update the observation space if needed.
        # Since a 180° rotation doesn't change the image shape,
        # we can just copy the existing space.
        self.observation_space = env.observation_space

    def observation(self, observation):
        # Extract the image from the observation using the specified key.
        image = observation[self.pixel_key]
        
         # Check if the image has a leading batch dimension.
        if image.shape[0] == 1:
            # Remove the batch dimension: shape becomes (height, width, 3)
            image = image[0]
            # Rotate the image by 180 degrees using OpenCV.
            rotated_image = cv2.rotate(image, cv2.ROTATE_180)
            # Re-add the batch dimension: shape becomes (1, height, width, 3)
            rotated_image = np.expand_dims(rotated_image, axis=0)
        else:
            # Otherwise, just rotate the image normally.
            rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        
        
        # Replace the image in the observation with the rotated version.
        observation[self.pixel_key] = rotated_image
        return observation
    
class SERLObsWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treat the observation space as a dictionary
    of a flattened state space and the images.
    """

    def __init__(self, env, proprio_keys=None):
        super().__init__(env)
        self.proprio_keys = proprio_keys
        if self.proprio_keys is None:
            self.proprio_keys = list(self.env.observation_space["state"].keys())

        self.proprio_space = gym.spaces.Dict(
            {key: self.env.observation_space["state"][key] for key in self.proprio_keys}
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": flatten_space(self.proprio_space),
                **(self.env.observation_space["images"]),
            }
        )

    def observation(self, obs):
        obs = {
            "state": flatten(
                self.proprio_space,
                {key: obs["state"][key] for key in self.proprio_keys},
            ),
            **(obs["images"]),
        }
        return obs

    def reset(self, **kwargs):
        obs, info =  self.env.reset(**kwargs)
        return self.observation(obs), info

def flatten_observations(obs, proprio_space, proprio_keys):
        obs = {
            "state": flatten(
                proprio_space,
                {key: obs["state"][key] for key in proprio_keys},
            ),
            **(obs["images"]),
        }
        return obs