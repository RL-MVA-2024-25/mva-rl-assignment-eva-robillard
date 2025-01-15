from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import os
from tqdm import tqdm
from xgboost import XGBRegressor
import zstandard as zstd
import pickle
from evaluate import evaluate_HIV, evaluate_HIV_population
import torch
import random

env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)  
# The time wrapper limits the number of steps in an episode at 200.

# Configuration parameters for the agent
config = {
    "horizon": 600000,          # Total number of interactions to collect samples
    "max_episode": 150,        # Maximum number of iterations for FQI
    "gamma": 0.97,             # Discount factor for future rewards
    "n_estimators": 650,       # Number of trees in the XGBRegressor
    "max_depth": 15            # Maximum depth of each tree in the XGBRegressor
}

class ProjectAgent:
    def __init__(self, config=config, env=env):
        """
        Initialize the agent with the provided configuration and environment.
        
        Args:
            config (dict): Configuration parameters for the agent.
            env (gym.Env): The environment in which the agent will operate.
        """
        self.horizon = config["horizon"]
        self.max_episode = config["max_episode"]
        self.gamma = config["gamma"]
        self.n_estimators = config["n_estimators"]
        self.max_depth = config["max_depth"]

        self.env = env
        self.model = None  # Placeholder for the model

    def collect_samples(self, horizon, disable_tqdm=False, print_done_states=False):
        """
        Collect samples from the environment for training.
        
        Args:
            horizon (int): Number of samples to collect.
            disable_tqdm (bool): If True, disables the progress bar.
            print_done_states (bool): If True, prints a message when an episode ends.
        
        Returns:
            tuple: Arrays of states (S), actions (A), rewards (R), next states (S2), and done flags (D).
        """
        s, _ = self.env.reset()
        S, A, R, S2, D = [], [], [], [], []

        for _ in tqdm(range(horizon), disable=disable_tqdm):
            a = self.env.action_space.sample()
            s2, r, done, trunc, _ = self.env.step(a)
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            D.append(done)

            if done or trunc:
                s, _ = self.env.reset()
                if done and print_done_states:
                    print("Episode done!")
            else:
                s = s2

        return np.array(S), np.array(A).reshape((-1, 1)), np.array(R), np.array(S2), np.array(D)

    def rf_fqi(self, S, A, R, S2, D, iterations, nb_actions, gamma, disable_tqdm=False):
        """
        Perform Fitted Q Iteration (FQI) using a Random Forest regressor (XGBRegressor).
        
        Args:
            S (ndarray): States.
            A (ndarray): Actions.
            R (ndarray): Rewards.
            S2 (ndarray): Next states.
            D (ndarray): Done flags.
            iterations (int): Number of FQI iterations.
            nb_actions (int): Number of possible actions in the environment.
            gamma (float): Discount factor.
            disable_tqdm (bool): If True, disables the progress bar.
        
        Returns:
            list: Trained Q functions.
        """
        nb_samples = S.shape[0]
        Qfunctions = []

        # Concatenate states and actions for input
        SA = np.append(S, A, axis=1)

        for iter in tqdm(range(iterations), disable=disable_tqdm):
            if iter == 0:
                value = R.copy()
            else:
                # Compute Q-values for all actions
                Q2 = np.zeros((nb_samples, nb_actions))
                for a2 in range(nb_actions):
                    A2 = a2 * np.ones((S.shape[0], 1))
                    S2A2 = np.append(S2, A2, axis=1)
                    Q2[:, a2] = Qfunctions[-1].predict(S2A2)
                max_Q2 = np.max(Q2, axis=1)
                value = R + gamma * (1 - D) * max_Q2

            # Train a new Q-function
            Q = XGBRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth)
            Q.fit(SA, value)
            self.model = Q
            Qfunctions.append(Q)

            # Evaluate the model periodically
            if iter % 5 == 0:
                print(f"Iteration {iter}: Evaluating the model...")
                indiv_score = evaluate_HIV(agent=self, nb_episode=5)
                pop_score = evaluate_HIV_population(agent=self, nb_episode=20)
                print(f"Scores: Individual = {indiv_score}, Population = {pop_score}")

        return Qfunctions

    def train(self, disable_tqdm=False, print_done_states=False):
        """
        Train the agent using Fitted Q Iteration (FQI).
        
        Args:
            disable_tqdm (bool): If True, disables the progress bar.
            print_done_states (bool): If True, prints a message when an episode ends.
        
        Returns:
            The trained model.
        """
        S, A, R, S2, D = self.collect_samples(self.horizon, disable_tqdm, print_done_states)
        nb_actions = self.env.action_space.n
        self.Qfunctions = self.rf_fqi(S, A, R, S2, D, self.max_episode, nb_actions, self.gamma, disable_tqdm)
        self.model = self.Qfunctions[-1]
        return self.model

    def greedy_action(self, Q, s, nb_actions):
        """
        Select the action with the highest Q-value for a given state.
        
        Args:
            Q: The Q-function.
            s: The current state.
            nb_actions: Number of possible actions.
        
        Returns:
            int: The selected action.
        """
        Qsa = [Q.predict(np.append(s, a).reshape(1, -1)) for a in range(nb_actions)]
        return np.argmax(Qsa)

    def act(self, observation, use_random=False):
        """
        Choose an action for a given observation.
        
        Args:
            observation: The current state.
            use_random (bool): If True, selects a random action.
        
        Returns:
            int: The selected action.
        """
        if use_random:
            return self.env.action_space.sample()
        else:
            return self.greedy_action(self.model, observation, self.env.action_space.n)

    def save(self, path="./models/model.pkl.zst"):
        """
        Save the model in a compressed format.
        
        Args:
            path (str): Path to save the model.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temp_model_path = "./temporary_model.pkl"
        with open(temp_model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(temp_model_path, "rb") as f_in, open(path, "wb") as f_out:
            compressor = zstd.ZstdCompressor(level=19)
            f_out.write(compressor.compress(f_in.read()))
            print("Model saved in a compressed format.")
        os.remove(temp_model_path)

    def load(self, path="./model.pkl.zst"):
        """
        Load the model from a compressed format.
        
        Args:
            path (str): Path to the saved model.
        """
        if not os.path.exists(path):
            print(f"No model found at {path}.")
            self.model = None
            return

        temp_model_path = "./model_temp.pkl"
        try:
            with open(path, "rb") as f_in, open(temp_model_path, "wb") as f_out:
                decompressor = zstd.ZstdDecompressor()
                f_out.write(decompressor.decompress(f_in.read()))

            with open(temp_model_path, "rb") as f:
                self.model = pickle.load(f)

            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error during model loading: {e}")
        finally:
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)

# Set random seeds for reproducibility
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
