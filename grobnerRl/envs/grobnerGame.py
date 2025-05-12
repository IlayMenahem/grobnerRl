from grobnerRl.envs.deepgroebner import BuchbergerEnv

def display_obs(obs):
    ideal, selectables = obs
    print("\nIdeal:")
    for poly in ideal:
        print(f"{poly}")

    print("\nSelectables:")
    for poly in selectables:
        print(f"{poly}")


def play_buchberger_game(ideal_params):
    env = BuchbergerEnv(ideal_params, mode='game')
    done = False
    obs, _ = env.reset()

    while not done:
        display_obs(obs)
        action = tuple(map(int, input("Enter your action as space-separated integers: ").split(' ')))
        obs, reward, done, _, _ = env.step(action)
        print(f'reward: {reward}, done: {done}')

    display_obs(obs)
    print("Game over!")
