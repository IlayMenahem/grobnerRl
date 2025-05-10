from envs.deepgroebner import BuchbergerEnv

def display_obs(obs):
    ideal, selectables = obs
    print("\nIdeal:")
    for poly in ideal:
        print(f"{poly}")

    print("\nSelectables:")
    for poly in selectables:
        print(f"{poly}")


if __name__ == '__main__':
    env = BuchbergerEnv('2-3-5-uniform', mode='game')
    done = False
    obs, _ = env.reset()

    while not done:
        display_obs(obs)
        action = tuple(map(int, input("Enter your action as comma-separated integers: ").split(',')))
        obs, reward, done, _, _ = env.step(action)
        print(f'reward: {reward}, done: {done}')

    display_obs(obs)
    print("Game over!")
