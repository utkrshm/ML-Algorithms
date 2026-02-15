"""Run Monte Carlo Simulations on a deck of cards to determine the probability of specific events, implemented along with multithreading for efficiency"""
from argparse import ArgumentParser
from copy import deepcopy

import numpy as np
from joblib import Parallel, delayed

deck = [rank + suit for rank in ['02', '03', '04', '05', '06', '07', '08', '09', '10', ' J', ' Q', ' K', ' A'] for suit in ['S', 'H', 'C', 'D']]

def _check_conditions(deck):
    has_kk = False
    has_kq = False
    
    for i in range(len(deck) - 1):
        card1 = deck[i]
        card2 = deck[i+1]
        
        is_k1 = ' K' in card1
        is_k2 = ' K' in card2
        is_q1 = ' Q' in card1
        is_q2 = ' Q' in card2
        
        if is_k1 and is_k2:
            has_kk = True
        
        if (is_k1 and is_q2) or (is_q1 and is_k2):
            has_kq = True
            
        if has_kk and has_kq:
            break
            
    return (has_kk, has_kq)

def single_simulation(deck):
    current_deck = deepcopy(deck)
    np.random.shuffle(current_deck)
    return _check_conditions(current_deck)

def monte_carlo(n_simulations):
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(single_simulation)(deck)
        for _ in range(n_simulations)
    )
    results = np.array(results)
    
    kk_prob = np.mean(results[:, 0])
    kq_prob = np.mean(results[:, 1])
    
    print(f"Probability that two kings appear together: {kk_prob:.5f}")
    print(f"Probability that a king and a queen appear together: {kq_prob:.5f}")

    print("True Probability of two kings appearing together: ~0.217")
    print("True Probability of a king and a queen appearing together: ~0.486")

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="MCS_Cards_Probs",
        description="Run Monte Carlo Simulations on a deck of cards to ascertain some probabilities"
    )
    parser.add_argument("-n", "--n-sims", type=int, default=10000, help="Number of simulations to run. Default: 10000")
    args = parser.parse_args()
    
    monte_carlo(args.n_sims)