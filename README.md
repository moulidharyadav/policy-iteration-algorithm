# POLICY ITERATION ALGORITHM

## AIM
The goal of the notebook is to implement and evaluate a policy iteration algorithm within a custom environment (gym-walk) to find the optimal policy that maximizes the agent's performance in terms of reaching a goal state with the highest probability and reward.

## PROBLEM STATEMENT
The task is to develop and apply a policy iteration algorithm to solve a grid-based environment (gym-walk). The environment consists of states the agent must navigate through to reach a goal. The agent has to learn the best sequence of actions (policy) that maximizes its chances of reaching the goal state while obtaining the highest cumulative reward.

## POLICY ITERATION ALGORITHM
Initialize: Start with a random policy for each state and initialize the value function arbitrarily.

Policy Evaluation: For each state, evaluate the current policy by computing the expected value function under the current policy.

Policy Improvement: Improve the policy by making it greedy with respect to the current value function (i.e., choose the action that maximizes the value function for each state).

Check Convergence: Repeat the evaluation and improvement steps until the policy stabilizes (i.e., when no further changes to the policy occur).

Optimal Policy: Once convergence is achieved, the policy is considered optimal, providing the best actions for the agent in each state.

## POLICY IMPROVEMENT FUNCTION
### Name:MOULIDHAR
### Register Number: 212223240042
```python
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
      for a in range(len(P[s])):
        for prob, next_state, reward, done in P[s][a]:
          Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
    new_pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return new_pi

pi_2 = policy_improvement(V1, P)
print("Name: Prasannalakshmi G")
print("Register Number: 212222240075")
print_policy(pi_2, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)

```
## POLICY ITERATION FUNCTION
### Name: DHEENA DARSHINI KARTHIK DHEEPAN
### Register Number: 212223240030
```python
def policy_iteration(P, gamma=1.0, theta=1e-10):
  random_actions = np.random.choice(tuple(P[0].keys()), len(P))
  pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]

  while True:
    old_pi = {s: pi(s) for s in range(len(P))}
    V = policy_evaluation(pi, P, gamma, theta)
    pi = policy_improvement(V, P, gamma)

    if old_pi == {s: pi(s) for s in range(len(P))}:
      break

  return V, pi
optimal_V, optimal_pi = policy_iteration(P)
print("Name: Prasannalakshmi G")
print("Register Number: 212222240075")
print('Optimal policy and state-value function (PI):')
print_policy(optimal_pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)

```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy
#### POLICY:
<img width="1024" height="196" alt="image" src="https://github.com/user-attachments/assets/aaf18268-2e8b-4561-8ed4-b5ba058515a0" />


#### STATE VALUE FUNCTION:
<img width="759" height="212" alt="image" src="https://github.com/user-attachments/assets/dc422905-5cca-4b94-a8d9-af0726b306a0" />


#### SUCCESS:
<img width="1220" height="73" alt="Screenshot 2025-09-27 082643" src="https://github.com/user-attachments/assets/ecaf1005-502a-4496-9703-04e5513c4161" />


### 2. Policy, Value function and success rate for the Improved Policy
#### POLICY:
<img width="750" height="201" alt="image" src="https://github.com/user-attachments/assets/6afbe09c-ef8d-4ab3-8784-88106e7a53b8" />

#### STATE VALUE FUNCTION:
<img width="901" height="190" alt="image" src="https://github.com/user-attachments/assets/f0a6a647-6057-4568-982f-b69345e42c58" />


#### SUCCESS:
<img width="644" height="72" alt="image" src="https://github.com/user-attachments/assets/4f72a228-b806-40aa-9b33-a897dae74c80" />
<img width="909" height="82" alt="image" src="https://github.com/user-attachments/assets/ecd6148c-172d-4756-9f1f-2f5060ea097d" />



### 3. Policy, Value function and success rate after policy iteration
#### POLICY:
<img width="764" height="192" alt="image" src="https://github.com/user-attachments/assets/549b54b1-6995-4e29-9c50-9b96ca9bde8a" />


#### STATE VALUE FUNCTION:
<img width="1193" height="172" alt="image" src="https://github.com/user-attachments/assets/6fb45cc0-da25-4806-86da-01a373ff132d" />


#### SUCCESS:
<img width="772" height="37" alt="image" src="https://github.com/user-attachments/assets/fcb37c4f-76d8-415e-9dea-17937677744b" />




## RESULT:
Thus the program to iterate the policy evaluation and policy improvement is executed successfully.
