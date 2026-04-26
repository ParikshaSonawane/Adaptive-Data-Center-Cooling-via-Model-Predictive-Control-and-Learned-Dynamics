# ❄️ Data Center Cooling using Reinforcement Learning & Model Predictive Control

An AI-driven thermal management system for data centers that uses **Reinforcement Learning concepts**, **Model Predictive Control (MPC)**, and **learned system dynamics** to optimize cooling efficiency while preventing overheating.

---

## 📌 Overview

Data centers generate massive heat due to continuously running servers. Traditional cooling systems often waste energy by applying uniform cooling without predicting future thermal conditions.

This project simulates a **data center environment** where an intelligent controller learns and predicts temperature changes, then applies optimized cooling actions.

### Objectives

- Maintain server temperatures near a target value  
- Prevent overheating  
- Reduce energy consumption  
- Compare different control strategies  

---

## 🚀 Features

- 🔥 Live thermal heatmap visualization  
- 🧠 AI-based cooling using Model Predictive Control  
- 📊 Real-time performance metrics and plots  
- 🎛️ Multiple controllers for comparison:
  - No Control  
  - Random Control  
  - MPC Control  
- 🧪 Scenario comparison dashboard  
- 📁 Dataset collection for training dynamics model  
- 🤖 Supports:
  - Linear Dynamics Model  
  - Neural Network Dynamics Model  

---

## 🏗️ Project Structure

```bash
DataCenterCooling/
│
├── env/
│   └── cooling_env.py           # Custom simulation environment
│
├── models/
│   └── dynamics_model.py        # Linear and Neural predictive models
│
├── controllers/
│   └── mpc_controller.py        # Cooling strategies and MPC logic
│
├── data/
│   └── data_collector.py        # Collects training dataset
│
├── dashboard/
│   └── dashboard.py             # Streamlit interactive dashboard
│
├── trained_models/
│   ├── linear_model.pkl
│   └── neural_model.pkl
│
└── README.md



#⚙️ Working of the Project
The project follows the below execution pipeline:

##Step 1: Initialize Environment
The custom environment simulates a real-world data center.

It initializes:
1.Grid of server temperatures
2.Ambient temperature
3.Heat generation pattern
4.Cooling zones

env = DataCenterEnv(seed=42)

The environment provides:
1.State → Current temperature values of all server zones
2.Action Space → Cooling power values for each cooling zone
3.Reward → Performance score for each action

##Step 2: Collect Training Data
A smooth random controller explores the environment and collects transitions:
(state, action, next_state)

Example:
next_state, reward, done, info = env.step(action)

This dataset is stored and later used to train predictive models.
Run: python data_collector.py


##Step 3: Train Dynamics Model
The predictive model learns environment behavior:
next state=f(state,action)

This helps estimate future temperatures without running the real environment repeatedly.

Two models are supported:
1.Linear Model (Ridge Regression)
2.Neural Model (MLP Regressor)


##Step 4: Controller Selection
The user selects one of three controllers from the dashboard.
1. No Control: No cooling is applied.
action = [0,0,0,0]
Temperature rises continuously.

2. Random Control: Random cooling values are applied.
action = random values
This may reduce heat but wastes energy.

3. MPC Control: Model Predictive Control uses the trained model to predict future states and choose the best action.
It minimizes:
<img width="328" height="99" alt="image" src="https://github.com/user-attachments/assets/631d630d-8717-42ff-8424-4b9e0f44a10f" />>

Where:
-->Temperature deviation is penalized
-->High energy usage is penalized
The action with minimum cost is selected.

##Step 5: Apply Action
The selected action is passed to environment:
next_state, reward, done, info = env.step(action)

Environment updates:
-->New temperatures
-->Energy consumed
-->Reward value
-->Overheating status


##Step 6: Dashboard Visualization
The Streamlit dashboard updates in real-time and displays:
🔥 Live temperature heatmap
🌬 Cooling power of each zone
📈 Mean temperature graph
⚡ Energy graph
🏆 Reward graph

Run dashboard:
streamlit run dashboard/dashboard.py



##Step 7: Scenario Comparison

The system runs multiple controllers for fixed steps and compares:
-->Average temperature
-->Maximum temperature
-->Energy consumption
-->Average reward

This helps evaluate controller performance.
🔄 Flow Summary
Initialize Environment
        ↓
Collect Data
        ↓
Train Dynamics Model
        ↓
Select Controller
        ↓
Predict Best Action
        ↓
Apply Action
        ↓
Update Dashboard
        ↓
Repeat Until Done


📊 Dashboard

The interactive dashboard includes:

Live heatmap 🔥
Cooling zone visualization 🌬️
Temperature / energy / reward graphs 📈
Controller comparison 📊
🛠️ Installation


Clone repository:

git clone <your_repo_link>
cd DataCenterCooling

Install dependencies:

pip install -r requirements.txt
📦 Dependencies

Main libraries used:
Python
NumPy
Pandas
Matplotlib
Streamlit
Scikit-learn
PyTorch / TensorFlow (if neural model used)



📈 Results

The MPC Controller performs better than random and no control by:

✅ Maintaining stable temperature
✅ Preventing overheating
✅ Reducing energy usage
✅ Maximizing cumulative reward



🔍 Future Improvements
Deep Reinforcement Learning (DQN / PPO)
Real IoT sensor integration
Multi-agent cooling control
Cloud deployment


👩‍💻 Author

Developed by Pariksha Sonawane, Manaswi Patil, Vishva Shah
AIML Student 🚀
