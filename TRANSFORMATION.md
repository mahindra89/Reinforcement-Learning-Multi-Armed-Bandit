# Coursework to Portfolio Transformation

This repository now keeps both versions of the project:

- `original/` preserves the initial coursework-style version.
- The repository root contains the portfolio-ready Streamlit app for Hugging Face Spaces.

## What Changed

| Area | Original version | Portfolio version |
|---|---|---|
| Framing | Homework assignment | Interactive decision-systems simulator |
| Interface | Static Python/Colab script | Streamlit web app |
| Code organization | One script with plotting and simulation together | Reusable simulation module plus app UI |
| User control | Fixed experiment settings | Sliders for steps, epsilon, UCB confidence, seed, and arm setup |
| Deployment | Local notebook/script only | Ready for Hugging Face Spaces |

## Files To Compare

- Original README: `original/README.md`
- Original script: `original/rla1-2.py`
- Portfolio README: `README.md`
- Interactive app: `app.py`
- Reusable simulation logic: `bandit_simulator.py`

The goal was not to hide the academic origin. The goal was to preserve it as
source material and wrap the same reinforcement learning idea in a cleaner,
more useful project presentation.
