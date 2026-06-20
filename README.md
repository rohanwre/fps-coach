# Adaptive FPS Coach

Adaptive FPS Coach is a Godot and Python prototype for an automatic gameplay
agent in an open-source FPS-style environment. The project has two main parts:

- a gameplay agent that controls enemy NPC behavior for player practice
- a coaching agent that converts gameplay events and player-view evidence into
  tactical feedback

The Godot project provides the arena, player, enemies, weapons, cover objects,
health, line-of-sight checks, and gameplay logging. The Python side provides PPO
training/evaluation scripts and coaching utilities.

## Repository Layout

- `godot_project/`: Godot 4 project, game scripts, scenes, and RL integration.
- `training/`: PPO training, policy playback, scripted baselines, benchmarks,
  and gameplay-log evaluation.
- `coaching/`: event-window building, VLM-compatible visual evidence parsing,
  retrieval-grounded coaching, and coaching metrics.
- `coaching/knowledge/tips.md`: small local coaching knowledge base used by the
  retrieval workflow.
- `tests/`: standard-library unit tests for training and coaching scripts.

Generated training logs, trained models, screenshots, presentation artifacts,
and report artifacts are intentionally ignored by git.

## Requirements

- Python 3.12 recommended
- Godot 4.x with Mono support
- Local Godot RL Agents Python package available at `../godot_rl_agents`, or an
  equivalent editable install

Install Python dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run Checks

Run the unit tests:

```bash
.venv/bin/python -m unittest discover tests
```

Compile the Python modules:

```bash
.venv/bin/python -m compileall coaching training tests
```

## Run The Godot Project

Open `godot_project/project.godot` in Godot and press Play. If no Python RL
server is connected, the enemy uses its scripted fallback behavior.

For a headless smoke run, use your local Godot executable:

```bash
Godot --headless --path godot_project
```

On macOS, the executable may be inside a `.app` bundle, for example:

```bash
/path/to/Godot_mono.app/Contents/MacOS/Godot --headless --path godot_project
```

## Train A PPO Agent

Start Python first so Godot can connect to the RL sync server:

```bash
MPLCONFIGDIR=.mplconfig XDG_CACHE_HOME=.cache \
.venv/bin/python training/train_enemy.py \
  --timesteps 512 \
  --eval-episodes 1 \
  --experiment-name enemy_ppo_smoke \
  --save-model-path training/models/enemy_ppo_smoke.zip \
  --save-metadata-path training/models/metadata/enemy_ppo_smoke.json
```

Then run the Godot project. For longer experiments, increase `--timesteps` and
use the benchmark scripts in `training/`.

To train the player-side policy against a scripted enemy, add:

```bash
--agent-role player --scripted-enemy-profile hard
```

## Evaluate Gameplay Logs

Godot writes gameplay logs to its app user-data directory. Evaluate a log with:

```bash
.venv/bin/python training/evaluate_gameplay_logs.py \
  --input path/to/gameplay_log.jsonl \
  --output training/models/evaluation_metrics.json
```

Useful metrics include round completions, timeouts, shots fired, hits landed,
hit rate, damage dealt, damage taken, line-of-sight violations, and movement
jitter warnings.

## Run Coaching Utilities

Build coaching windows from a gameplay log:

```bash
.venv/bin/python coaching/build_event_windows.py \
  --input path/to/gameplay_log.jsonl \
  --output coaching/output/windows.jsonl
```

Run coaching with offline visual-evidence fallback:

```bash
.venv/bin/python coaching/run_visual_coach.py \
  --windows coaching/output/windows.jsonl \
  --output coaching/output/coaching.jsonl
```

If a local OpenAI-compatible VLM server is available, pass its base URL and
model name using the options exposed by `coaching/run_visual_coach.py`.

## Notes For Review

This repository is intentionally kept focused on source code and runnable
configuration. Generated presentation decks, final-report assets, model
checkpoints, logs, and local screenshots are excluded so the codebase stays
small enough for inspection.
