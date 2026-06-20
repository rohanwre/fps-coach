extends AIController3D
class_name EnemyAIController3D

## Godot RL Agents bridge for the enemy.
##
## This node owns the RL protocol only: observations, rewards, actions, done,
## and reset bookkeeping. EnemyController still owns movement and shooting.

@export var enemy_path: NodePath = NodePath("..")
@export var target_path: NodePath = NodePath("../../Player")
@export var game_path: NodePath = NodePath("../..")
@export var max_episode_steps := 900
@export var blocked_shot_penalty := 0.02
@export var low_alignment_block_penalty := 0.01
@export var reaction_delay_block_penalty := 0.01
@export var valid_shot_request_reward := 0.01
@export var enemy_miss_penalty := 0.05
@export var time_pressure_penalty := 0.001
@export var aim_alignment_reward_scale := 0.001
@export var timeout_penalty := 1.0
@export var timeout_damage_credit_scale := 0.0
@export var enemy_hit_reward := 1.0
@export var enemy_damage_reward_scale := 1.0
@export var target_death_reward := 10.0
@export var player_health_reduction_reward := 0.0
@export var line_of_sight_reward := 0.0
@export var useful_range_reward := 0.0
@export var useful_range_min := 4.0
@export var useful_range_max := 12.0
@export var centered_aim_reward_scale := 0.0
@export var centered_aim_threshold := 0.75
@export var poorly_aligned_shot_penalty := 0.0
@export var marginal_hit_penalty_threshold := 0.45

const ARENA_HALF_SIZE := 12.0
const ARENA_FULL_SIZE := 24.0
const ARENA_DIAGONAL := 33.95

@onready var enemy: EnemyController = get_node(enemy_path)
@onready var target: PlayerController = get_node(target_path)
@onready var game: Game = get_node(game_path)

var enemy_weapon: RaycastWeapon
var target_weapon: RaycastWeapon
var enemy_health: Health
var target_health: Health
var _needs_timeout_log := false
var _episode_enemy_damage_dealt := 0


func _ready() -> void:
	super._ready()
	_resolve_actor_references()
	_connect_reward_signals()


func _physics_process(_delta: float) -> void:
	n_steps += 1

	# Small time pressure: the enemy should solve the round, not circle forever.
	if heuristic == "model" and not done:
		reward -= time_pressure_penalty
		reward += max(enemy.get_aim_alignment(), 0.0) * aim_alignment_reward_scale
		_apply_round_completion_shaping()
		_apply_centering_shaping()

	if heuristic == "model" and n_steps >= max_episode_steps and not done:
		reward -= _calculate_timeout_penalty()
		done = true
		needs_reset = true
		_needs_timeout_log = true

	if needs_reset:
		needs_reset = false
		n_steps = 0
		enemy.clear_control_input()
		if _needs_timeout_log:
			_needs_timeout_log = false
			game.end_round_for_training_timeout()
		else:
			game.reset_round_for_training()


func get_obs() -> Dictionary:
	var relative := target.global_position - enemy.global_position
	var local_direction := enemy.get_direction_to_target_local()
	var enemy_forward := -enemy.global_basis.z
	var weapon_ready := enemy_weapon != null and enemy_weapon.can_shoot()
	var line_of_sight := _has_line_of_sight_to_target()
	var aim_alignment := enemy.get_aim_alignment()
	var reaction_delay := maxf(enemy.get_model_min_reaction_delay_seconds(), 0.001)
	var line_of_sight_elapsed_norm := clampf(enemy.get_line_of_sight_elapsed() / reaction_delay, 0.0, 1.0)
	var alignment_margin := clampf(aim_alignment - enemy.get_model_shot_alignment_threshold(), -1.0, 1.0)
	var humanlike_shot_ready := weapon_ready and line_of_sight and line_of_sight_elapsed_norm >= 1.0 and alignment_margin >= 0.0
	var enemy_left := -enemy.global_basis.x
	var enemy_right := enemy.global_basis.x

	# Observation values are deliberately compact scalar features. They are
	# easier to debug than pixels and enough for the first RL wiring pass.
	var obs := [
		# 0-1: Enemy x/z position normalized to the arena half-size.
		enemy.global_position.x / ARENA_HALF_SIZE,
		enemy.global_position.z / ARENA_HALF_SIZE,

		# 2-3: Player position relative to enemy, normalized to arena width.
		relative.x / ARENA_FULL_SIZE,
		relative.z / ARENA_FULL_SIZE,

		# 4: Distance to player normalized by arena diagonal.
		enemy.get_distance_to_target() / ARENA_DIAGONAL,

		# 5-6: Direction to player in enemy-local space.
		local_direction.x,
		local_direction.z,

		# 7-8: Enemy facing direction in world x/z.
		enemy_forward.x,
		enemy_forward.z,

		# 9-10: Enemy ground velocity normalized by enemy move speed.
		enemy.velocity.x / enemy.move_speed,
		enemy.velocity.z / enemy.move_speed,

		# 11-12: Health fractions for self and opponent.
		enemy_health.get_health_fraction() if enemy_health != null else 0.0,
		target_health.get_health_fraction() if target_health != null else 0.0,

		# 13: Weapon readiness. 1 means ready to shoot, 0 means cooling down.
		1.0 if weapon_ready else 0.0,

		# 14: Line of sight. 1 means the raycast weapon can see the player.
		1.0 if line_of_sight else 0.0,

		# 15: Aim alignment. 1 means facing player, -1 means facing away.
		aim_alignment,

		# 16: Reaction-delay progress. 1 means the human-like delay is satisfied.
		line_of_sight_elapsed_norm,

		# 17: Alignment margin above/below the model shot threshold.
		alignment_margin,

		# 18: Human-like shot readiness after cooldown, visibility, delay, and aim.
		1.0 if humanlike_shot_ready else 0.0,

		# 19-21: Normalized obstacle clearance forward, left, and right.
		enemy.get_obstacle_clearance(enemy_forward),
		enemy.get_obstacle_clearance(enemy_left),
		enemy.get_obstacle_clearance(enemy_right),
	]

	return {"obs": obs}


func get_reward() -> float:
	return reward


func get_action_space() -> Dictionary:
	return {
		# move[0]: strafe right/left, move[1]: move forward/back.
		"move": {"size": 2, "action_type": "continuous"},

		# turn[0]: yaw input. Positive/negative values rotate the enemy.
		"turn": {"size": 1, "action_type": "continuous"},

		# shoot[0]: fire when greater than the active difficulty threshold.
		"shoot": {"size": 1, "action_type": "continuous"},
	}


func set_action(action) -> void:
	var move := Vector2(action["move"][0], action["move"][1])
	var turn := float(action["turn"][0])
	var shoot := float(action["shoot"][0]) > enemy.get_model_shoot_action_threshold()
	enemy.set_control_input(move, turn, shoot)


func reset() -> void:
	n_steps = 0
	needs_reset = false
	_needs_timeout_log = false
	reward = 0.0
	done = false
	_episode_enemy_damage_dealt = 0
	enemy.clear_control_input()


func on_round_reset() -> void:
	# Called by EnemyController after Game resets actor state. Keep reward/done
	# untouched so Sync can still report the terminal transition to Python.
	n_steps = 0
	needs_reset = false
	_needs_timeout_log = false
	_episode_enemy_damage_dealt = 0
	enemy.clear_control_input()


func _resolve_actor_references() -> void:
	# EnemyAIController3D is a child of Enemy, so its _ready() can run before
	# EnemyController._ready(). Resolve components directly from the scene tree
	# instead of assuming controller onready fields are already populated.
	enemy_weapon = _resolve_weapon_from_actor(enemy, enemy.weapon_path, "enemy")
	target_weapon = _resolve_weapon_from_actor(target, target.weapon_path, "target")
	enemy_health = _resolve_health_from_actor(enemy, enemy.health_path, "enemy")
	target_health = _resolve_health_from_actor(target, target.health_path, "target")


func _resolve_weapon_from_actor(actor: Node, weapon_path: NodePath, label: String) -> RaycastWeapon:
	if actor == null:
		push_warning("EnemyAIController3D: Cannot resolve %s weapon because actor is null." % label)
		return null

	if str(weapon_path).is_empty():
		push_warning("EnemyAIController3D: Cannot resolve %s weapon because weapon_path is empty." % label)
		return null

	var weapon_node := actor.get_node_or_null(weapon_path)
	if weapon_node == null:
		push_warning(
			"EnemyAIController3D: Could not find %s weapon at %s/%s."
			% [label, actor.get_path(), weapon_path]
		)
		return null

	if not (weapon_node is RaycastWeapon):
		push_warning(
			"EnemyAIController3D: Node at %s/%s is not a RaycastWeapon."
			% [actor.get_path(), weapon_path]
		)
		return null

	return weapon_node as RaycastWeapon


func _resolve_health_from_actor(actor: Node, health_path: NodePath, label: String) -> Health:
	if actor == null:
		push_warning("EnemyAIController3D: Cannot resolve %s health because actor is null." % label)
		return null

	if str(health_path).is_empty():
		push_warning("EnemyAIController3D: Cannot resolve %s health because health_path is empty." % label)
		return null

	var health_node := actor.get_node_or_null(health_path)
	if health_node == null:
		push_warning(
			"EnemyAIController3D: Could not find %s health at %s/%s."
			% [label, actor.get_path(), health_path]
		)
		return null

	if not (health_node is Health):
		push_warning(
			"EnemyAIController3D: Node at %s/%s is not a Health component."
			% [actor.get_path(), health_path]
		)
		return null

	return health_node as Health


func _has_line_of_sight_to_target() -> bool:
	if enemy_weapon == null or target == null:
		return false

	var origin := enemy_weapon.global_position
	var target_point := target.global_position + Vector3.UP * enemy.aim_height
	var direction := target_point - origin
	var query := PhysicsRayQueryParameters3D.create(origin, origin + direction.normalized() * enemy_weapon.max_range)
	query.exclude = [enemy.get_rid()]

	var result := get_world_3d().direct_space_state.intersect_ray(query)
	if result.is_empty():
		return false

	return result.get("collider") == target


func _connect_reward_signals() -> void:
	if enemy != null:
		enemy.shot_blocked.connect(_on_enemy_shot_blocked)

	if enemy_weapon != null:
		enemy_weapon.hit_landed.connect(_on_weapon_hit)
		enemy_weapon.shot_missed.connect(_on_weapon_missed)
	else:
		push_warning("EnemyAIController3D: Enemy weapon is null; enemy shot rewards will not be connected.")

	if target_weapon != null:
		target_weapon.hit_landed.connect(_on_weapon_hit)
		target_weapon.shot_missed.connect(_on_weapon_missed)
	else:
		push_warning("EnemyAIController3D: Target weapon is null; incoming shot penalties will not be connected.")

	if target_health != null:
		target_health.damaged.connect(_on_target_damaged)
		target_health.died.connect(_on_target_died)
	else:
		push_warning("EnemyAIController3D: Target health is null; target damage/death rewards will not be connected.")

	if enemy_health != null:
		enemy_health.damaged.connect(_on_enemy_damaged)
		enemy_health.died.connect(_on_enemy_died)
	else:
		push_warning("EnemyAIController3D: Enemy health is null; enemy damage/death penalties will not be connected.")


func apply_reward_profile(profile: String) -> void:
	if profile == "damage":
		enemy_hit_reward = 1.25
		enemy_damage_reward_scale = 2.0
	elif profile == "timeout":
		time_pressure_penalty = 0.002
		timeout_penalty = 3.0
	elif profile == "timeout_aim_assist":
		time_pressure_penalty = 0.002
		timeout_penalty = 3.0
		valid_shot_request_reward = 0.02
		enemy_miss_penalty = 0.1
		enemy_hit_reward = 2.0
		enemy_damage_reward_scale = 3.0
		player_health_reduction_reward = 1.0
		aim_alignment_reward_scale = 0.00125
		centered_aim_reward_scale = 0.0015
		centered_aim_threshold = 0.85
		blocked_shot_penalty = 0.025
		low_alignment_block_penalty = 0.01
		enemy.set_model_aim_assist_enabled(true)
		enemy.set_model_difficulty_profile(enemy.model_difficulty_profile)
		enemy.set_model_assisted_shots_use_target_point(true)
		enemy.set_model_assisted_shot_spread_degrees(0.0)
	elif profile == "volume_combat":
		time_pressure_penalty = 0.0025
		timeout_penalty = 3.0
		valid_shot_request_reward = 0.08
		enemy_miss_penalty = 0.015
		enemy_hit_reward = 1.25
		enemy_damage_reward_scale = 4.0
		player_health_reduction_reward = 2.0
		target_death_reward = 18.0
		aim_alignment_reward_scale = 0.001
		centered_aim_reward_scale = 0.00075
		centered_aim_threshold = 0.45
		blocked_shot_penalty = 0.015
		low_alignment_block_penalty = 0.005
		reaction_delay_block_penalty = 0.005
		enemy.set_model_aim_assist_enabled(true)
		enemy.set_model_difficulty_profile(enemy.model_difficulty_profile)
		enemy.set_model_shot_alignment_threshold(0.25)
		enemy.set_model_assisted_shots_use_target_point(true)
		enemy.set_model_assisted_shot_spread_degrees(4.5)
	elif profile == "timeout_centered":
		time_pressure_penalty = 0.002
		timeout_penalty = 3.0
		aim_alignment_reward_scale = 0.0015
		centered_aim_reward_scale = 0.0015
		centered_aim_threshold = 0.75
		poorly_aligned_shot_penalty = 0.08
		marginal_hit_penalty_threshold = 0.55
		blocked_shot_penalty = 0.025
		low_alignment_block_penalty = 0.025
		enemy.set_model_shot_alignment_threshold(0.55)
	elif profile == "timeout_v2":
		valid_shot_request_reward = 0.0
		enemy_miss_penalty = 0.08
		blocked_shot_penalty = 0.03
		low_alignment_block_penalty = 0.02
		reaction_delay_block_penalty = 0.015
		enemy_hit_reward = 1.5
		enemy_damage_reward_scale = 2.0
		player_health_reduction_reward = 1.0
		time_pressure_penalty = 0.002
		timeout_penalty = 3.0
		timeout_damage_credit_scale = 0.75
		aim_alignment_reward_scale = 0.00125
	elif profile == "pressure":
		enemy_hit_reward = 1.2
		enemy_damage_reward_scale = 1.5
		player_health_reduction_reward = 0.75
		time_pressure_penalty = 0.0015
		timeout_penalty = 2.0
		aim_alignment_reward_scale = 0.00125
		line_of_sight_reward = 0.00025
	elif profile == "range_los":
		aim_alignment_reward_scale = 0.0015
		line_of_sight_reward = 0.0005
		useful_range_reward = 0.0005
	elif profile == "combined":
		enemy_hit_reward = 1.25
		enemy_damage_reward_scale = 2.0
		time_pressure_penalty = 0.002
		timeout_penalty = 3.0
		aim_alignment_reward_scale = 0.0015
		line_of_sight_reward = 0.0005
		useful_range_reward = 0.0005


func get_reward_config() -> Dictionary:
	return {
		"blocked_shot_penalty": blocked_shot_penalty,
		"low_alignment_block_penalty": low_alignment_block_penalty,
		"reaction_delay_block_penalty": reaction_delay_block_penalty,
		"valid_shot_request_reward": valid_shot_request_reward,
		"enemy_miss_penalty": enemy_miss_penalty,
		"time_pressure_penalty": time_pressure_penalty,
		"aim_alignment_reward_scale": aim_alignment_reward_scale,
		"timeout_penalty": timeout_penalty,
		"timeout_damage_credit_scale": timeout_damage_credit_scale,
		"enemy_hit_reward": enemy_hit_reward,
		"enemy_damage_reward_scale": enemy_damage_reward_scale,
		"target_death_reward": target_death_reward,
		"player_health_reduction_reward": player_health_reduction_reward,
		"line_of_sight_reward": line_of_sight_reward,
		"useful_range_reward": useful_range_reward,
		"useful_range_min": useful_range_min,
		"useful_range_max": useful_range_max,
		"centered_aim_reward_scale": centered_aim_reward_scale,
		"centered_aim_threshold": centered_aim_threshold,
		"poorly_aligned_shot_penalty": poorly_aligned_shot_penalty,
		"marginal_hit_penalty_threshold": marginal_hit_penalty_threshold,
		"shoot_action_threshold": enemy.get_model_shoot_action_threshold(),
		"assisted_shots_use_target_point": enemy.model_assisted_shots_use_target_point,
		"assisted_shot_spread_degrees": enemy.model_assisted_shot_spread_degrees,
		"model_difficulty": enemy.get_model_difficulty_config(),
	}


func _apply_round_completion_shaping() -> void:
	if line_of_sight_reward > 0.0 and _has_line_of_sight_to_target():
		reward += line_of_sight_reward

	if useful_range_reward <= 0.0:
		return

	var distance := enemy.get_distance_to_target()
	if distance >= useful_range_min and distance <= useful_range_max:
		reward += useful_range_reward


func _apply_centering_shaping() -> void:
	if centered_aim_reward_scale <= 0.0:
		return
	if not _has_line_of_sight_to_target():
		return

	var aim_alignment := enemy.get_aim_alignment()
	if aim_alignment >= centered_aim_threshold:
		reward += (aim_alignment - centered_aim_threshold) * centered_aim_reward_scale


func _calculate_timeout_penalty() -> float:
	if timeout_damage_credit_scale <= 0.0:
		return timeout_penalty

	var damage_fraction := clampf(float(_episode_enemy_damage_dealt) / 100.0, 0.0, 1.0)
	var credit := clampf(damage_fraction * timeout_damage_credit_scale, 0.0, 0.9)
	return timeout_penalty * (1.0 - credit)


func _on_weapon_hit(shooter_id: String, _target_name: String, _damage: int, _hit_position: Vector3) -> void:
	if shooter_id == "enemy":
		# Reward making a raycast hit, separate from damage amount.
		reward += valid_shot_request_reward
		reward += enemy_hit_reward
		if poorly_aligned_shot_penalty > 0.0 and enemy.get_aim_alignment() < marginal_hit_penalty_threshold:
			reward -= poorly_aligned_shot_penalty
	elif shooter_id == "player":
		# Penalize being hit, separate from damage amount.
		reward -= 1.0


func _on_weapon_missed(shooter_id: String, _hit_position: Vector3) -> void:
	if shooter_id == "enemy":
		# Small ammo/discipline penalty for firing without a hit.
		reward += valid_shot_request_reward
		reward -= enemy_miss_penalty
		if poorly_aligned_shot_penalty > 0.0 and enemy.get_aim_alignment() < marginal_hit_penalty_threshold:
			reward -= poorly_aligned_shot_penalty


func _on_enemy_shot_blocked(reason: String, _aim_alignment: float, _line_of_sight: bool, _reaction_elapsed: float) -> void:
	if heuristic != "model":
		return

	reward -= blocked_shot_penalty
	if reason == "low_alignment":
		reward -= low_alignment_block_penalty
	elif reason == "reaction_delay":
		reward -= reaction_delay_block_penalty


func _on_target_damaged(amount: int, source_id: String, _remaining_health: int) -> void:
	if source_id == "enemy":
		# Reward is proportional to damage dealt to the player.
		_episode_enemy_damage_dealt += amount
		reward += (float(amount) / 100.0) * enemy_damage_reward_scale
		reward += (float(amount) / 100.0) * player_health_reduction_reward


func _on_enemy_damaged(amount: int, source_id: String, _remaining_health: int) -> void:
	if source_id == "player":
		# Penalty is proportional to damage taken from the player.
		reward -= float(amount) / 100.0


func _on_target_died(source_id: String) -> void:
	if heuristic == "model" and source_id == "enemy" and not done:
		# Winning the round is the main sparse positive reward.
		reward += target_death_reward
		done = true


func _on_enemy_died(source_id: String) -> void:
	if heuristic == "model" and source_id == "player" and not done:
		# Losing the round is the main sparse negative reward.
		reward -= 10.0
		done = true
