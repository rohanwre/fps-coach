extends CharacterBody3D
class_name EnemyController

signal shot_blocked(reason: String, aim_alignment: float, line_of_sight: bool, reaction_elapsed: float)

## Enemy movement/weapon actuator.
##
## When no Python trainer is connected, this script runs a tiny scripted
## fallback. When the Godot RL Sync node is connected, EnemyAIController3D
## supplies the movement, turn, and shoot inputs through set_control_input().

@export var health_path: NodePath
@export var weapon_path: NodePath
@export var target_path: NodePath

@export var move_speed := 4.0
@export var gravity := 24.0
@export var preferred_distance := 7.0
@export var stopping_distance := 4.0
@export var aim_height := 0.6
@export var turn_speed := 4.0
@export var ai_controller_path: NodePath
@export_enum("easy", "medium", "hard", "adaptive") var scripted_profile := "adaptive"
@export var adaptive_profile_change_streak := 2
@export var model_min_reaction_delay_seconds := 0.18
@export var model_shot_alignment_threshold := 0.15
@export var move_input_smoothing_rate := 5.0
@export var turn_input_smoothing_rate := 12.0
@export var model_move_deadzone := 0.15
@export var model_aim_assist_enabled := false
@export var model_aim_assist_turn_multiplier := 1.0
@export var model_manual_turn_influence := 0.35
@export var model_shoot_action_threshold := 0.25
@export var model_assisted_shots_use_target_point := true
@export var model_assisted_shot_spread_degrees := 0.0
@export_enum("easy", "medium", "hard", "adaptive") var model_difficulty_profile := "adaptive"

@onready var health: Health = get_node(health_path)
@onready var weapon: RaycastWeapon = get_node(weapon_path)
@onready var target: Node3D = get_node(target_path)
@onready var ai_controller: Node = get_node_or_null(ai_controller_path)

var _spawn_transform := Transform3D.IDENTITY
var _target_move_input := Vector2.ZERO
var _target_turn_input := 0.0
var _target_shoot_requested := false
var _move_input := Vector2.ZERO
var _turn_input := 0.0
var _shoot_requested := false
var _line_of_sight_elapsed := 0.0
var _has_line_of_sight_this_tick := false
var _cooldown_block_suppressed := false
var _strafe_direction := 1.0
var _strafe_flip_timer := 0.0
var _adaptive_profile_index := 1
var _last_skill_trend := "stable"
var _skill_trend_streak := 0

const PROFILE_SETTINGS := {
	"easy": {
		"preferred_distance": 8.5,
		"stopping_distance": 5.0,
		"strafe_strength": 0.25,
		"turn_multiplier": 0.7,
		"aggression": 0.65,
		"shot_alignment_threshold": 0.55,
		"reaction_delay": 0.55,
	},
	"medium": {
		"preferred_distance": 7.0,
		"stopping_distance": 4.0,
		"strafe_strength": 0.55,
		"turn_multiplier": 1.0,
		"aggression": 1.0,
		"shot_alignment_threshold": 0.35,
		"reaction_delay": 0.36,
	},
	"hard": {
		"preferred_distance": 6.0,
		"stopping_distance": 3.0,
		"strafe_strength": 0.85,
		"turn_multiplier": 1.35,
		"aggression": 1.25,
		"shot_alignment_threshold": 0.2,
		"reaction_delay": 0.22,
	},
}
const ADAPTIVE_PROFILE_ORDER := ["easy", "medium", "hard"]
const MODEL_DIFFICULTY_SETTINGS := {
	"easy": {
		"reaction_delay": 0.4,
		"shot_alignment_threshold": 0.75,
		"aim_assist_turn_multiplier": 0.7,
		"manual_turn_influence": 0.2,
		"move_speed_multiplier": 0.85,
		"shoot_action_threshold": 0.15,
	},
	"medium": {
		"reaction_delay": 0.25,
		"shot_alignment_threshold": 0.7,
		"aim_assist_turn_multiplier": 1.0,
		"manual_turn_influence": 0.35,
		"move_speed_multiplier": 1.0,
		"shoot_action_threshold": -0.1,
	},
	"hard": {
		"reaction_delay": 0.175,
		"shot_alignment_threshold": 0.55,
		"aim_assist_turn_multiplier": 1.2,
		"manual_turn_influence": 0.5,
		"move_speed_multiplier": 1.1,
		"shoot_action_threshold": -0.25,
	},
}


func _ready() -> void:
	_spawn_transform = global_transform
	_apply_model_difficulty_settings()
	if ai_controller != null and ai_controller.has_method("init"):
		ai_controller.init(self)


func _physics_process(delta: float) -> void:
	if health.is_dead:
		velocity = Vector3.ZERO
		return

	_update_visibility_state(delta)
	if weapon.can_shoot():
		_cooldown_block_suppressed = false

	if _is_model_controlled():
		_apply_control_input(delta)
	else:
		_apply_scripted_fallback(delta)


func reset_for_round() -> void:
	global_transform = _spawn_transform
	velocity = Vector3.ZERO
	health.reset_health()
	weapon.reset_weapon()
	clear_control_input()
	_line_of_sight_elapsed = 0.0
	_has_line_of_sight_this_tick = false
	_cooldown_block_suppressed = false
	_strafe_direction = 1.0
	_strafe_flip_timer = 0.0
	if ai_controller != null and ai_controller.has_method("on_round_reset"):
		ai_controller.on_round_reset()


func set_control_input(move: Vector2, turn: float, shoot: bool) -> void:
	# Shared input surface for both future RL control and simple scripted tests.
	# move.x: strafe right/left in local space, [-1, 1].
	# move.y: move forward/back in local space, [-1, 1].
	# turn: yaw right/left, [-1, 1].
	# shoot: request one raycast shot this physics tick.
	_target_move_input = move.limit_length(1.0)
	_target_turn_input = clamp(turn, -1.0, 1.0)
	_target_shoot_requested = shoot


func clear_control_input() -> void:
	_target_move_input = Vector2.ZERO
	_target_turn_input = 0.0
	_target_shoot_requested = false
	_move_input = Vector2.ZERO
	_turn_input = 0.0
	_shoot_requested = false


func _is_model_controlled() -> bool:
	return ai_controller != null and ai_controller.get("heuristic") == "model"


func is_model_controlled() -> bool:
	return _is_model_controlled()


func get_active_scripted_profile() -> String:
	if scripted_profile != "adaptive":
		return scripted_profile
	return ADAPTIVE_PROFILE_ORDER[_adaptive_profile_index]


func get_active_model_difficulty_profile() -> String:
	if model_difficulty_profile != "adaptive":
		return model_difficulty_profile
	return ADAPTIVE_PROFILE_ORDER[_adaptive_profile_index]


func set_adaptive_skill_trend(trend: String) -> void:
	if scripted_profile != "adaptive" and model_difficulty_profile != "adaptive":
		return

	if trend == "stable":
		_last_skill_trend = trend
		_skill_trend_streak = 0
		return

	if trend == _last_skill_trend:
		_skill_trend_streak += 1
	else:
		_last_skill_trend = trend
		_skill_trend_streak = 1

	if _skill_trend_streak < adaptive_profile_change_streak:
		return

	if trend == "improving":
		_adaptive_profile_index = mini(_adaptive_profile_index + 1, ADAPTIVE_PROFILE_ORDER.size() - 1)
	elif trend == "struggling":
		_adaptive_profile_index = maxi(_adaptive_profile_index - 1, 0)
	_skill_trend_streak = 0
	_apply_model_difficulty_settings()


func set_model_difficulty_profile(profile: String) -> void:
	if profile not in ["easy", "medium", "hard", "adaptive"]:
		return
	model_difficulty_profile = profile
	_apply_model_difficulty_settings()


func get_model_difficulty_config() -> Dictionary:
	var profile := get_active_model_difficulty_profile()
	var config: Dictionary = MODEL_DIFFICULTY_SETTINGS.get(profile, MODEL_DIFFICULTY_SETTINGS["medium"]).duplicate()
	config["profile"] = profile
	config["configured_profile"] = model_difficulty_profile
	return config


func get_line_of_sight_elapsed() -> float:
	return _line_of_sight_elapsed


func get_model_min_reaction_delay_seconds() -> float:
	return model_min_reaction_delay_seconds


func get_model_shot_alignment_threshold() -> float:
	return model_shot_alignment_threshold


func get_model_shoot_action_threshold() -> float:
	return model_shoot_action_threshold


func set_model_shot_alignment_threshold(value: float) -> void:
	model_shot_alignment_threshold = clampf(value, -1.0, 1.0)


func set_model_aim_assist_enabled(value: bool) -> void:
	model_aim_assist_enabled = value


func set_model_aim_assist_turn_multiplier(value: float) -> void:
	model_aim_assist_turn_multiplier = maxf(value, 0.0)


func set_model_manual_turn_influence(value: float) -> void:
	model_manual_turn_influence = clampf(value, 0.0, 1.0)


func set_model_assisted_shots_use_target_point(value: bool) -> void:
	model_assisted_shots_use_target_point = value


func set_model_assisted_shot_spread_degrees(value: float) -> void:
	model_assisted_shot_spread_degrees = maxf(value, 0.0)


func _apply_scripted_fallback(delta: float) -> void:
	# Heuristic baseline used for non-RL play and curriculum warm-up.
	# The adaptive profile maps current duel state to easy/medium/hard behavior.
	var profile: Dictionary = _resolve_scripted_profile()
	var turn_multiplier: float = float(profile["turn_multiplier"])
	_face_target_with_speed(delta, turn_multiplier)

	var to_target := target.global_position - global_position
	to_target.y = 0.0
	var distance := to_target.length()
	var forward_direction := Vector3.ZERO

	var desired_preferred: float = float(profile["preferred_distance"])
	var desired_stopping: float = float(profile["stopping_distance"])
	if distance > desired_preferred:
		forward_direction = to_target.normalized()
	elif distance < desired_stopping:
		forward_direction = -to_target.normalized()

	_strafe_flip_timer -= delta
	if _strafe_flip_timer <= 0.0:
		_strafe_direction *= -1.0
		_strafe_flip_timer = randf_range(0.35, 1.15)

	var to_target_horizontal := to_target.normalized() if distance > 0.001 else -global_basis.z
	var strafe_direction := to_target_horizontal.cross(Vector3.UP).normalized() * _strafe_direction
	var strafe_strength: float = float(profile["strafe_strength"])
	var aggression: float = float(profile["aggression"])
	var move_direction := (forward_direction * aggression) + (strafe_direction * strafe_strength)
	move_direction = _apply_scripted_obstacle_avoidance(move_direction, strafe_direction)
	_apply_world_movement(move_direction, delta)

	var threshold: float = float(profile["shot_alignment_threshold"])
	var reaction_delay: float = float(profile["reaction_delay"])
	if weapon.can_shoot() and _has_line_of_sight_this_tick:
		_try_shoot_at_target(threshold, reaction_delay)


func _apply_control_input(delta: float) -> void:
	var move_blend := clampf(move_input_smoothing_rate * delta, 0.0, 1.0)
	var turn_blend := clampf(turn_input_smoothing_rate * delta, 0.0, 1.0)
	_move_input = _move_input.lerp(_target_move_input, move_blend)
	_turn_input = lerpf(_turn_input, _target_turn_input, turn_blend)
	_shoot_requested = _target_shoot_requested
	_target_shoot_requested = false

	if model_aim_assist_enabled and _has_line_of_sight_this_tick:
		_face_target_with_speed(delta, model_aim_assist_turn_multiplier)
		rotate_y(-_turn_input * turn_speed * model_manual_turn_influence * delta)
	else:
		rotate_y(-_turn_input * turn_speed * delta)

	# Convert local RL movement inputs into a world-space movement direction.
	var effective_move := _move_input
	if effective_move.length() < model_move_deadzone:
		effective_move = Vector2.ZERO
	var move_direction := (global_basis.x * effective_move.x) + (-global_basis.z * effective_move.y)
	_apply_world_movement(move_direction, delta)

	if _shoot_requested:
		if model_aim_assist_enabled and model_assisted_shots_use_target_point:
			_try_shoot_at_target(model_shot_alignment_threshold, model_min_reaction_delay_seconds)
		else:
			_try_shoot_forward(model_shot_alignment_threshold, model_min_reaction_delay_seconds)


func _update_visibility_state(delta: float) -> void:
	_has_line_of_sight_this_tick = has_line_of_sight_to_target()
	if _has_line_of_sight_this_tick:
		_line_of_sight_elapsed += delta
	else:
		_line_of_sight_elapsed = 0.0


func _face_target_with_speed(delta: float, turn_multiplier: float) -> void:
	var target_position := target.global_position
	target_position.y = global_position.y

	var to_target := target_position - global_position
	if to_target.length_squared() <= 0.001:
		return

	var desired_dir: Vector3 = to_target.normalized()
	var current_dir: Vector3 = -global_basis.z
	var blend: float = clampf(turn_speed * turn_multiplier * delta, 0.0, 1.0)
	var blended_dir := current_dir.slerp(desired_dir, blend)
	look_at(global_position + blended_dir, Vector3.UP)


func _apply_world_movement(move_direction: Vector3, delta: float) -> void:
	if move_direction.length() > 1.0:
		move_direction = move_direction.normalized()
	var speed_multiplier := 1.0
	if _is_model_controlled():
		speed_multiplier = float(MODEL_DIFFICULTY_SETTINGS.get(
			get_active_model_difficulty_profile(),
			MODEL_DIFFICULTY_SETTINGS["medium"]
		)["move_speed_multiplier"])
	velocity.x = move_direction.x * move_speed * speed_multiplier
	velocity.z = move_direction.z * move_speed * speed_multiplier

	if not is_on_floor():
		velocity.y -= gravity * delta
	else:
		velocity.y = 0.0

	move_and_slide()


func shoot_forward() -> void:
	var origin := weapon.global_position
	var direction := -global_basis.z
	weapon.shoot(origin, direction, self)


func shoot_at_target() -> void:
	var origin := weapon.global_position
	var target_point := target.global_position + Vector3.UP * aim_height
	var direction := _apply_assisted_shot_spread(target_point - origin)
	weapon.shoot(origin, direction, self)


func _apply_assisted_shot_spread(direction: Vector3) -> Vector3:
	if model_assisted_shot_spread_degrees <= 0.0:
		return direction
	var normalized := direction.normalized()
	var right := normalized.cross(Vector3.UP)
	if right.length_squared() <= 0.001:
		right = Vector3.RIGHT
	right = right.normalized()
	var up := right.cross(normalized).normalized()
	var spread_radius := tan(deg_to_rad(model_assisted_shot_spread_degrees))
	var offset := (
		right * randf_range(-spread_radius, spread_radius)
		+ up * randf_range(-spread_radius, spread_radius)
	)
	return (normalized + offset).normalized()


func _try_shoot_forward(alignment_threshold: float, reaction_delay: float) -> bool:
	if not _can_fire_humanlike(alignment_threshold, reaction_delay):
		return false
	shoot_forward()
	if _is_model_controlled():
		_cooldown_block_suppressed = true
	return true


func _try_shoot_at_target(alignment_threshold: float, reaction_delay: float) -> bool:
	if not _can_fire_humanlike(alignment_threshold, reaction_delay):
		return false
	shoot_at_target()
	return true


func _can_fire_humanlike(alignment_threshold: float, reaction_delay: float) -> bool:
	var alignment := get_aim_alignment()
	if not weapon.can_shoot():
		if not (_is_model_controlled() and _cooldown_block_suppressed):
			_emit_shot_blocked("cooldown", alignment)
		return false
	if not _has_line_of_sight_this_tick:
		_emit_shot_blocked("no_line_of_sight", alignment)
		return false
	if _line_of_sight_elapsed < reaction_delay:
		_emit_shot_blocked("reaction_delay", alignment)
		return false
	if alignment < alignment_threshold:
		_emit_shot_blocked("low_alignment", alignment)
		return false
	return true


func _emit_shot_blocked(reason: String, alignment: float) -> void:
	shot_blocked.emit(reason, alignment, _has_line_of_sight_this_tick, _line_of_sight_elapsed)


func has_line_of_sight_to_target() -> bool:
	var origin := weapon.global_position
	var target_point := target.global_position + Vector3.UP * aim_height
	return _has_line_of_sight(origin, target_point - origin)


func get_direction_to_target_world() -> Vector3:
	return (target.global_position - global_position).normalized()


func get_direction_to_target_local() -> Vector3:
	return global_basis.inverse() * get_direction_to_target_world()


func get_distance_to_target() -> float:
	return global_position.distance_to(target.global_position)


func get_aim_alignment() -> float:
	var to_target := get_direction_to_target_world()
	var forward := -global_basis.z
	return clamp(forward.dot(to_target), -1.0, 1.0)


func get_obstacle_clearance(direction: Vector3, max_distance := 4.0) -> float:
	if direction.length_squared() <= 0.001 or max_distance <= 0.0:
		return 1.0
	var origin := global_position + Vector3.UP * 0.5
	var query := PhysicsRayQueryParameters3D.create(origin, origin + direction.normalized() * max_distance)
	query.exclude = [get_rid()]
	var result := get_world_3d().direct_space_state.intersect_ray(query)
	if result.is_empty() or result.get("collider") == target:
		return 1.0
	var hit_position: Vector3 = result.get("position", origin)
	return clampf(origin.distance_to(hit_position) / max_distance, 0.0, 1.0)


func _apply_scripted_obstacle_avoidance(move_direction: Vector3, fallback_strafe: Vector3) -> Vector3:
	if move_direction.length_squared() <= 0.001:
		return move_direction
	var clearance := get_obstacle_clearance(move_direction, 1.8)
	if clearance >= 0.75:
		return move_direction
	return (move_direction * clearance) + (fallback_strafe * (1.0 - clearance) * 1.5)


func _resolve_scripted_profile() -> Dictionary:
	if scripted_profile != "adaptive":
		return PROFILE_SETTINGS.get(scripted_profile, PROFILE_SETTINGS["medium"])

	var active_profile := get_active_scripted_profile()
	var enemy_health_fraction := health.get_health_fraction()
	var target_controller := target as PlayerController
	var target_health_node := target_controller.health if target_controller != null else null
	var target_health_fraction := target_health_node.get_health_fraction() if target_health_node != null else 1.0
	var health_advantage := enemy_health_fraction - target_health_fraction

	if health_advantage > 0.25:
		return PROFILE_SETTINGS["hard"]
	if health_advantage < -0.2:
		return PROFILE_SETTINGS["easy"]
	return PROFILE_SETTINGS.get(active_profile, PROFILE_SETTINGS["medium"])


func _apply_model_difficulty_settings() -> void:
	var settings: Dictionary = MODEL_DIFFICULTY_SETTINGS.get(
		get_active_model_difficulty_profile(),
		MODEL_DIFFICULTY_SETTINGS["medium"]
	)
	model_min_reaction_delay_seconds = float(settings["reaction_delay"])
	model_shot_alignment_threshold = float(settings["shot_alignment_threshold"])
	model_aim_assist_turn_multiplier = float(settings["aim_assist_turn_multiplier"])
	model_manual_turn_influence = float(settings["manual_turn_influence"])
	model_shoot_action_threshold = float(settings["shoot_action_threshold"])


func _has_line_of_sight(origin: Vector3, direction: Vector3) -> bool:
	var query := PhysicsRayQueryParameters3D.create(origin, origin + direction.normalized() * weapon.max_range)
	query.exclude = [get_rid()]

	var result := get_world_3d().direct_space_state.intersect_ray(query)
	if result.is_empty():
		return false

	var collider := result.get("collider") as Node
	return collider == target
