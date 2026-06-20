extends CharacterBody3D
class_name PlayerController

## Human-controlled FPS capsule.
##
## This is intentionally minimal: WASD movement, mouse look, and click to shoot.
## There is no sprint, crouch, weapon swapping, animation, or networking yet.

@export var health_path: NodePath
@export var weapon_path: NodePath
@export var camera_path: NodePath
@export var target_path: NodePath
@export var ai_controller_path: NodePath

@export var move_speed := 7.0
@export var mouse_sensitivity := 0.0025
@export var gravity := 24.0
@export var model_turn_speed := 4.0
@export var model_aim_assist_enabled := true
@export var model_aim_height := 0.0
@export var model_auto_fire_enabled := true
@export var model_auto_fire_reaction_delay_seconds := 0.25
@export var model_auto_fire_alignment_threshold := 0.985
@export_enum("human", "easy", "medium", "hard", "adaptive") var scripted_profile := "human"
@export var scripted_adaptive_profile_change_streak := 2
@onready var health: Health = get_node(health_path)
@onready var weapon: RaycastWeapon = get_node(weapon_path)
@onready var camera: Camera3D = get_node(camera_path)
@onready var target: Node3D = get_node(target_path)
@onready var ai_controller: Node = get_node_or_null(ai_controller_path)

var _look_pitch := 0.0
var _spawn_transform := Transform3D.IDENTITY
var _model_move_input := Vector2.ZERO
var _model_turn_input := 0.0
var _model_shoot_requested := false
var _model_line_of_sight_elapsed := 0.0
var _scripted_strafe_direction := 1.0
var _scripted_strafe_flip_timer := 0.0
var _scripted_post_shot_timer := 0.0
var _scripted_adaptive_profile_index := 1
var _scripted_last_enemy_trend := "stable"
var _scripted_enemy_trend_streak := 0

const SCRIPTED_PROFILE_SETTINGS := {
	"easy": {
		"preferred_distance": 8.5,
		"stopping_distance": 5.0,
		"strafe_strength": 0.2,
		"turn_multiplier": 0.65,
		"aggression": 0.6,
		"shot_alignment_threshold": 0.999,
		"reaction_delay": 0.9,
		"post_shot_delay": 1.1,
	},
	"medium": {
		"preferred_distance": 7.0,
		"stopping_distance": 4.0,
		"strafe_strength": 0.5,
		"turn_multiplier": 0.9,
		"aggression": 0.9,
		"shot_alignment_threshold": 0.985,
		"reaction_delay": 0.4,
		"post_shot_delay": 0.45,
	},
	"hard": {
		"preferred_distance": 6.0,
		"stopping_distance": 3.0,
		"strafe_strength": 0.8,
		"turn_multiplier": 1.15,
		"aggression": 1.15,
		"shot_alignment_threshold": 0.97,
		"reaction_delay": 0.25,
		"post_shot_delay": 0.25,
	},
}
const SCRIPTED_PROFILE_ORDER := ["easy", "medium", "hard"]


func _ready() -> void:
	_spawn_transform = global_transform
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
	if ai_controller != null and ai_controller.has_method("init"):
		ai_controller.init(self)


func _unhandled_input(event: InputEvent) -> void:
	if health.is_dead:
		return

	if event.is_action_pressed("ui_cancel"):
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
		return

	if event is InputEventMouseButton and event.pressed:
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED

	if Input.mouse_mode == Input.MOUSE_MODE_CAPTURED and event is InputEventMouseMotion:
		_rotate_view(event.relative)

	if event.is_action_pressed("shoot"):
		_shoot()


func _physics_process(delta: float) -> void:
	if health.is_dead:
		velocity = Vector3.ZERO
		return

	if is_model_controlled():
		_apply_model_control(delta)
	elif scripted_profile != "human":
		_apply_scripted_control(delta)
	else:
		_apply_movement(delta)


func reset_for_round() -> void:
	# Called by Game.gd after either actor dies.
	global_transform = _spawn_transform
	velocity = Vector3.ZERO
	_look_pitch = 0.0
	camera.rotation.x = 0.0
	health.reset_health()
	weapon.reset_weapon()
	clear_control_input()
	_scripted_strafe_direction = 1.0
	_scripted_strafe_flip_timer = 0.0
	_scripted_post_shot_timer = 0.0
	if ai_controller != null and ai_controller.has_method("on_round_reset"):
		ai_controller.on_round_reset()


func set_control_input(move: Vector2, turn: float, shoot: bool) -> void:
	_model_move_input = move.limit_length(1.0)
	_model_turn_input = clampf(turn, -1.0, 1.0)
	_model_shoot_requested = shoot


func clear_control_input() -> void:
	_model_move_input = Vector2.ZERO
	_model_turn_input = 0.0
	_model_shoot_requested = false
	_model_line_of_sight_elapsed = 0.0


func is_model_controlled() -> bool:
	return ai_controller != null and ai_controller.get("heuristic") == "model"


func get_active_scripted_profile() -> String:
	if scripted_profile != "adaptive":
		return scripted_profile
	return SCRIPTED_PROFILE_ORDER[_scripted_adaptive_profile_index]


func set_scripted_enemy_trend(trend: String) -> void:
	if scripted_profile != "adaptive":
		return
	if trend == "stable":
		_scripted_last_enemy_trend = trend
		_scripted_enemy_trend_streak = 0
		return
	if trend == _scripted_last_enemy_trend:
		_scripted_enemy_trend_streak += 1
	else:
		_scripted_last_enemy_trend = trend
		_scripted_enemy_trend_streak = 1
	if _scripted_enemy_trend_streak < scripted_adaptive_profile_change_streak:
		return
	if trend == "improving":
		_scripted_adaptive_profile_index = mini(_scripted_adaptive_profile_index + 1, SCRIPTED_PROFILE_ORDER.size() - 1)
	elif trend == "struggling":
		_scripted_adaptive_profile_index = maxi(_scripted_adaptive_profile_index - 1, 0)
	_scripted_enemy_trend_streak = 0


func _apply_model_control(delta: float) -> void:
	var has_line_of_sight := has_line_of_sight_to_target()
	if has_line_of_sight:
		_model_line_of_sight_elapsed += delta
	else:
		_model_line_of_sight_elapsed = 0.0

	if model_aim_assist_enabled and has_line_of_sight:
		_face_target(delta)
	else:
		rotate_y(-_model_turn_input * model_turn_speed * delta)

	var wish_direction := (global_basis.x * _model_move_input.x) + (-global_basis.z * _model_move_input.y)
	_apply_world_movement(wish_direction, delta)
	var assisted_shot_ready := (
		model_auto_fire_enabled
		and _model_line_of_sight_elapsed >= model_auto_fire_reaction_delay_seconds
		and get_aim_alignment() >= model_auto_fire_alignment_threshold
	)
	if (_model_shoot_requested or assisted_shot_ready) and weapon.can_shoot() and has_line_of_sight:
		_shoot()
	_model_shoot_requested = false


func _apply_scripted_control(delta: float) -> void:
	var profile := _resolve_scripted_profile()
	_scripted_post_shot_timer = maxf(_scripted_post_shot_timer - delta, 0.0)
	var has_line_of_sight := has_line_of_sight_to_target()
	if has_line_of_sight:
		_model_line_of_sight_elapsed += delta
	else:
		_model_line_of_sight_elapsed = 0.0
	_face_target(delta, float(profile["turn_multiplier"]))

	var to_target := target.global_position - global_position
	to_target.y = 0.0
	var distance := to_target.length()
	var forward_direction := Vector3.ZERO
	if distance > float(profile["preferred_distance"]):
		forward_direction = to_target.normalized()
	elif distance < float(profile["stopping_distance"]):
		forward_direction = -to_target.normalized()

	_scripted_strafe_flip_timer -= delta
	if _scripted_strafe_flip_timer <= 0.0:
		_scripted_strafe_direction *= -1.0
		_scripted_strafe_flip_timer = randf_range(0.45, 1.25)
	var to_target_horizontal := to_target.normalized() if distance > 0.001 else -global_basis.z
	var strafe_direction := to_target_horizontal.cross(Vector3.UP).normalized() * _scripted_strafe_direction
	var move_direction := (
		forward_direction * float(profile["aggression"])
		+ strafe_direction * float(profile["strafe_strength"])
	)
	move_direction = _apply_scripted_obstacle_avoidance(move_direction, strafe_direction)
	_apply_world_movement(move_direction, delta)

	if (
		weapon.can_shoot()
		and _scripted_post_shot_timer <= 0.0
		and has_line_of_sight
		and _model_line_of_sight_elapsed >= float(profile["reaction_delay"])
		and get_aim_alignment() >= float(profile["shot_alignment_threshold"])
	):
		_shoot()
		_scripted_post_shot_timer = float(profile["post_shot_delay"])


func _apply_movement(delta: float) -> void:
	var input_vector := Vector2(
		Input.get_action_strength("move_right") - Input.get_action_strength("move_left"),
		Input.get_action_strength("move_forward") - Input.get_action_strength("move_back")
	)

	if input_vector.length() > 1.0:
		input_vector = input_vector.normalized()

	# In Godot, -Z is forward for a 3D node.
	var wish_direction := (global_basis.x * input_vector.x) + (-global_basis.z * input_vector.y)
	_apply_world_movement(wish_direction, delta)


func _apply_world_movement(wish_direction: Vector3, delta: float) -> void:
	velocity.x = wish_direction.x * move_speed
	velocity.z = wish_direction.z * move_speed

	if not is_on_floor():
		velocity.y -= gravity * delta
	else:
		velocity.y = 0.0

	move_and_slide()


func _face_target(delta: float, turn_multiplier := 1.0) -> void:
	var aim_point := get_target_aim_point()
	var horizontal_target := aim_point
	horizontal_target.y = global_position.y
	var horizontal_direction := horizontal_target - global_position
	if horizontal_direction.length_squared() <= 0.001:
		return
	var blend := clampf(model_turn_speed * turn_multiplier * delta, 0.0, 1.0)
	var blended := (-global_basis.z).slerp(horizontal_direction.normalized(), blend)
	look_at(global_position + blended, Vector3.UP)

	var camera_to_target := aim_point - camera.global_position
	var horizontal_distance := Vector2(camera_to_target.x, camera_to_target.z).length()
	var desired_pitch := atan2(camera_to_target.y, horizontal_distance)
	_look_pitch = lerp_angle(_look_pitch, desired_pitch, blend)
	camera.rotation.x = _look_pitch


func get_target_aim_point() -> Vector3:
	return target.global_position + Vector3.UP * model_aim_height


func has_line_of_sight_to_target() -> bool:
	var origin := camera.global_position
	var target_point := get_target_aim_point()
	var query := PhysicsRayQueryParameters3D.create(origin, target_point)
	query.exclude = [get_rid()]
	var result := get_world_3d().direct_space_state.intersect_ray(query)
	return not result.is_empty() and result.get("collider") == target


func get_aim_alignment() -> float:
	var direction := (get_target_aim_point() - camera.global_position).normalized()
	return clampf((-camera.global_basis.z).dot(direction), -1.0, 1.0)


func get_obstacle_clearance(direction: Vector3, max_distance := 4.0) -> float:
	if direction.length_squared() <= 0.001:
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
	var active_profile := get_active_scripted_profile()
	if scripted_profile != "adaptive":
		return SCRIPTED_PROFILE_SETTINGS.get(active_profile, SCRIPTED_PROFILE_SETTINGS["medium"])
	var health_advantage := health.get_health_fraction()
	if target is EnemyController:
		health_advantage -= (target as EnemyController).health.get_health_fraction()
	if health_advantage > 0.25:
		return SCRIPTED_PROFILE_SETTINGS["hard"]
	if health_advantage < -0.2:
		return SCRIPTED_PROFILE_SETTINGS["easy"]
	return SCRIPTED_PROFILE_SETTINGS.get(active_profile, SCRIPTED_PROFILE_SETTINGS["medium"])


func _rotate_view(mouse_delta: Vector2) -> void:
	rotate_y(-mouse_delta.x * mouse_sensitivity)

	_look_pitch = clamp(
		_look_pitch - mouse_delta.y * mouse_sensitivity,
		deg_to_rad(-85.0),
		deg_to_rad(85.0)
	)
	camera.rotation.x = _look_pitch


func _shoot() -> void:
	var origin := camera.global_position
	var direction := -camera.global_basis.z
	weapon.shoot(origin, direction, self)
