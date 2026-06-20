extends AIController3D
class_name PlayerAIController3D

@export var player_path: NodePath = NodePath("..")
@export var target_path: NodePath = NodePath("../../Enemy")
@export var game_path: NodePath = NodePath("../..")
@export var max_episode_steps := 900

@onready var player: PlayerController = get_node(player_path)
@onready var target: EnemyController = get_node(target_path)
@onready var game: Game = get_node(game_path)

var player_health: Health
var target_health: Health
var player_weapon: RaycastWeapon


func _ready() -> void:
	super._ready()
	player_health = player.get_node(player.health_path) as Health
	target_health = target.get_node(target.health_path) as Health
	player_weapon = player.get_node(player.weapon_path) as RaycastWeapon
	target_health.damaged.connect(_on_target_damaged)
	target_health.died.connect(_on_target_died)
	player_health.damaged.connect(_on_player_damaged)
	player_health.died.connect(_on_player_died)
	player_weapon.hit_landed.connect(_on_weapon_hit)
	player_weapon.shot_missed.connect(_on_weapon_missed)


func _physics_process(_delta: float) -> void:
	n_steps += 1
	if heuristic == "model" and not done:
		reward -= 0.001
		if player.has_line_of_sight_to_target():
			reward += 0.0005
		if player_health.get_health_fraction() <= 0.35 and player.has_line_of_sight_to_target():
			reward -= 0.001
	if heuristic == "model" and n_steps >= max_episode_steps and not done:
		reward -= 2.0
		done = true
		needs_reset = true
	if needs_reset:
		needs_reset = false
		n_steps = 0
		player.clear_control_input()
		game.end_round_for_training_timeout()


func get_obs() -> Dictionary:
	var relative := target.global_position - player.global_position
	var local_direction := player.global_basis.inverse() * relative.normalized()
	var forward := -player.global_basis.z
	return {"obs": [
		player.global_position.x / 12.0,
		player.global_position.z / 12.0,
		relative.x / 24.0,
		relative.z / 24.0,
		player.global_position.distance_to(target.global_position) / 33.95,
		local_direction.x,
		local_direction.z,
		forward.x,
		forward.z,
		player.velocity.x / player.move_speed,
		player.velocity.z / player.move_speed,
		player_health.get_health_fraction(),
		target_health.get_health_fraction(),
		1.0 if player_weapon.can_shoot() else 0.0,
		1.0 if player.has_line_of_sight_to_target() else 0.0,
		player.get_aim_alignment(),
		player.get_obstacle_clearance(forward),
		player.get_obstacle_clearance(-player.global_basis.x),
		player.get_obstacle_clearance(player.global_basis.x),
	]}


func get_reward() -> float:
	return reward


func get_action_space() -> Dictionary:
	return {
		"move": {"size": 2, "action_type": "continuous"},
		"turn": {"size": 1, "action_type": "continuous"},
		"shoot": {"size": 1, "action_type": "continuous"},
	}


func set_action(action) -> void:
	player.set_control_input(
		Vector2(action["move"][0], action["move"][1]),
		float(action["turn"][0]),
		float(action["shoot"][0]) > 0.5
	)


func reset() -> void:
	n_steps = 0
	needs_reset = false
	reward = 0.0
	done = false
	player.clear_control_input()


func on_round_reset() -> void:
	n_steps = 0
	needs_reset = false
	player.clear_control_input()


func _on_weapon_hit(shooter_id: String, _target_name: String, _damage: int, _hit_position: Vector3) -> void:
	if heuristic == "model" and shooter_id == "player":
		reward += 1.0


func _on_weapon_missed(shooter_id: String, _hit_position: Vector3) -> void:
	if heuristic == "model" and shooter_id == "player":
		reward -= 0.05


func _on_target_damaged(amount: int, source_id: String, _remaining_health: int) -> void:
	if heuristic == "model" and source_id == "player":
		reward += float(amount) / 50.0


func _on_player_damaged(amount: int, source_id: String, _remaining_health: int) -> void:
	if heuristic == "model" and source_id == "enemy":
		reward -= float(amount) / 50.0


func _on_target_died(source_id: String) -> void:
	if heuristic == "model" and source_id == "player" and not done:
		reward += 10.0
		done = true


func _on_player_died(source_id: String) -> void:
	if heuristic == "model" and source_id == "enemy" and not done:
		reward -= 10.0
		done = true
