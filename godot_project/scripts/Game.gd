extends Node3D
class_name Game

## Round coordinator for the MVP.
##
## This is the only script that knows about both actors. It wires gameplay logs,
## listens for death, waits briefly for human play, and resets immediately
## when the enemy is controlled by Godot RL Agents.

@export var player_path: NodePath
@export var enemy_path: NodePath
@export var gameplay_log_writer_path: NodePath
@export var player_frame_capture_path: NodePath
@export var round_reset_delay_seconds := 1.25
@export var training_mode := false
@export var behavior_sample_interval_frames := 10
@export var adaptive_skill_window_rounds := 5
@export var baseline_duration_seconds := 0.0
@export var coaching_capture_interval_seconds := 2.0

@onready var player: PlayerController = get_node(player_path)
@onready var enemy: EnemyController = get_node(enemy_path)
@onready var gameplay_logs: Node = get_node(gameplay_log_writer_path)
@onready var player_frame_capture: Node = get_node_or_null(player_frame_capture_path)

var _round_index := 0
var _round_is_resetting := false
var _round_start_time_msec := 0.0
var _baseline_elapsed_seconds := 0.0
var _current_player_hits := 0
var _current_player_misses := 0
var _current_player_damage_dealt := 0
var _current_player_damage_taken := 0
var _player_skill_history: Array[Dictionary] = []
var _reward_profile := "default"
var _agent_role := "enemy"
var _last_coaching_line_of_sight := false
var _has_coaching_line_of_sight_sample := false
var _coaching_capture_elapsed_seconds := 0.0


func _ready() -> void:
	_apply_command_line_overrides()
	_connect_actor_signals("player", player, player.health, player.weapon)
	_connect_actor_signals("enemy", enemy, enemy.health, enemy.weapon)
	enemy.shot_blocked.connect(_on_enemy_shot_blocked)
	_start_round()


func _physics_process(delta: float) -> void:
	_update_baseline_timer(delta)
	_update_periodic_coaching_capture(delta)

	if behavior_sample_interval_frames <= 0:
		return
	if _round_is_resetting or player.health.is_dead or enemy.health.is_dead:
		return
	if Engine.get_physics_frames() % behavior_sample_interval_frames != 0:
		return

	_capture_line_of_sight_transition()
	_log_enemy_behavior_sample("periodic")


func _connect_actor_signals(actor_id: String, actor: Node3D, health: Health, weapon: RaycastWeapon) -> void:
	health.damaged.connect(_on_actor_damaged.bind(actor_id, actor))
	health.died.connect(_on_actor_died.bind(actor_id, actor))
	weapon.shot_fired.connect(_on_shot_fired)
	weapon.hit_landed.connect(_on_hit_landed)
	weapon.shot_missed.connect(_on_shot_missed)


func _start_round() -> void:
	_round_is_resetting = false
	_round_index += 1
	_round_start_time_msec = Time.get_unix_time_from_system() * 1000.0
	_current_player_hits = 0
	_current_player_misses = 0
	_current_player_damage_dealt = 0
	_current_player_damage_taken = 0

	gameplay_logs.log_event("round_started", {
		"round_index": _round_index,
		"player_position": _vec3_to_dict(player.global_position),
		"enemy_position": _vec3_to_dict(enemy.global_position),
		"active_scripted_profile": enemy.get_active_scripted_profile(),
		"active_scripted_player_profile": player.get_active_scripted_profile(),
		"active_enemy_ppo_difficulty": enemy.get_active_model_difficulty_profile(),
		"reward_profile": _reward_profile,
		"agent_role": _agent_role,
		"player_model_auto_fire": player.model_auto_fire_enabled,
	})


func _on_actor_damaged(amount: int, source_id: String, remaining_health: int, actor_id: String, actor: Node3D) -> void:
	gameplay_logs.log_event("actor_damaged", {
		"round_index": _round_index,
		"actor_id": actor_id,
		"source_id": source_id,
		"amount": amount,
		"remaining_health": remaining_health,
		"position": _vec3_to_dict(actor.global_position),
	})
	if actor_id == "player" and source_id == "enemy":
		_current_player_damage_taken += amount
	elif actor_id == "enemy" and source_id == "player":
		_current_player_damage_dealt += amount
	if actor_id == "player":
		_capture_player_frame("player_damaged")
	_log_enemy_behavior_sample("actor_damaged")


func _on_actor_died(source_id: String, actor_id: String, actor: Node3D) -> void:
	if _round_is_resetting:
		return

	_round_is_resetting = true

	gameplay_logs.log_event("round_ended", {
		"round_index": _round_index,
		"dead_actor_id": actor_id,
		"killer_id": source_id,
		"death_position": _vec3_to_dict(actor.global_position),
	})
	_capture_player_frame("round_ended_%s" % actor_id)
	_log_enemy_behavior_sample("round_ended")
	_record_player_skill_round(source_id, actor_id, "")

	if training_mode or _is_enemy_training_active():
		_reset_round()
		return

	await get_tree().create_timer(round_reset_delay_seconds).timeout
	_reset_round()


func reset_round_for_training() -> void:
	# Godot RL Agents expects the environment to produce a fresh observation
	# quickly after reset. This bypasses the human-friendly visual delay.
	_round_is_resetting = true
	_reset_round()


func end_round_for_training_timeout() -> void:
	if _round_is_resetting:
		return

	_round_is_resetting = true
	gameplay_logs.log_event("round_ended", {
		"round_index": _round_index,
		"dead_actor_id": "",
		"killer_id": "",
		"end_reason": "timeout",
		"enemy_position": _vec3_to_dict(enemy.global_position),
		"player_position": _vec3_to_dict(player.global_position),
	})
	_log_enemy_behavior_sample("round_timeout")
	_record_player_skill_round("", "", "timeout")
	_reset_round()


func _reset_round() -> void:
	player.reset_for_round()
	enemy.reset_for_round()
	_start_round()


func _is_enemy_training_active() -> bool:
	return enemy.ai_controller != null and enemy.ai_controller.get("heuristic") == "model"


func _on_shot_fired(shooter_id: String, origin: Vector3, direction: Vector3) -> void:
	gameplay_logs.log_event("shot_fired", {
		"round_index": _round_index,
		"shooter_id": shooter_id,
		"origin": _vec3_to_dict(origin),
		"direction": _vec3_to_dict(direction),
	})
	if shooter_id == "enemy":
		_log_enemy_behavior_sample("shot_fired")


func _on_hit_landed(shooter_id: String, target_name: String, damage: int, hit_position: Vector3) -> void:
	gameplay_logs.log_event("shot_hit", {
		"round_index": _round_index,
		"shooter_id": shooter_id,
		"target_name": target_name,
		"damage": damage,
		"hit_position": _vec3_to_dict(hit_position),
	})
	if shooter_id == "enemy" or target_name == enemy.name:
		_log_enemy_behavior_sample("shot_hit")
	if shooter_id == "player":
		_current_player_hits += 1
		_capture_player_frame("player_hit")


func _on_shot_missed(shooter_id: String, hit_position: Vector3) -> void:
	gameplay_logs.log_event("shot_missed", {
		"round_index": _round_index,
		"shooter_id": shooter_id,
		"end_position": _vec3_to_dict(hit_position),
	})
	if shooter_id == "enemy":
		_log_enemy_behavior_sample("shot_missed")
	elif shooter_id == "player":
		_current_player_misses += 1
		_capture_player_frame("player_missed")


func _on_enemy_shot_blocked(reason: String, aim_alignment: float, line_of_sight: bool, reaction_elapsed: float) -> void:
	gameplay_logs.log_event("enemy_shot_blocked", {
		"round_index": _round_index,
		"reason": reason,
		"aim_alignment": aim_alignment,
		"line_of_sight": line_of_sight,
		"reaction_elapsed": reaction_elapsed,
		"weapon_ready": enemy.weapon.can_shoot(),
	})
	_log_enemy_behavior_sample("shot_blocked_%s" % reason)


func _log_enemy_behavior_sample(reason: String) -> void:
	if gameplay_logs == null or player == null or enemy == null:
		return

	var enemy_forward := -enemy.global_basis.z
	gameplay_logs.log_event("enemy_behavior_sample", {
		"round_index": _round_index,
		"sample_reason": reason,
		"physics_frame": Engine.get_physics_frames(),
		"enemy_position": _vec3_to_dict(enemy.global_position),
		"player_position": _vec3_to_dict(player.global_position),
		"enemy_forward": _vec3_to_dict(enemy_forward),
		"enemy_velocity": _vec3_to_dict(enemy.velocity),
		"distance_to_player": enemy.get_distance_to_target(),
		"line_of_sight": enemy.has_line_of_sight_to_target(),
		"line_of_sight_elapsed": enemy.get_line_of_sight_elapsed(),
		"aim_alignment": enemy.get_aim_alignment(),
		"weapon_ready": enemy.weapon.can_shoot(),
		"enemy_health_fraction": enemy.health.get_health_fraction(),
		"player_health_fraction": player.health.get_health_fraction(),
		"is_model_controlled": enemy.is_model_controlled(),
		"scripted_profile": enemy.scripted_profile,
		"active_scripted_profile": enemy.get_active_scripted_profile(),
		"scripted_player_profile": player.scripted_profile,
		"active_scripted_player_profile": player.get_active_scripted_profile(),
		"enemy_ppo_difficulty": enemy.get_active_model_difficulty_profile(),
		"enemy_ppo_difficulty_config": enemy.get_model_difficulty_config(),
	})


func _capture_player_frame(reason: String) -> void:
	gameplay_logs.log_event("macro_state", _build_macro_state(reason))
	if player_frame_capture != null:
		player_frame_capture.request_capture(reason, _round_index)


func _build_macro_state(reason: String) -> Dictionary:
	var player_health_fraction := player.health.get_health_fraction()
	var enemy_health_fraction := enemy.health.get_health_fraction()
	var line_of_sight := enemy.has_line_of_sight_to_target()
	var nearest_cover_distance := _nearest_cover_distance_to_player()
	return {
		"round_index": _round_index,
		"reason": reason,
		"player_health_fraction": player_health_fraction,
		"enemy_health_fraction": enemy_health_fraction,
		"player_health_advantage": player_health_fraction - enemy_health_fraction,
		"distance_to_enemy": player.global_position.distance_to(enemy.global_position),
		"line_of_sight": line_of_sight,
		"nearest_cover_distance": nearest_cover_distance,
		"player_near_cover": nearest_cover_distance <= 4.5,
		"player_in_danger": line_of_sight and player_health_fraction <= 0.35,
		"enemy_is_low": enemy_health_fraction <= 0.35,
	}


func _capture_line_of_sight_transition() -> void:
	var line_of_sight := enemy.has_line_of_sight_to_target()
	if not _has_coaching_line_of_sight_sample:
		_last_coaching_line_of_sight = line_of_sight
		_has_coaching_line_of_sight_sample = true
		return
	if line_of_sight == _last_coaching_line_of_sight:
		return
	_last_coaching_line_of_sight = line_of_sight
	_capture_player_frame("line_of_sight_gained" if line_of_sight else "line_of_sight_lost")


func _update_periodic_coaching_capture(delta: float) -> void:
	if coaching_capture_interval_seconds <= 0.0 or _round_is_resetting:
		return
	_coaching_capture_elapsed_seconds += delta
	if _coaching_capture_elapsed_seconds < coaching_capture_interval_seconds:
		return
	_coaching_capture_elapsed_seconds = 0.0
	_capture_player_frame("tactical_interval")


func _nearest_cover_distance_to_player() -> float:
	var nearest := INF
	for cover in get_tree().get_nodes_in_group("cover"):
		if cover is Node3D:
			nearest = minf(nearest, player.global_position.distance_to((cover as Node3D).global_position))
	return nearest if nearest < INF else 999.0


func _record_player_skill_round(killer_id: String, dead_actor_id: String, end_reason: String) -> void:
	var round_duration_sec: float = maxf(0.0, ((Time.get_unix_time_from_system() * 1000.0) - _round_start_time_msec) / 1000.0)
	var entry := {
		"round_index": _round_index,
		"hits": _current_player_hits,
		"misses": _current_player_misses,
		"damage_dealt": _current_player_damage_dealt,
		"damage_taken": _current_player_damage_taken,
		"kills": 1 if killer_id == "player" else 0,
		"deaths": 1 if dead_actor_id == "player" else 0,
		"duration_sec": round_duration_sec,
		"end_reason": end_reason,
	}
	_player_skill_history.append(entry)
	while _player_skill_history.size() > adaptive_skill_window_rounds:
		_player_skill_history.pop_front()

	var summary := _build_player_skill_summary()
	var previous_ppo_difficulty := enemy.get_active_model_difficulty_profile()
	enemy.set_adaptive_skill_trend(str(summary["recent_trend"]))
	var enemy_trend := _invert_skill_trend(str(summary["recent_trend"]))
	player.set_scripted_enemy_trend(enemy_trend)
	summary["active_enemy_ppo_difficulty"] = enemy.get_active_model_difficulty_profile()
	summary["active_scripted_player_profile"] = player.get_active_scripted_profile()
	gameplay_logs.log_event("player_skill_updated", summary)
	if previous_ppo_difficulty != enemy.get_active_model_difficulty_profile():
		gameplay_logs.log_event("enemy_ppo_difficulty_changed", {
			"round_index": _round_index,
			"previous_profile": previous_ppo_difficulty,
			"active_profile": enemy.get_active_model_difficulty_profile(),
			"player_trend": summary["recent_trend"],
			"difficulty_config": enemy.get_model_difficulty_config(),
		})


func _build_player_skill_summary() -> Dictionary:
	var hits := 0
	var misses := 0
	var damage_dealt := 0
	var damage_taken := 0
	var kills := 0
	var deaths := 0
	var survival_time_total := 0.0

	for entry in _player_skill_history:
		hits += int(entry["hits"])
		misses += int(entry["misses"])
		damage_dealt += int(entry["damage_dealt"])
		damage_taken += int(entry["damage_taken"])
		kills += int(entry["kills"])
		deaths += int(entry["deaths"])
		survival_time_total += float(entry["duration_sec"])

	var total_shots := hits + misses
	var hit_rate := float(hits) / float(total_shots) if total_shots > 0 else 0.0
	var damage_diff := damage_dealt - damage_taken
	var survival_mean := survival_time_total / float(_player_skill_history.size()) if _player_skill_history.size() > 0 else 0.0
	var trend := _classify_player_skill_trend(hit_rate, damage_diff, kills, deaths)
	return {
		"round_index": _round_index,
		"window_rounds": _player_skill_history.size(),
		"hit_rate": hit_rate,
		"damage_dealt": damage_dealt,
		"damage_taken": damage_taken,
		"damage_diff": damage_diff,
		"kills": kills,
		"deaths": deaths,
		"survival_time_mean_sec": survival_mean,
		"recent_trend": trend,
		"active_scripted_profile": enemy.get_active_scripted_profile(),
		"active_scripted_player_profile": player.get_active_scripted_profile(),
		"active_enemy_ppo_difficulty": enemy.get_active_model_difficulty_profile(),
	}


func _classify_player_skill_trend(hit_rate: float, damage_diff: int, kills: int, deaths: int) -> String:
	if kills > deaths or (damage_diff > 50 and hit_rate >= 0.25):
		return "improving"
	if deaths > kills or (damage_diff < -50 and hit_rate < 0.2):
		return "struggling"
	return "stable"


func _invert_skill_trend(trend: String) -> String:
	if trend == "improving":
		return "struggling"
	if trend == "struggling":
		return "improving"
	return "stable"


func _apply_command_line_overrides() -> void:
	var args := _get_all_command_line_args()
	var index := 0
	while index < args.size():
		var key := str(args[index])
		var value := ""
		var step := 1
		if key.find("=") > -1:
			var key_value := key.split("=", false, 1)
			key = key_value[0]
			value = key_value[1] if key_value.size() > 1 else ""
		else:
			value = str(args[index + 1]) if index + 1 < args.size() else ""
			step = 2
		if key == "--baseline-profile" and value in ["easy", "medium", "hard", "adaptive"]:
			enemy.scripted_profile = value
			training_mode = true
			index += step
		elif key == "--scripted-player-profile" and value in ["human", "easy", "medium", "hard", "adaptive"]:
			player.scripted_profile = value
			training_mode = true
			index += step
		elif key == "--ppo-difficulty-profile" and value in ["easy", "medium", "hard", "adaptive"]:
			enemy.set_model_difficulty_profile(value)
			index += step
		elif key == "--baseline-duration-seconds" and value.is_valid_float():
			baseline_duration_seconds = maxf(float(value), 0.0)
			training_mode = true
			index += step
		elif key == "--reward-profile" and value in ["default", "damage", "timeout", "timeout_aim_assist", "volume_combat", "timeout_centered", "timeout_v2", "pressure", "range_los", "combined"]:
			_apply_reward_profile(value)
			index += step
		elif key == "--agent-role" and value in ["enemy", "player"]:
			_configure_agent_role(value)
			index += step
		else:
			index += 1


func _get_all_command_line_args() -> Array[String]:
	var all_args: Array[String] = []
	for argument in OS.get_cmdline_args():
		all_args.append(str(argument))
	for argument in OS.get_cmdline_user_args():
		all_args.append(str(argument))
	return all_args


func _apply_reward_profile(profile: String) -> void:
	_reward_profile = profile
	if enemy.ai_controller != null and enemy.ai_controller.has_method("apply_reward_profile"):
		enemy.ai_controller.apply_reward_profile(profile)
	if gameplay_logs != null:
		var reward_config: Dictionary = {}
		if enemy.ai_controller != null and enemy.ai_controller.has_method("get_reward_config"):
			reward_config = enemy.ai_controller.get_reward_config()
		gameplay_logs.log_event("reward_profile_configured", {
			"reward_profile": profile,
			"reward_config": reward_config,
		})


func _configure_agent_role(role: String) -> void:
	_agent_role = role
	if player.ai_controller == null or enemy.ai_controller == null:
		return
	if role == "player":
		player.ai_controller.control_mode = player.ai_controller.ControlModes.TRAINING
		enemy.ai_controller.control_mode = enemy.ai_controller.ControlModes.HUMAN
		training_mode = true
	else:
		player.ai_controller.control_mode = player.ai_controller.ControlModes.HUMAN
		enemy.ai_controller.control_mode = enemy.ai_controller.ControlModes.TRAINING
		training_mode = true


func _update_baseline_timer(delta: float) -> void:
	if baseline_duration_seconds <= 0.0:
		return
	_baseline_elapsed_seconds += delta
	if _baseline_elapsed_seconds < baseline_duration_seconds:
		return
	gameplay_logs.log_event("baseline_finished", {
		"duration_seconds": baseline_duration_seconds,
		"scripted_profile": enemy.scripted_profile,
		"active_scripted_profile": enemy.get_active_scripted_profile(),
	})
	get_tree().quit()


func _vec3_to_dict(value: Vector3) -> Dictionary:
	return {
		"x": value.x,
		"y": value.y,
		"z": value.z,
	}
