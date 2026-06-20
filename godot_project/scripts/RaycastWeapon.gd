extends Node3D
class_name RaycastWeapon

## Simple hitscan weapon.
##
## This script does not spawn bullets. It casts one ray from an origin in a
## direction, applies damage to a hit Health component, and emits gameplay-log
## friendly signals that the Game node can log.

signal shot_fired(shooter_id: String, origin: Vector3, direction: Vector3)
signal hit_landed(shooter_id: String, target_name: String, damage: int, hit_position: Vector3)
signal shot_missed(shooter_id: String, hit_position: Vector3)

@export var owner_id := "unknown"
@export var damage := 25
@export var max_range := 40.0
@export var cooldown_seconds := 0.25
@export var muzzle_path: NodePath = NodePath("Muzzle")
@export var muzzle_flash_path: NodePath = NodePath("MuzzleFlash")
@export var tracer_color := Color(1.0, 0.9, 0.25, 1.0)
@export var tracer_lifetime_seconds := 0.06

@onready var muzzle: Node3D = get_node_or_null(muzzle_path)
@onready var muzzle_flash: Node3D = get_node_or_null(muzzle_flash_path)

var _cooldown_remaining := 0.0
var _flash_remaining := 0.0


func _physics_process(delta: float) -> void:
	# Cooldowns are measured in simulation time, not wall-clock time.
	_cooldown_remaining = max(_cooldown_remaining - delta, 0.0)
	_flash_remaining = max(_flash_remaining - delta, 0.0)

	if muzzle_flash != null:
		muzzle_flash.visible = _flash_remaining > 0.0


func can_shoot() -> bool:
	return _cooldown_remaining <= 0.0


func get_cooldown_fraction() -> float:
	# Observation helper: 0 means ready, 1 means the whole cooldown remains.
	if cooldown_seconds <= 0.0:
		return 0.0
	return clamp(_cooldown_remaining / cooldown_seconds, 0.0, 1.0)


func reset_weapon() -> void:
	# Episode resets should not inherit a cooldown from the previous round.
	_cooldown_remaining = 0.0


func shoot(origin: Vector3, direction: Vector3, excluded_body: CollisionObject3D = null) -> bool:
	if not can_shoot():
		return false

	_cooldown_remaining = cooldown_seconds

	var normalized_direction := direction.normalized()
	shot_fired.emit(owner_id, origin, normalized_direction)

	var query := PhysicsRayQueryParameters3D.create(
		origin,
		origin + normalized_direction * max_range
	)

	# Do not let a character shoot itself.
	if excluded_body != null:
		query.exclude = [excluded_body.get_rid()]

	var result := get_world_3d().direct_space_state.intersect_ray(query)
	if result.is_empty():
		var end_position := origin + normalized_direction * max_range
		_show_shot_visual(end_position)
		shot_missed.emit(owner_id, end_position)
		return true

	var collider: Node = result.get("collider") as Node
	var hit_position: Vector3 = result.get("position", origin + normalized_direction * max_range)
	_show_shot_visual(hit_position)
	var health := _find_health(collider)

	if health != null:
		health.apply_damage(damage, owner_id)
		hit_landed.emit(owner_id, collider.name, damage, hit_position)
	else:
		shot_missed.emit(owner_id, hit_position)

	return true


func _show_shot_visual(end_position: Vector3) -> void:
	# Visual-only debugging. Damage, gameplay logs, and RL rewards still use the
	# raycast above; this just makes the shot direction visible for a moment.
	var start_position := global_position
	if muzzle != null:
		start_position = muzzle.global_position

	_flash_remaining = 0.05
	if muzzle_flash != null:
		muzzle_flash.visible = true

	_spawn_tracer(start_position, end_position)


func _spawn_tracer(start_position: Vector3, end_position: Vector3) -> void:
	var mesh := ImmediateMesh.new()
	var material := StandardMaterial3D.new()
	material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	material.albedo_color = tracer_color
	material.emission_enabled = true
	material.emission = tracer_color

	mesh.surface_begin(Mesh.PRIMITIVE_LINES, material)
	mesh.surface_add_vertex(start_position)
	mesh.surface_add_vertex(end_position)
	mesh.surface_end()

	var tracer := MeshInstance3D.new()
	tracer.name = "%sTracer" % owner_id.capitalize()
	tracer.mesh = mesh

	var root := get_tree().current_scene
	if root == null:
		root = get_tree().root
	root.add_child(tracer)

	var timer := get_tree().create_timer(tracer_lifetime_seconds)
	timer.timeout.connect(Callable(tracer, "queue_free"))


func _find_health(node: Node) -> Health:
	# The current MVP stores Health as a direct child named "Health".
	# Walking up one level keeps the weapon tolerant of hitting a child collider.
	var cursor := node
	while cursor != null:
		var health := cursor.get_node_or_null("Health")
		if health is Health:
			return health
		cursor = cursor.get_parent()
	return null
