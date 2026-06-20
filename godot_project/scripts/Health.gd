extends Node
class_name Health

## Small, reusable health component.
##
## Keep health as its own node so player, enemy, future bots, and future
## training experiments can all use the same damage/death behavior.

signal damaged(amount: int, source_id: String, remaining_health: int)
signal died(source_id: String)
signal reset(current_health: int)

@export var max_health := 100

var current_health := max_health
var is_dead := false


func _ready() -> void:
	# Exported values are available by _ready(), so initialize here.
	current_health = max_health


func apply_damage(amount: int, source_id: String = "unknown") -> void:
	# Ignore extra damage after death. This prevents duplicate round resets.
	if is_dead:
		return

	current_health = max(current_health - amount, 0)
	damaged.emit(amount, source_id, current_health)

	if current_health == 0:
		is_dead = true
		died.emit(source_id)


func reset_health() -> void:
	# Used by the round manager when starting a fresh round.
	is_dead = false
	current_health = max_health
	reset.emit(current_health)


func get_health_fraction() -> float:
	if max_health <= 0:
		return 0.0
	return float(current_health) / float(max_health)

