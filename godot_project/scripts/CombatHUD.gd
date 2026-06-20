extends CanvasLayer
class_name CombatHUD

@export var player_health_path: NodePath
@export var enemy_health_path: NodePath
@export var player_fill_path: NodePath
@export var enemy_fill_path: NodePath
@export var player_label_path: NodePath
@export var enemy_label_path: NodePath
@export var bar_width := 260.0

@onready var player_health: Health = get_node(player_health_path)
@onready var enemy_health: Health = get_node(enemy_health_path)
@onready var player_fill: ColorRect = get_node(player_fill_path)
@onready var enemy_fill: ColorRect = get_node(enemy_fill_path)
@onready var player_label: Label = get_node(player_label_path)
@onready var enemy_label: Label = get_node(enemy_label_path)


func _ready() -> void:
	player_health.damaged.connect(_on_health_changed.bind("player"))
	player_health.reset.connect(_on_health_reset.bind("player"))
	enemy_health.damaged.connect(_on_health_changed.bind("enemy"))
	enemy_health.reset.connect(_on_health_reset.bind("enemy"))
	_refresh()


func _on_health_changed(_amount: int, _source_id: String, _remaining_health: int, _actor_id: String) -> void:
	_refresh()


func _on_health_reset(_current_health: int, _actor_id: String) -> void:
	_refresh()


func _refresh() -> void:
	_update_actor(player_health, player_fill, player_label, "PLAYER")
	_update_actor(enemy_health, enemy_fill, enemy_label, "ENEMY")


func _update_actor(health: Health, fill: ColorRect, label: Label, actor_name: String) -> void:
	var fraction := health.get_health_fraction()
	fill.size.x = bar_width * fraction
	label.text = "%s  %d / %d" % [actor_name, health.current_health, health.max_health]
