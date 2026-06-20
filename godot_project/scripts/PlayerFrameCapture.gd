extends Node
class_name PlayerFrameCapture

@export var gameplay_log_writer_path: NodePath
@export var capture_directory := "user://coaching_frames"
@export var enabled := true
@export var min_capture_interval_seconds := 0.25

@onready var gameplay_logs: GameplayLogWriter = get_node(gameplay_log_writer_path)

var _last_capture_time_msec := -1000000.0
var _capture_index := 0


func _ready() -> void:
	_apply_command_line_overrides()


func request_capture(reason: String, round_index: int) -> void:
	if not enabled:
		return
	var now_msec := Time.get_unix_time_from_system() * 1000.0
	if now_msec - _last_capture_time_msec < min_capture_interval_seconds * 1000.0:
		return
	_last_capture_time_msec = now_msec
	call_deferred("_capture_after_draw", reason, round_index)


func _capture_after_draw(reason: String, round_index: int) -> void:
	await RenderingServer.frame_post_draw
	var image := get_viewport().get_texture().get_image()
	if image == null or image.is_empty():
		return

	var global_directory := ProjectSettings.globalize_path(capture_directory)
	DirAccess.make_dir_recursive_absolute(global_directory)
	var file_name := "frame_%s_round-%d_%06d_%s.png" % [
		gameplay_logs.get_run_id(),
		round_index,
		_capture_index,
		_sanitize(reason),
	]
	_capture_index += 1
	var path := "%s/%s" % [global_directory, file_name]
	if image.save_png(path) != OK:
		push_warning("PlayerFrameCapture: Could not save frame at %s" % path)
		return

	gameplay_logs.log_event("player_frame_captured", {
		"round_index": round_index,
		"reason": reason,
		"frame_path": path,
		"width": image.get_width(),
		"height": image.get_height(),
	})


func _apply_command_line_overrides() -> void:
	for argument in _get_all_command_line_args():
		var text := str(argument)
		if text.begins_with("--coaching-frame-dir="):
			capture_directory = text.trim_prefix("--coaching-frame-dir=")
		elif text.begins_with("--disable-coaching-frame-capture"):
			enabled = text.get_slice("=", 1).to_lower() == "false" if "=" in text else false


func _get_all_command_line_args() -> Array[String]:
	var all_args: Array[String] = []
	for argument in OS.get_cmdline_args():
		all_args.append(str(argument))
	for argument in OS.get_cmdline_user_args():
		all_args.append(str(argument))
	return all_args


func _sanitize(value: String) -> String:
	return value.to_lower().replace(" ", "-").replace("/", "-").replace(":", "-")
