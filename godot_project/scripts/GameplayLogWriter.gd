extends Node
class_name GameplayLogWriter

## JSONL gameplay-log writer.
##
## Every event is one JSON object on one line. This is intentionally boring:
## JSONL is easy to append, inspect, stream, and parse later from Python.

@export var log_directory := "user://gameplay_logs"

var _file: FileAccess
var _run_id := ""
var _event_index := 0


func _ready() -> void:
	_apply_command_line_overrides()
	start_new_run()


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
		if key == "--gameplay-log-dir" and not value.is_empty():
			log_directory = value
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


func start_new_run() -> void:
	var global_log_directory := ProjectSettings.globalize_path(log_directory)
	DirAccess.make_dir_recursive_absolute(global_log_directory)

	_run_id = _build_run_id()
	var path := "%s/run_%s.jsonl" % [global_log_directory, _run_id]
	_file = FileAccess.open(path, FileAccess.WRITE)

	if _file == null:
		push_error("Could not open gameplay log at %s" % path)
		return

	log_event("run_started", {"path": path})


func _build_run_id() -> String:
	var parts := [
		Time.get_datetime_string_from_system(false, true).replace(":", "-"),
		str(Time.get_ticks_usec()),
	]
	var args := OS.get_cmdline_args()
	for argument in args:
		var text := str(argument)
		if text.begins_with("--port="):
			parts.append("port-%s" % text.trim_prefix("--port="))
		elif text.begins_with("--env_seed="):
			parts.append("seed-%s" % text.trim_prefix("--env_seed="))
	return "_".join(parts)


func log_event(event_type: String, payload: Dictionary = {}) -> void:
	if _file == null:
		return

	var event := {
		"event_index": _event_index,
		"event_type": event_type,
		"run_id": _run_id,
		"unix_time_msec": Time.get_unix_time_from_system() * 1000.0,
		"engine_frame": Engine.get_physics_frames(),
		"payload": payload,
	}

	_file.store_line(JSON.stringify(event))
	_file.flush()
	_event_index += 1


func get_run_id() -> String:
	return _run_id


func _exit_tree() -> void:
	if _file != null:
		log_event("run_finished")
		_file.close()
