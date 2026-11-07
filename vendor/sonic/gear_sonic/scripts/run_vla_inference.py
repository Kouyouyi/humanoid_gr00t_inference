"""
VLA inference runner — NO ROS 2 DEPENDENCY.

Runs an Isaac-GR00T VLA policy against the Sonic whole-body control stack.
All communication uses ZMQ:
  1. Robot state  -> ZMQ SUB on ``g1_debug`` topic (from C++ zmq_output_handler)
  2. Actions out  -> ZMQ PUB (latent protocol v4: motion token + hand joints)
  3. Camera       -> ZMQ/TCP via ComposedCameraClientSensor
  4. Keyboard     -> ZMQ SUB via ZMQKeyboardSubscriber

Uses the Isaac-GR00T PolicyClient (ZMQ REQ/REP) to communicate with a
running PolicyServer.

Keyboard commands (received via ZMQ from the standalone keyboard publisher):
  p  -> pause / resume the policy loop
  k  -> start / stop the C++ control loop
  i  -> smoothly blend from current controller latent to initial pose, including first use
  t  -> change prompt at runtime (publisher sends ``prompt:<text>``)
  [  -> toggle left hand open/closed for initial pose
  ]  -> toggle right hand open/closed for initial pose
  c  -> start recording (handled by data exporter if running)
  s  -> stop recording success (handled by data exporter)
  f  -> stop recording failure (handled by data exporter)
"""

from dataclasses import dataclass
import queue
import threading
import time
import uuid
from gear_sonic.utils.startup_trace import from_env, request_context

import numpy as np
import tyro
import zmq

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.utils.data_collection.keyboard_subscriber import (
    DEFAULT_ZMQ_KEYBOARD_PORT,
    ZMQKeyboardSubscriber,
)
from gear_sonic.utils.data_collection.telemetry import Telemetry
from gear_sonic.utils.data_collection.transforms import compute_projected_gravity
from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
from gear_sonic.utils.inference.initial_pose_transition import InitialPoseTransition
from gear_sonic.utils.inference.initial_poses import LATENT_INITIAL_MOTION_TOKEN
from gear_sonic.utils.inference.vla_utils import (
    calculate_latency_compensated_index,
    concat_action,
    prepare_observation_for_eval,
    should_trigger_new_inference,
)
from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import (
    G1GripperInverseKinematicsSolver,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
)


_STARTUP_TRACE = None


@dataclass
class InferenceConfig:
    """CLI config for the VLA inference runner."""

    # Policy server (Isaac-GR00T PolicyServer)
    host: str = "localhost"
    """The host address of the Isaac-GR00T PolicyServer."""

    port: int = 5550
    """The port of the Isaac-GR00T PolicyServer."""

    # Control
    action_publish_rate: int = 50
    """Rate at which individual actions are published to the C++ control loop (Hz)."""

    action_horizon: int = 40
    """Action horizon of the VLA policy (number of future actions per inference)."""

    rate: float = 1 / 0.4
    """Rate at which we run the forward pass of the VLA policy (Hz)."""

    # Camera
    camera_host: str = "localhost"
    """Camera server host."""

    camera_port: int = 5555
    """Camera server port."""

    # ZMQ: Robot state (from C++ zmq_output_handler, g1_debug topic)
    state_zmq_host: str = "localhost"
    """ZMQ host for robot state (g1_debug topic from C++ deploy)."""

    state_zmq_port: int = 5557
    """ZMQ port for robot state (same socket as robot_config topic)."""

    # ZMQ: Action output (latent actions to C++ control loop)
    action_zmq_host: str = "localhost"
    """ZMQ host for action output (PUB socket)."""

    action_zmq_port: int = 5556
    """ZMQ port for action output."""

    # ZMQ: Keyboard input
    keyboard_zmq_host: str = "localhost"
    """ZMQ host for keyboard input."""

    keyboard_zmq_port: int = DEFAULT_ZMQ_KEYBOARD_PORT
    """ZMQ port for keyboard input."""

    # Embodiment
    embodiment_tag: str = "unitree_g1_sonic"
    """Embodiment tag for policy inference."""

    # Prompt / eval
    prompt: str = "demo"
    """The language prompt for the VLA policy."""

    # Initial pose
    initial_pose_blend_duration: float = 4.0
    """Positive duration in seconds for smoothstep initial-pose interpolation,
    including the FIRST 'i', from the controller's current token_state.
    The policy stays paused until completion; no instant-pose fallback."""

    # Debug
    verbose_timing: bool = False
    """Whether to always print timing info (not just when loop is slow)."""


def print_green(x):
    print(f"\033[92m{x}\033[0m")


# ---------------------------------------------------------------------------
# Action packing (latent protocol v4)
# ---------------------------------------------------------------------------


def pack_latent_action_message(
    motion_token: np.ndarray,
    frame_index: np.ndarray,
    left_hand_joints: np.ndarray = None,
    right_hand_joints: np.ndarray = None,
) -> bytes:
    """Pack a single motion-token action into a ZMQ message (Protocol v4).

    Args:
        motion_token: Shape ``[64]`` (flat) or ``[1, 64]``.
        frame_index:  Shape ``[1]``.
        left_hand_joints:  Shape ``[7]`` or ``[1, 7]``, optional.
        right_hand_joints: Shape ``[7]`` or ``[1, 7]``, optional.

    Returns:
        Packed ZMQ message bytes.
    """
    motion_token = np.asarray(motion_token, dtype=np.float32)
    frame_index = np.asarray(frame_index, dtype=np.int64)

    if frame_index.ndim == 0:
        frame_index = np.array([frame_index], dtype=np.int64)
    elif frame_index.shape[0] != 1:
        frame_index = frame_index[:1]

    if motion_token.ndim == 1:
        motion_token = motion_token.reshape(1, -1)

    pose_data = {
        "token_state": motion_token,
        "frame_index": frame_index,
    }

    if left_hand_joints is not None:
        left_hand_joints = np.asarray(left_hand_joints, dtype=np.float32)
        if left_hand_joints.ndim == 1:
            if left_hand_joints.shape[0] != 7:
                raise ValueError(
                    f"left_hand_joints must have shape [7], got {left_hand_joints.shape}"
                )
            left_hand_joints = left_hand_joints.reshape(1, 7)
        pose_data["left_hand_joints"] = left_hand_joints

    if right_hand_joints is not None:
        right_hand_joints = np.asarray(right_hand_joints, dtype=np.float32)
        if right_hand_joints.ndim == 1:
            if right_hand_joints.shape[0] != 7:
                raise ValueError(
                    f"right_hand_joints must have shape [7], got {right_hand_joints.shape}"
                )
            right_hand_joints = right_hand_joints.reshape(1, 7)
        pose_data["right_hand_joints"] = right_hand_joints

    return pack_pose_message(pose_data, topic="pose", version=4)


def get_action_field(action_dict: dict, key: str):
    """Get action field from dict, checking both with and without 'action.' prefix."""
    value = action_dict.get(key)
    if value is not None:
        return value
    value = action_dict.get(f"action.{key}")
    if value is not None:
        return value
    raise AssertionError(
        f"Required action field '{key}' (or 'action.{key}') not found in processed_action. "
        f"Available keys: {list(action_dict.keys())}"
    )


# ---------------------------------------------------------------------------
# Observation / inference helpers
# ---------------------------------------------------------------------------


def prepare_observation_from_sensors(
    camera_subscriber,
    state_subscriber,
    robot_model,
    language_prompt: str,
    log_errors: bool = False,
):
    """Read sensors and prepare observation for the VLA policy.

    Returns:
        observation dict, or None if sensor data not yet available.
    """
    camera_msg = camera_subscriber.read()
    if camera_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for camera msg..", flush=True)
        return None

    state_msg = state_subscriber.get_msg()
    if state_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for state msg..", flush=True)
        return None

    if _STARTUP_TRACE is not None:
        _STARTUP_TRACE.record('input_snapshot', request_id=request_context.get(), raw_state=state_msg, camera=camera_msg, camera_local_arrival_ns=camera_msg.get("diagnostic_receive_monotonic_ns"), retrieval_ns=time.monotonic_ns(), camera_clock_domain='camera:unknown')
    cam_img = camera_msg["images"]["ego_view"]

    # Copy index finger data to middle finger (hardware coupling)
    state_msg["left_hand_q"][5] = state_msg["left_hand_q"][3]
    state_msg["left_hand_q"][6] = state_msg["left_hand_q"][4]

    qpos = robot_model.get_configuration_from_actuated_joints(
        body_actuated_joint_values=state_msg["body_q"],
        left_hand_actuated_joint_values=state_msg["left_hand_q"],
        right_hand_actuated_joint_values=state_msg["right_hand_q"],
    )

    video = {"ego_view": cam_img[np.newaxis, np.newaxis]}
    if "left_wrist" in camera_msg["images"]:
        video["left_wrist"] = camera_msg["images"]["left_wrist"][np.newaxis, np.newaxis]
    if "right_wrist" in camera_msg["images"]:
        video["wrist_view"] = camera_msg["images"]["right_wrist"][np.newaxis, np.newaxis]

    observation = {
        "video": video,
        "state": {},
        "language": {
            "annotation.human.task_description": [[language_prompt]],
        },
        "q": np.asarray(qpos, dtype=np.float32)[np.newaxis, np.newaxis],
        "timestamps": camera_msg["timestamps"]["ego_view"],
    }

    observation = prepare_observation_for_eval(robot_model, observation)

    # Projected gravity for Sonic latent embodiment
    assert "base_quat" in state_msg, "base_quat not found in state_msg"
    base_quat = np.asarray(state_msg["base_quat"], dtype=np.float64)
    assert base_quat.shape == (4,), "base_quat must have shape (4,)"
    projected_gravity = compute_projected_gravity(base_quat)
    observation["state"]["projected_gravity"] = np.asarray(
        projected_gravity, dtype=np.float32
    )[np.newaxis, np.newaxis]

    return observation


def run_policy_inference_and_process(policy, observation, robot_model):
    """Run policy inference via Isaac-GR00T PolicyClient and process results.

    Returns:
        processed_action dict or None on error.
    """
    try:
        action, _info = policy.get_action(observation)

        action.pop("task_progress", None)
        action.pop("action.task_progress", None)

        motion_key = "motion_token" if "motion_token" in action else "action.motion_token"
        if np.abs(action[motion_key]).max() > 1.25:
            print(
                f"[Warning] action['{motion_key}'] max "
                f"({np.abs(action[motion_key]).max():.4f}) > 1.25. "
                "Exceeds action bound, skipping."
            )
            return None

        processed_action = concat_action(robot_model, action)
        return processed_action
    except Exception as e:
        print(f"Error in inference: {e}")
        import traceback

        traceback.print_exc()
        return None


def _inference_worker_loop(
    inference_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    busy_event: threading.Event,
    prepare_obs_fn,
    inference_fn,
):
    """Persistent worker thread for async inference."""
    while not stop_event.is_set():
        try:
            try:
                inference_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            busy_event.set()
            try:
                request_id = uuid.uuid4().hex
                request_context.set(request_id if _STARTUP_TRACE.enabled else None)
                inference_start_time = time.monotonic()
                observation = prepare_obs_fn()
                if observation is None:
                    print("[DEBUG] Worker thread: Observation is None, skipping", flush=True)
                    continue

                _STARTUP_TRACE.record('request_start', request_id=request_id, observation=observation, request_start=inference_start_time)
                diagnostic_context = request_context.set(request_id if _STARTUP_TRACE.enabled else None)
                try:
                    processed_action = inference_fn(observation)
                finally:
                    request_context.reset(diagnostic_context)
                _STARTUP_TRACE.record('model_response', request_id=request_id, chunk=processed_action)

                # None is a failure notification: discard cached actions and
                # pause, while this worker keeps retrying the policy service.
                try:
                    result_queue.put_nowait((processed_action, inference_start_time, request_id))
                except queue.Full:
                    try:
                        discarded = result_queue.get_nowait()
                        _STARTUP_TRACE.record('response_discarded', request_id=discarded[2], reason='result_queue_replaced')
                        result_queue.put_nowait((processed_action, inference_start_time, request_id))
                    except queue.Empty:
                        result_queue.put_nowait((processed_action, inference_start_time, request_id))
            finally:
                request_context.set(None)
                busy_event.clear()
        except Exception as e:
            _STARTUP_TRACE.record('inference_error', error=repr(e))
            print(f"Error in inference worker thread: {e}")
            import traceback

            traceback.print_exc()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _compute_closed_hand_joints(side: str) -> np.ndarray:
    """Compute closed hand joint positions using G1GripperInverseKinematicsSolver."""
    side_str = "left" if side.upper() == "L" else "right"
    solver = G1GripperInverseKinematicsSolver(side=side_str)
    return solver._get_middle_close_q_desired().astype(np.float32)


def main(config: InferenceConfig):
    initial_pose_transition = InitialPoseTransition(
        LATENT_INITIAL_MOTION_TOKEN, config.initial_pose_blend_duration, config.action_publish_rate
    )
    initial_pose_ready = False
    initial_pose_just_started = False
    minimum_inference_start_time = 0.0
    global _STARTUP_TRACE
    _STARTUP_TRACE = from_env('groot-client', metadata={'config': vars(config), 'generation_policy': 'discard inference started before initial-pose handoff or controller start/stop'})
    trace = _STARTUP_TRACE
    diagnostic_generation = 0
    chunk_request_id = None
    last_publish_ns = None
    pause_loop = True

    robot_model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")

    # Isaac-GR00T PolicyClient
    from gr00t.policy.server_client import PolicyClient

    n1_policy = PolicyClient(host=config.host, port=config.port)

    print(f"Connecting to PolicyServer at {config.host}:{config.port}...")
    if n1_policy.ping():
        print_green("PolicyServer is reachable.")
    else:
        print("WARNING: PolicyServer not reachable. Inference will fail until server is up.")

    state_subscriber = ZMQStateSubscriber(
        host=config.state_zmq_host,
        port=config.state_zmq_port,
    )

    # Dedicated main-thread subscriber: do not share a ZMQ socket with the
    # inference worker when taking the initialization snapshot.
    initial_pose_state_subscriber = ZMQStateSubscriber(
        host=config.state_zmq_host, port=config.state_zmq_port
    )

    camera_subscriber = ComposedCameraClientSensor(
        server_ip=config.camera_host, port=config.camera_port
    )

    zmq_context = zmq.Context()
    zmq_socket = zmq_context.socket(zmq.PUB)
    zmq_socket.bind(f"tcp://{config.action_zmq_host}:{config.action_zmq_port}")
    time.sleep(0.1)
    print_green(
        f"ZMQ action socket bound to tcp://{config.action_zmq_host}:{config.action_zmq_port}"
    )
    print_green(f"Using embodiment tag: {config.embodiment_tag}")

    keyboard_listener = ZMQKeyboardSubscriber(
        port=config.keyboard_zmq_port, host=config.keyboard_zmq_host
    )

    telemetry = Telemetry(window_size=100)

    loop_rate = config.action_publish_rate
    loop_period = 1.0 / loop_rate

    # Track C++ control loop state
    cpp_loop_running = False
    cpp_mode = "OFF"  # "OFF", "PLANNER", or "POSE"

    # Track initial pose hand states
    initial_pose_left_hand_closed = False
    initial_pose_right_hand_closed = False

    def publish_initial_pose_step():
        """Publish one blend sample so keyboard input stays responsive."""
        nonlocal last_sent_motion_token, zmq_frame_counter, initial_pose_ready
        nonlocal cached_action_chunk, minimum_inference_start_time
        token, done = initial_pose_transition.next_token()
        left_hand = (
            _compute_closed_hand_joints("L")
            if initial_pose_left_hand_closed else np.zeros(7, dtype=np.float32)
        )
        right_hand = (
            _compute_closed_hand_joints("R")
            if initial_pose_right_hand_closed else np.zeros(7, dtype=np.float32)
        )
        zmq_message = pack_latent_action_message(
            motion_token=token,
            frame_index=np.array([zmq_frame_counter], dtype=np.int64),
            left_hand_joints=left_hand,
            right_hand_joints=right_hand,
        )
        zmq_socket.send(zmq_message)
        trace.record('initial_pose_publish', frame_id=zmq_frame_counter, token=token,
                     mode='initial_blend', blend_step=initial_pose_transition.step,
                     generation=diagnostic_generation)
        zmq_frame_counter += 1
        last_sent_motion_token = token.copy()
        if done:
            initial_pose_ready = True
            cached_action_chunk = None
            minimum_inference_start_time = time.monotonic()
            print_green("Initial pose blend complete. VLA remains paused; press 'p' to resume.")

    def send_cpp_control_command(start: bool, planner: bool = False):
        """Send C++ control loop start/stop commands via ZMQ."""
        nonlocal cpp_loop_running, cpp_mode
        try:
            cmd_msg = build_command_message(start=start, stop=not start, planner=planner)
            zmq_socket.send(cmd_msg)
            time.sleep(0.01)
            action_str = "start" if start else "stop"
            mode_str = "planner" if planner else "pose"
            cpp_loop_running = start
            if start:
                cpp_mode = "PLANNER" if planner else "POSE"
            else:
                cpp_mode = "OFF"
            print_green(f"Sent ZMQ command: {action_str} control loop ({mode_str} mode)")
            return True
        except Exception as e:
            action_str = "start" if start else "stop"
            print(f"Warning: Failed to send {action_str} command message: {e}")
            return False

    # Async inference state
    cached_action_chunk = None
    action_chunk_index = 0
    last_inference_time = 0.0
    inference_interval = 1.0 / config.rate

    zmq_frame_counter = 0
    last_sent_motion_token: np.ndarray | None = None

    PROMPT_MSG_PREFIX = "prompt:"

    def check_keyboard_input():
        nonlocal pause_loop, cpp_loop_running, cpp_mode, diagnostic_generation
        nonlocal initial_pose_left_hand_closed, initial_pose_right_hand_closed
        nonlocal cached_action_chunk, action_chunk_index, last_inference_time
        nonlocal zmq_frame_counter, last_sent_motion_token
        nonlocal initial_pose_ready, minimum_inference_start_time, initial_pose_just_started

        key = keyboard_listener.read_msg()
        if key is None:
            return

        before = dict(paused=pause_loop, running=cpp_loop_running, mode=cpp_mode, generation=diagnostic_generation)
        diagnostic_generation += 1
        trace.record('key_received', key=key, before=before)
        if key.startswith(PROMPT_MSG_PREFIX):
            new_prompt = key[len(PROMPT_MSG_PREFIX):]
            if new_prompt:
                old_prompt = language_prompt_ref[0]
                language_prompt_ref[0] = new_prompt
                print_green(f'Inference prompt changed: "{old_prompt}" -> "{new_prompt}"')
            else:
                print("Received empty prompt change -- ignoring.")
            return

        if key == "c":
            print("Keyboard: 'c' (start recording -- handled by data exporter)")
        elif key == "s":
            print("Keyboard: 's' (stop recording success -- handled by data exporter)")
        elif key == "f":
            print("Keyboard: 'f' (stop recording failure -- handled by data exporter)")
        elif key == "i":
            pause_loop = True
            if not cpp_loop_running:
                print("Initial pose not started: press 'k' to start the C++ loop first.")
                return
            try:
                initial_pose_transition.start(initial_pose_state_subscriber.get_msg())
            except ValueError as error:
                print(f"Initial pose not started: {error}")
                return
            initial_pose_ready = False
            cached_action_chunk = None
            action_chunk_index = 0
            minimum_inference_start_time = time.monotonic()
            # Prime the pose stream with the CURRENT latent before switching
            # away from the planner, never with the final raised-arm target.
            publish_initial_pose_step()
            initial_pose_just_started = True
            if cpp_mode == "PLANNER":
                if not send_cpp_control_command(start=True, planner=False):
                    initial_pose_transition.cancel()
                    print("Initial pose cancelled: could not switch to POSE mode.")
                    return
            print(f"Blending current controller latent to initial pose over "
                  f"{config.initial_pose_blend_duration:.2f}s; VLA paused.")
        elif key == "p":
            if initial_pose_transition.active:
                print("Initial pose is still blending; wait for completion, then press 'p'.")
                return
            if not cpp_loop_running or not initial_pose_ready:
                print("Complete 'k' then 'i' initialization before resuming VLA.")
                return
            pause_loop = not pause_loop
            # A pause/resume is also a model handoff boundary. Never replay a
            # cached chunk from the previous server after terminal 1 restarts.
            cached_action_chunk = None
            action_chunk_index = 0
            last_inference_time = 0.0
            minimum_inference_start_time = time.monotonic()
            print(f"{'Paused' if pause_loop else 'Resumed'} policy loop")
            if pause_loop:
                print("Policy loop paused (C++ loop still running - press 'k' to stop)")
            else:
                print("Policy loop resumed")
        elif key == "k":
            pause_loop = True
            initial_pose_transition.cancel()
            initial_pose_just_started = False
            initial_pose_ready = False
            cached_action_chunk = None
            minimum_inference_start_time = time.monotonic()
            # g1_debug with a current token is only emitted by the C++
            # CONTROL state. Recover activity after a client-only restart;
            # a local False must not turn a stop into a PLANNER/start command.
            controller_state = initial_pose_state_subscriber.get_msg()
            live_controller = False
            if controller_state is not None:
                stamp = controller_state.get("state_monotonic_ns")
                token = np.asarray(controller_state.get("token_state"), dtype=np.float32)
                live_controller = (
                    isinstance(stamp, int)
                    and 0 <= time.monotonic_ns() - stamp <= 250_000_000
                    and token.shape == (64,)
                    and np.isfinite(token).all()
                )
            if cpp_loop_running or live_controller:
                current_planner = cpp_mode == "PLANNER"
                print(f"Stopping C++ control loop (from {cpp_mode} mode)...")
                if send_cpp_control_command(start=False, planner=current_planner):
                    print("Stopped C++ control loop")
            else:
                print("Starting C++ control loop in PLANNER mode...")
                if send_cpp_control_command(start=True, planner=True):
                    print("Started C++ control loop in PLANNER mode")
                    print("Press 'i' to send initial pose and switch to POSE mode")
                    if pause_loop:
                        print("Note: Policy loop is paused - press 'p' to resume")
        elif key == "[":
            initial_pose_left_hand_closed = not initial_pose_left_hand_closed
            print(
                f"Initial pose left hand: {'closed' if initial_pose_left_hand_closed else 'open'}"
            )
        elif key == "]":
            initial_pose_right_hand_closed = not initial_pose_right_hand_closed
            print(
                f"Initial pose right hand: "
                f"{'closed' if initial_pose_right_hand_closed else 'open'}"
            )

        trace.record('control_change', key=key, before=before, after=dict(paused=pause_loop, running=cpp_loop_running, mode=cpp_mode, generation=diagnostic_generation))

    # Mutable prompt container (single-writer from keyboard, single-reader from inference)
    language_prompt_ref: list[str] = [config.prompt]
    print(f"Starting the policy loop with language prompt: {language_prompt_ref[0]}")

    inference_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    inference_stop_event = threading.Event()
    inference_busy_event = threading.Event()

    inference_worker_thread = threading.Thread(
        target=_inference_worker_loop,
        args=(
            inference_queue,
            result_queue,
            inference_stop_event,
            inference_busy_event,
            lambda: prepare_observation_from_sensors(
                camera_subscriber=camera_subscriber,
                state_subscriber=state_subscriber,
                robot_model=robot_model,
                language_prompt=language_prompt_ref[0],
                log_errors=True,
            ),
            lambda obs: run_policy_inference_and_process(
                policy=n1_policy,
                observation=obs,
                robot_model=robot_model,
            ),
        ),
        daemon=True,
    )
    inference_worker_thread.start()

    try:
        while True:
            t_start = time.monotonic()
            check_keyboard_input()

            # Consume result first so last_inference_time is fresh before trigger check
            try:
                processed_action, inference_start_time, chunk_request_id = result_queue.get_nowait()
                if inference_start_time < minimum_inference_start_time:
                    trace.record('response_discarded', request_id=chunk_request_id,
                                 reason='before_initial_pose_handoff')
                elif processed_action is None:
                    pause_loop = True
                    cached_action_chunk = None
                    action_chunk_index = 0
                    last_inference_time = 0.0
                    trace.record('policy_unavailable', request_id=chunk_request_id)
                    print("Policy request failed: paused, cached actions cleared; retrying server.")
                else:
                    inference_delay = time.monotonic() - inference_start_time
                    action_chunk_index = calculate_latency_compensated_index(
                        inference_delay, config.action_publish_rate, config.action_horizon
                    )
                    trace.record('response_received', request_id=chunk_request_id, accepted=True, chunk_index=action_chunk_index, paused=pause_loop, generation=diagnostic_generation)
                    cached_action_chunk = processed_action
                    last_inference_time = time.monotonic()
                    print_green(
                        f'New action chunk (prompt: "{language_prompt_ref[0]}", '
                        f"latency: {inference_delay:.3f}s)"
                    )
            except queue.Empty:
                pass

            worker_is_busy = inference_busy_event.is_set()
            should_start = should_trigger_new_inference(
                cached_chunk_exists=(cached_action_chunk is not None),
                inference_thread_running=worker_is_busy,
                time_since_last_inference=(time.monotonic() - last_inference_time),
                inference_interval=inference_interval,
            )

            if should_start and not initial_pose_transition.active:
                try:
                    inference_queue.put_nowait(None)
                    trace.record('request_trigger', paused=pause_loop, generation=diagnostic_generation)
                except queue.Full:
                    pass

            if initial_pose_transition.active:
                if not initial_pose_just_started:
                    publish_initial_pose_step()
                initial_pose_just_started = False
                _sleep_remaining(t_start, loop_period)
                continue

            if pause_loop:
                trace.record('no_publish', reason='paused')
                print("Pausing...", end="", flush=True)
                time.sleep(0.2)
                print(".", end="", flush=True)
                continue

            with telemetry.timer("total_loop"):
                if cached_action_chunk is None:
                    trace.record('no_publish', reason='no_chunk')
                    print("[DEBUG] No cached chunk yet, waiting...", flush=True)
                    _sleep_remaining(t_start, loop_period)
                    continue

                processed_action = cached_action_chunk

                if processed_action is None or not processed_action:
                    print("[DEBUG] processed_action is None or empty, skipping", flush=True)
                else:
                    motion_token = np.asarray(
                        get_action_field(processed_action, "motion_token"),
                        dtype=np.float32,
                    )
                    left_hand_joints = np.asarray(
                        get_action_field(processed_action, "left_hand_joints"),
                        dtype=np.float32,
                    )
                    right_hand_joints = np.asarray(
                        get_action_field(processed_action, "right_hand_joints"),
                        dtype=np.float32,
                    )

                    # Action arrays arrive as (B, T, D) from the model.
                    # Squeeze batch dim to get (T, D), then index by time step.
                    if motion_token.ndim == 3:
                        motion_token = motion_token[0]
                    if left_hand_joints.ndim == 3:
                        left_hand_joints = left_hand_joints[0]
                    if right_hand_joints.ndim == 3:
                        right_hand_joints = right_hand_joints[0]

                    horizon = motion_token.shape[0] if motion_token.ndim == 2 else 1
                    current_idx = min(action_chunk_index, horizon - 1)

                    if motion_token.ndim == 2:
                        motion_token = motion_token[current_idx]
                    if left_hand_joints.ndim == 2:
                        left_hand_joints = left_hand_joints[current_idx]
                    if right_hand_joints.ndim == 2:
                        right_hand_joints = right_hand_joints[current_idx]

                    frame_index = np.array([zmq_frame_counter], dtype=np.int64)
                    zmq_frame_counter += 1

                    zmq_message = pack_latent_action_message(
                        motion_token,
                        frame_index,
                        left_hand_joints=left_hand_joints,
                        right_hand_joints=right_hand_joints,
                    )
                    send_start_ns = time.monotonic_ns()
                    zmq_socket.send(zmq_message)
                    trace.record('action_publish', request_id=chunk_request_id, chunk_index=current_idx, frame_id=int(frame_index[0]), token=motion_token, generation=diagnostic_generation, active=not pause_loop, mode=cpp_mode, send_start_ns=send_start_ns, send_return_ns=time.monotonic_ns(), interval_ns=None if last_publish_ns is None else send_start_ns-last_publish_ns)
                    last_publish_ns = send_start_ns
                    last_sent_motion_token = motion_token.copy()
                    if zmq_frame_counter % 50 == 0:
                        print_green(
                            f"ZMQ: Sent latent action - "
                            f"frame: {frame_index[0]}, "
                            f"token shape: {motion_token.shape}"
                        )

                action_chunk_index = min(action_chunk_index + 1, config.action_horizon - 1)

            end_time = time.monotonic()
            trace.record('loop_timing', started=t_start, elapsed=end_time-t_start, target_period=loop_period)

            if config.verbose_timing:
                telemetry.log_timing_info(context="VLA Inference Loop", threshold=0.0)
            elif (end_time - t_start) > (1 / config.rate):
                telemetry.log_timing_info(
                    context="VLA Inference Loop Missed", threshold=0.001
                )

            _sleep_remaining(t_start, loop_period)

    except KeyboardInterrupt:
        print("VLA inference loop terminated by user")

    finally:
        inference_stop_event.set()
        inference_worker_thread.join(timeout=1.0)
        zmq_socket.close()
        zmq_context.term()
        state_subscriber.close()
        initial_pose_state_subscriber.close()
        keyboard_listener.close()
        trace.close()
        print("Shutdown complete.")


def _sleep_remaining(t_start: float, loop_period: float):
    """Sleep for the remainder of the loop period."""
    elapsed = time.monotonic() - t_start
    remaining = loop_period - elapsed
    if remaining > 0:
        time.sleep(remaining)


if __name__ == "__main__":
    config = tyro.cli(InferenceConfig)
    main(config)
