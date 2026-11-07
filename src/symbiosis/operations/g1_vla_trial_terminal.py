"""Read-only startup telemetry validation; no keyboard/robot I/O."""
TEMPERATURE_ALERT_C = 130

def health_reason(health, now):
    if not isinstance(health, dict):
        return 'invalid_motor_telemetry'
    stamp = health.get('monotonic')
    if type(stamp) not in (int, float) or not 0 <= now - stamp < 1:
        return 'missing_or_stale_motor_telemetry'
    motors = health.get('motors')
    if not isinstance(motors, list) or len(motors) != 29:
        return 'incomplete_motor_telemetry'
    try:
        for motor in motors:
            temperatures = motor['temperature_case_winding_c']
            if (len(temperatures) != 2 or
                    any(type(t) not in (int, float) or not -40 <= t <= 200 for t in temperatures)):
                return 'invalid_motor_temperature'
            if motor['motorstate'] != 0:
                return 'motor_fault'
            if max(temperatures) >= TEMPERATURE_ALERT_C:
                return f'temperature_alert_{TEMPERATURE_ALERT_C}C'
    except (KeyError, TypeError):
        return 'invalid_motor_telemetry'
    return None
