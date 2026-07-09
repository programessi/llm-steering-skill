You are generating a restricted Python driving policy for Stage 1.

Define only:

```python
def policy():
    ...
```

The policy may call only these environment APIs. They are already imported by
the runtime; treat this list as capability descriptions, not implementation
source code:

```python
observe_driving_state()
observe_execution_feedback()
task_finished()
mark_task_complete()
execute_primitive(name, target_angle_rad, duration_s, speed_mps, stage="...", trigger="...")
```

`execute_primitive(...)` is the only action API. It calls the current
`SteeringSkill` implementation and can later be backed by an imitation-learned
robot steering skill.

Do not access images, simulator internals, robot joints, files, network,
imports, helper functions, or raw vehicle control.

The policy should loop until `task_finished()` and use feedback events to recover from mistakes.
