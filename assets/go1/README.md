# Model and Scene of Unitree Go1

This is from the [official unitree model repo](https://github.com/unitreerobotics/unitree_mujoco), with a few modifications:

1. Separate the model and the scene with the floor. This makes it convenient for other composition.
   - Move the `<visual>` tag to `scene.xml`.
   - Move the floor to `scene.xml`.
2. Remove the tracking camera. The user can choose to attach their own camera.

The license is replicated and retainted as BSD-3.

## Example Usage

For example, to load the Mujoco model of the go1 on the floor:

```python
from hobot.environment.mj_models.go1 import go1

go1.load_scene()
```
