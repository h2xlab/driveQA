# Generating Intersection VQA with CARLA

To generate QA pairs for intersection VQA tasks with CARLA, use the scripts `data/generate_T_intersection_QA_CARLA.py` and `data/generate_cross_intersection_QA_CARLA.py`. We use [CARLA 0.9.10.1](https://github.com/carla-simulator/carla/releases/tag/0.9.10.1).

First launch CARLA:
```bash
DISPLAY= ./CarlaUE4.sh -opengl
```

Then run the data generation scripts:

```bash
python data/generate_T_intersection_QA_CARLA.py
python data/generate_cross_intersection_QA_CARLA.py
```
