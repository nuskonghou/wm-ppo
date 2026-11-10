"""Locations used by the code. Every value can be overridden with an environment variable.

  WMPPO_MODEL           base model directory              default <repo>/models/Qwen3-4B-Instruct-2507
  ALFWORLD_DATA         ALFWorld data (holds json_2.1.1/)  default <repo>/data/alfworld
  WMPPO_TRAINER_IMAGE   Docker image of the trainers and the inference servers (docker/Dockerfile.trainer)
                                                          default wmppo-trainer
  WMPPO_TRAINER_VENV    Python environment inside that image  default /opt/trainvenv
  WMPPO_SOKOBAN_WARMUP  run directory of the Sokoban warm-up  default <repo>/sokoban/runs/ppo
  WMPPO_TEACHER_MODEL   Qwen3-32B, the teacher of the OCSD / OPD (32B) baselines  default <base model dir>/../Qwen3-32B
  WMPPO_MOUNTS          extra host directories (colon-separated) to mount into the trainer containers

The trainers run in containers that see the host directories at the same absolute paths (docker_mounts)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get('WMPPO_MODEL', os.path.join(ROOT, 'models', 'Qwen3-4B-Instruct-2507'))
ALFWORLD_DATA = os.environ.get('ALFWORLD_DATA', os.path.join(ROOT, 'data', 'alfworld'))
TRAINER_IMAGE = os.environ.get('WMPPO_TRAINER_IMAGE', 'wmppo-trainer')
TRAINER_VENV = os.environ.get('WMPPO_TRAINER_VENV', '/opt/trainvenv')
SOKOBAN_WARMUP = os.environ.get('WMPPO_SOKOBAN_WARMUP', os.path.join(ROOT, 'sokoban', 'runs', 'ppo'))
TEACHER_MODEL = os.environ.get('WMPPO_TEACHER_MODEL', os.path.join(os.path.dirname(MODEL), 'Qwen3-32B'))


def docker_mounts():
    """'-v d:d' options for the repository, the base and teacher models, the run directories (also when they are
    symbolic links to another disk) and WMPPO_MOUNTS."""
    dirs = [ROOT, os.path.dirname(MODEL), TEACHER_MODEL]
    dirs += [os.path.join(ROOT, env, 'runs') for env in ('alfworld', 'scienceworld', 'sokoban')]
    dirs += [d for d in os.environ.get('WMPPO_MOUNTS', '').split(':') if d]
    out = []
    for d in dirs:
        for p in (os.path.abspath(d), os.path.realpath(d)):
            if os.path.exists(p) and p not in out and not any(p.startswith(x.rstrip(os.sep) + os.sep) for x in out):
                out.append(p)
    return ' '.join(f'-v {p}:{p}' for p in out)
