import glob
import os

files = glob.glob('**/*.py', recursive=True)
for f in files:
    if '.venv' in f: continue
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    new_content = content.replace('maneuver, target_speed = planner.plan(', 'maneuver, target_speed = planner.plan(')
    new_content = new_content.replace("controller.set_maneuver(maneuver, ego_state_now['y'], ego_state_now['vx'], target_speed=target_speed)", "controller.set_maneuver(maneuver, ego_state_now['y'], ego_state_now['vx'], target_speed=target_speed)")
    new_content = new_content.replace("controller.set_maneuver(maneuver, ego_state['y'], ego_state['vx'], target_speed=target_speed)", "controller.set_maneuver(maneuver, ego_state['y'], ego_state['vx'], target_speed=target_speed)")
    
    if 'test_binding_constraints.py' in f:
        new_content = new_content.replace('action = planner.plan(', 'action, _ = planner.plan(')
        
    if content != new_content:
        with open(f, 'w', encoding='utf-8') as file:
            file.write(new_content)
        print(f'Updated {f}')
