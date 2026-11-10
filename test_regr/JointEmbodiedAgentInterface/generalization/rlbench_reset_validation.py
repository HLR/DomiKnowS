"""Validate physical/scoring reset state without gating on unused task dummies."""


def validate_reset(expected, restored, *, robot_names, robot_base, task_base,
                   dummy_names, pose_error, physical_tolerance=1e-5,
                   passive_tolerance=.002):
    required={row['name'] for row in expected}
    if required-set(restored):raise ValueError('frozen reset objects missing')
    bases={robot_base,task_base}
    unused=set(dummy_names)-bases-set(robot_names)
    physical=max((pose_error(row['pose'],restored[row['name']]['pose']) for row in expected
                  if row['name'] not in unused and
                  (row['name'] not in robot_names or row['name']==robot_base)),default=0.)
    joint=max((abs(row['joint']-restored[row['name']]['joint']) for row in expected if 'joint' in row),default=0.)
    passive=max((pose_error(row['pose'],restored[row['name']]['pose']) for row in expected
                 if row['name'] in robot_names),default=0.)
    helpers=max((pose_error(row['pose'],restored[row['name']]['pose']) for row in expected
                 if row['name'] in unused),default=0.)
    error=max(physical,joint)
    if error>physical_tolerance:raise ValueError('frozen physical/scoring reset differs: '+str(error))
    if passive>passive_tolerance:raise ValueError('passive robot reset differs: '+str(passive))
    return {'reset_pose_max_error':error,'passive_robot_pose_error':passive,
            'unused_helper_pose_error':helpers}
