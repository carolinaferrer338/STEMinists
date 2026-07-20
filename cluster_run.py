import subprocess
import shlex
import multiprocessing as mp

def run_job(job_str):
  cmd_args = shlex.split(job_str)
  subprocess.run(cmd_args)

jobs = []

for comp_alpha in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]: #competitiveness
  for cut_alpha in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]: #cut edges
      for county_alpha in [0, 0.2, 0.4, 0.6, 0.8, 1.0]: # county
        jobs.append(f"python ./PA_ensemble_write_to_file.py 'tn' {comp_alpha} {cut_alpha} {county_alpha}")


for state in ['co', 'ut', 'pa', 'ma']:
   for comp_alpha in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]: #competitiveness
     for cut_alpha in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]: #cut edges
       for county_alpha in [0, 0.2, 0.4, 0.6, 0.8]: # county
           jobs.append(f"python ./PA_ensemble_write_to_file.py {state} {comp_alpha} {cut_alpha} {county_alpha}")

pool = mp.Pool(len(jobs))
pool.map(run_job, jobs)
