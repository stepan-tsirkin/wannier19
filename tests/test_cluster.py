import os
import pytest

"""
Test creation of submission scripts for slurm and PBS batch systems.
No jobs are submitted in the test.
"""

@pytest.fixture(scope="module")
def check_command_output():
    from sys import version_info
    assert version_info.major == 3
    if version_info.minor>=7:
        from subprocess import run
        def _inner(command):
            sp = run(command,capture_output=True)
            return str(sp.stdout)
    else:
        from subprocess import Popen,PIPE,STDOUT
        def _inner(command):
            sp = Popen(command, stdout=PIPE, stderr=STDOUT,encoding='UTF-8')
            return sp.stdout.read()
    return _inner

def compare_texts(script_text, ref_text, variable_strings):
    for l1,l2 in zip(script_text,ref_text):
        if l1 !=l2 :
            try:
                match=False
                for l in variable_strings:
                    if l in l1:
                        assert l1[:len(l)]==l2[:len(l)]
                        match=True
                        break
                if not match:
                    raise AssertionError()
            except AssertionError:
                raise AssertionError(f"Lines \n<{l1}>\n and \n<{l2}>\n do not match")


@pytest.mark.parametrize("cluster_type", ["slurm", "pbs"])
def test_cluster_script(cluster_type, rootdir, output_dir, check_command_output):
    if cluster_type == "slurm":
        variable_strings = ["#SBATCH --job-name=my_first_job_","#SBATCH --output=my_first_job_"]
    elif cluster_type == "pbs":
        variable_strings = ["#PBS -N my_first_job_", "#PBS -o my_first_job_", "#PBS -e my_first_job_"]
    else:
        raise ValueError("cluster_type not identified. Only slurm or pbs.")

    command=['python3.8',
         '-m',
         'wannierberri.cluster',
         '--batch-system',
         cluster_type,
         '--exp-name',
         'my_first_job_'+cluster_type,
         '--num-nodes',
         '4',
         '--partition',
         'express',
         '--command',
         "'python -u wb-example.py'",
         '--num-gpus=3',
         '--sleep-head',
         '12.34',
         '--sleep-worker=5',
         '--no-submit']
    stdout=check_command_output(command)
    print (stdout)
    script_name=stdout.split("'")[-2].split()[-1]
    print (script_name)
    script_text = open(script_name,"r").readlines()
    ref_text    = open(os.path.join(rootdir,"reference",f"my_first_job_{cluster_type}.sh"),"r").readlines()
    compare_texts(script_text, ref_text, variable_strings)
    os.replace(script_name, os.path.join(output_dir, script_name))
