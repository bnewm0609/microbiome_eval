# This script is used to submit jobs to hyak cluster using slurm.
# Please change the slurm template as appropriate!

template_header = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --account=xlab
#SBATCH --nodes=1
#SBATCH --mem={memory}
#SBATCH --time={time}
#SBATCH --output={out_file_path}-%j.out
{additional_args}

"""

apptainer_template = template_header + """
apptainer run --nv --overlay /gscratch/xlab/blnewman/apptainer/conda-overlay-jupyter.img:ro /gscratch/xlab/blnewman/apptainer/hyak-container-3.sif "{command}"
"""

raw_template = template_header + """
source ~/.bashrc
echo `date +%Y%m%d-%H%M%S`

cd /gscratch/xlab/blnewman/microbiome_eval

{command}
"""


from argparse import ArgumentParser
import os
import time as pytime
import tempfile
import re
import subprocess


def submit_job(
    job_name,
    command,
    partition="gpu-rtx6k",
    time="9:00:00",
    #time="7:00:00",
    dependency=None,
    mem="64G",
    ngpu="1",
    out_dir=None,
    enable_apptainer=False,
    signal=None,
    ncpu="4",
):
    additional_args = []
    if "ckpt" in partition:
        partition, gpu = partition.split("-", 1)
        partition="ckpt-all"
        num_gpus = ngpu
        if gpu == "any":
            additional_args.append(f"#SBATCH --gpus={num_gpus}")
            additional_args.append("#SBATCH --exclude=z[3001-3002,3005-3006],g[3001-3007,3014-3017,3027]")  # should exclude small gpus
        else:
            additional_args.append(f"#SBATCH --gpus={gpu}:{num_gpus}")
    else:
        partition = partition
        num_gpus = ngpu
        additional_args.append(f"#SBATCH --gres=gpu:{num_gpus}")

    if dependency is not None:
        additional_args.append(f"#SBATCH --dependency={dependency}")
    if signal is not None:
        additional_args.append(f"#SBATCH --signal={signal}")

    additional_args.append(f"#SBATCH --cpus-per-task={ncpu}") # 4 by default


    if out_dir is not None:
        if out_dir == "__script__":
            out_dir = os.path.dirname(command)
        out_file_path = os.path.join(out_dir, job_name)
    else:
        out_file_path = job_name
    
    if enable_apptainer:
        template = apptainer_template
    else:
        template = raw_template

    filled_template = template.format(
        job_name=job_name,
        command=command,
        partition=partition,
        out_file_path=out_file_path,
        additional_args="\n".join(additional_args),
        time=time,
        memory=mem,
    )
    #print(filled_template)

    submitted_job_id = None
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as fp:

        fp.write(filled_template)
        fp.close()
        result = subprocess.run(["sbatch", fp.name], capture_output=True)
        # result = os.system(f"sbatch {fp.name}")
        print("Closing up with result:", result)
        match = re.search("\d+", result.stdout.decode())
        if match:
            submitted_job_id = match[0]

    # for the output file, put a symlink in the home dir if the output_dir is specified.
    # Having a symlink to the output file in the home dir makes it easier to track what's running.
    # the output file only exists once the job is in the RUNNING state, so we have to wait
    print(f"Output at:\n{out_file_path}-{submitted_job_id}.out")
    # if out_dir is not None and dependency is None:
    #     num_retries = 11
    #     for i in range(num_retries):
    #         pytime.sleep(2 ** (i))  # wait for job to start with exponential backoff
    #         # get the job id and create a symlink
    #         cmd = [
    #             "squeue",
    #             "--user",
    #             "blnewman",
    #             "--states",
    #             "RUNNING",
    #             "--name",
    #             job_name,
    #             "--Format",
    #             "JobId",
    #             "--noheader",
    #         ]
    #         job_id = subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode("utf-8").strip()
    #         if job_id:
    #             print("making symlink")
    #             print(f"{out_file_path}-{job_id}.out", f"{job_name}-{job_id}.out")
    #             os.symlink(f"{out_file_path}-{job_id}.out", f"{job_name}-{job_id}.out")
    #             break
    #         print("Waiting for job to start running...")

    return submitted_job_id


def main():
    import sys;
    print(sys.argv)
    argp = ArgumentParser()
    argp.add_argument("job_name", type=str)
    argp.add_argument("command", type=str)
    argp.add_argument(
        "--partition",
        type=str,
        choices=["gpu-a100", "gpu-rtx6k", "ckpt-a40", "ckpt-a100", "ckpt-rtx6k", "ckpt-l40s", "ckpt-l40", "ckpt-any"],
        default="gpu-a100",
    )
    argp.add_argument("--time", type=str, default="24:00:00")
    argp.add_argument("--dependency", type=str)
    argp.add_argument("--mem", type=str, default="64G")
    argp.add_argument("--ngpu", type=str, default="1")
    argp.add_argument("--ncpu", type=str, default="4")
    argp.add_argument(
        "--out_dir",
        type=str,
        help="Where to place the output file. (Makes a symlink to that file in the home directory as well)",
    )
    argp.add_argument("--enable_apptainer", action="store_true", help="If set, uses apptainer [this is legacy and should basically never happen]")
    argp.add_argument("--signal", type=str, default=None, help="If set, will send this signal to the job after a certain amount of time to trigger cleanup. (Only works if the job script is set up to handle the signal) [This is legacy and should basically never happen]")
    args = argp.parse_args()
    args_dict = vars(args)
    submit_job(**args_dict)


# squeue --user blnewman --states RUNNING --name klone-container --Format NodeList --noheader
if __name__ == "__main__":
    main()


# uv run -- python submit_job.py "llm_judge_Qwen3-32B" "uv run -- vllm serve Qwen/Qwen3-8B --tensor-parallel-size 4 --host 0.0.0.0 --api-key synthesis_rc --port 37140 --max-model-len 36032 --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser deepseek_r1 --enable-prefix-caching --log-error-stack" --ngpu 4 --partition gpu-a100 --time 24:00:00 --mem 128G --out_dir /gscratch/xlab/blnewman/vllm_logs/